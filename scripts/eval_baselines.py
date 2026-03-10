#!/usr/bin/env python3
"""Evaluate HuggingFace baselines (GPT-2 Medium, OPT-350M) on WikiText-103.

Computes PPL, ECE, MCE, Brier Score for comparison with AletheionV2.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_wikitext():
    """Download WikiText-103 test set, return token IDs (GPT-2 tokenizer)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join([t for t in ds["text"] if t.strip()])
    return text


def compute_ece(confidences, corrects, n_bins=15):
    """Compute ECE, MCE, Brier Score from arrays."""
    confidences = np.array(confidences)
    corrects = np.array(corrects)
    n = len(confidences)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    mce_bin = 0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = mask | (confidences == hi)
        count = mask.sum()

        if count > 0:
            acc = corrects[mask].mean()
            conf = confidences[mask].mean()
            gap = abs(acc - conf)
            ece += (count / n) * gap
            if gap > mce:
                mce = gap
                mce_bin = i
        else:
            acc = 0.0
            conf = (lo + hi) / 2
            gap = 0.0

        bin_data.append({
            "bin": i, "lo": round(lo, 4), "hi": round(hi, 4),
            "count": int(count), "acc": round(float(acc), 4),
            "conf": round(float(conf), 4), "gap": round(float(gap), 4),
        })

    brier = ((confidences - corrects) ** 2).mean()
    overconf = (confidences > corrects).mean()

    return {
        "ece": float(ece),
        "mce": float(mce),
        "mce_bin": mce_bin,
        "mce_bin_range": f"{bin_boundaries[mce_bin]:.2f}-{bin_boundaries[mce_bin+1]:.2f}",
        "brier_score": float(brier),
        "overconfidence_ratio": float(overconf),
        "mean_confidence": float(confidences.mean()),
        "mean_accuracy": float(corrects.mean()),
        "calibration_bins": bin_data,
    }


@torch.no_grad()
def eval_hf_model(model_name, tokens, seq_len=1024, max_seqs=2000, batch_size=2, device="cuda",
                   truncate_logits=None):
    """Evaluate a HuggingFace model using pre-tokenized GPT-2 tokens.

    All models are evaluated on the same token sequence (GPT-2 tokenizer)
    and using bfloat16 for consistency with Aletheion eval.

    For models with vocab_size > 50257 (e.g. OPT-350M has 50272),
    the GPT-2 token IDs are valid since OPT's tokenizer is a superset.
    PPL for OPT will be slightly higher than its "native" PPL because
    the tokenization is suboptimal, but ECE/Brier comparisons are fair.
    """
    print(f"\n{'='*60}")
    print(f"  AVALIANDO {model_name}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = model.config.vocab_size
    print(f"  Params: {n_params:,}")
    print(f"  Vocab: {vocab_size} (using GPT-2 tokens, vocab_size=50257)")
    print(f"  Tokens: {len(tokens):,}")

    # Verify all token IDs are within model's vocab
    max_token = tokens.max()
    if max_token >= vocab_size:
        print(f"  [WARN] Max token ID {max_token} >= vocab_size {vocab_size}, clipping")
        tokens = np.clip(tokens, 0, vocab_size - 1)

    n_seqs = min(len(tokens) // (seq_len + 1), max_seqs)
    print(f"  Sequences: {n_seqs}")

    total_loss = 0.0
    total_tokens = 0
    all_confidences = []
    all_corrects = []

    for i in range(0, n_seqs, batch_size):
        batch_seqs = min(batch_size, n_seqs - i)
        input_ids = []
        labels = []

        for j in range(batch_seqs):
            idx = (i + j) * (seq_len + 1)
            seq = tokens[idx: idx + seq_len + 1]
            input_ids.append(seq[:seq_len])
            labels.append(seq[1: seq_len + 1])

        input_ids_t = torch.tensor(np.array(input_ids), dtype=torch.long, device=device)
        labels_t = torch.tensor(np.array(labels), dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids_t)
            logits = outputs.logits

            # Truncate logits if using non-native tokenizer
            if truncate_logits is not None:
                logits = logits[:, :, :truncate_logits]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels_t.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += labels_t.numel()

            probs = F.softmax(logits, dim=-1)
            max_probs, preds = probs.max(dim=-1)

            correct = (preds == labels_t).float()
            all_confidences.extend(max_probs.cpu().float().numpy().flatten().tolist())
            all_corrects.extend(correct.cpu().float().numpy().flatten().tolist())

        if (i // batch_size) % 50 == 0 and i > 0:
            ppl_so_far = math.exp(total_loss / total_tokens)
            print(f"  [{i}/{n_seqs}] PPL={ppl_so_far:.2f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    results = compute_ece(all_confidences, all_corrects)
    results["perplexity"] = ppl
    results["ce_loss"] = avg_loss
    results["model"] = model_name
    results["n_params"] = n_params
    results["total_tokens"] = total_tokens

    print(f"  PPL:          {ppl:.2f}")
    print(f"  CE Loss:      {avg_loss:.4f}")
    print(f"  ECE:          {results['ece']:.4f}")
    print(f"  MCE:          {results['mce']:.4f}  (bin {results['mce_bin']}: {results['mce_bin_range']})")
    print(f"  Brier Score:  {results['brier_score']:.4f}")
    print(f"  Overconf:     {results['overconfidence_ratio']:.2%}")
    print(f"  Mean Conf:    {results['mean_confidence']:.4f}")
    print(f"  Mean Acc:     {results['mean_accuracy']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def print_full_comparison(aletheion_bb, aletheion_ft, gpt2, opt):
    """Print full comparison table."""
    print(f"\n{'='*80}")
    print("  COMPARACAO COMPLETA — AletheionV2 350M vs Baselines")
    print(f"{'='*80}")

    print("  Note: GPT2-Med uses same GPT-2 tokenizer (fully comparable).")
    print("  Note: OPT-350M uses native tokenizer (PPL not comparable, ECE/Brier are).\n")

    header = f"  {'Metric':<20} {'GPT2-Med':>10} {'OPT-350M':>10} {'Aleth-BB':>10} {'Aleth-FT':>10}"
    print(header)
    print("  " + "-" * 75)

    rows = [
        ("PPL*", "perplexity", ".2f"),
        ("CE Loss*", "ce_loss", ".4f"),
        ("ECE", "ece", ".4f"),
        ("MCE", "mce", ".4f"),
        ("Brier Score", "brier_score", ".4f"),
        ("Overconf", "overconfidence_ratio", ".2%"),
        ("Mean Conf", "mean_confidence", ".4f"),
        ("Mean Acc", "mean_accuracy", ".4f"),
    ]

    for label, key, fmt in rows:
        vals = []
        for r in [gpt2, opt, aletheion_bb, aletheion_ft]:
            v = r.get(key, 0)
            if fmt == ".2%":
                vals.append(f"{v:.2%}")
            else:
                vals.append(f"{v:{fmt}}")
        print(f"  {label:<20} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    print("  " + "-" * 75)
    print("  * PPL/CE not comparable across different tokenizers")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aletheion-eval", default="eval_results/finetune/eval_comparison.json")
    parser.add_argument("--output-dir", default="eval_results/finetune")
    parser.add_argument("--max-seqs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Aletheion results
    with open(args.aletheion_eval) as f:
        aletheion = json.load(f)

    aletheion_bb = aletheion["backbone"]
    aletheion_ft = aletheion["finetuned"]

    # Download WikiText and tokenize once with GPT-2 tokenizer
    print("[DATA] Baixando WikiText-103...")
    text = download_wikitext()
    print(f"[DATA] {len(text):,} chars")

    print("[DATA] Tokenizando com GPT-2 tokenizer (compartilhado por todos os modelos)...")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = gpt2_tokenizer.encode(text)
    tokens = np.array(tokens, dtype=np.int64)
    print(f"[DATA] {len(tokens):,} tokens (GPT-2)")

    # GPT-2 Medium: same tokenizer, fully comparable
    gpt2_results = eval_hf_model("gpt2-medium", tokens, max_seqs=args.max_seqs,
                                  batch_size=args.batch_size, device=args.device,
                                  truncate_logits=50257)

    # OPT-350M: different tokenizer, use native tokens (ECE/Brier still comparable)
    print("\n[DATA] Tokenizando com OPT tokenizer (nativo)...")
    opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    opt_tokens = opt_tokenizer.encode(text)
    opt_tokens = np.array(opt_tokens, dtype=np.int64)
    print(f"[DATA] {len(opt_tokens):,} tokens (OPT)")

    opt_results = eval_hf_model("facebook/opt-350m", opt_tokens, max_seqs=args.max_seqs,
                                 batch_size=args.batch_size, device=args.device)

    # Full comparison
    print_full_comparison(aletheion_bb, aletheion_ft, gpt2_results, opt_results)

    # Save all
    all_results = {
        "methodology": {
            "precision": "bfloat16 (all models)",
            "dataset": "WikiText-103 test",
            "seq_len": 1024,
            "n_bins": 15,
            "note": "GPT-2 Medium and Aletheion use GPT-2 tokenizer (50257 vocab). "
                    "OPT-350M uses native tokenizer (50272 vocab). "
                    "PPL/CE only comparable within same tokenizer. "
                    "ECE/Brier/MCE are calibration metrics and comparable across tokenizers.",
        },
        "gpt2_medium": gpt2_results,
        "opt_350m": opt_results,
        "aletheion_backbone": aletheion_bb,
        "aletheion_finetuned": aletheion_ft,
    }
    out_path = out_dir / "eval_baselines_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVE] {out_path}")

    # Plot comparison
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "text.color": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#21262d",
            "grid.alpha": 0.8,
            "font.family": "monospace",
            "font.size": 11,
        })

        models = ["GPT2-Med\n(355M)", "OPT-350M", "Aletheion\nBackbone", "Aletheion\nFine-tuned"]
        colors = ["#8b949e", "#8b949e", "#58a6ff", "#3fb950"]
        results_list = [gpt2_results, opt_results, aletheion_bb, aletheion_ft]

        metrics = [
            ("Perplexity", "perplexity", False),
            ("ECE", "ece", False),
            ("MCE", "mce", False),
            ("Brier Score", "brier_score", False),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        for ax, (label, key, _) in zip(axes, metrics):
            vals = [r[key] for r in results_list]
            bars = ax.bar(models, vals, color=colors, alpha=0.85, edgecolor="#30363d")
            for bar, val in zip(bars, vals):
                fmt = f"{val:.4f}" if val < 1 else f"{val:.1f}"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        fmt, ha="center", va="bottom", fontsize=10,
                        fontweight="bold", color="#c9d1d9")
            ax.set_title(label, fontsize=13, fontweight="bold")
            ax.grid(True, axis="y")

        fig.suptitle("AletheionV2 350M vs Baselines — WikiText-103 (OOD)",
                     fontsize=16, fontweight="bold", y=1.03)
        fig.tight_layout()
        fig.savefig(out_dir / "baselines_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {out_dir / 'baselines_comparison.png'}")

    except ImportError:
        print("[WARN] matplotlib nao disponivel")


if __name__ == "__main__":
    main()
