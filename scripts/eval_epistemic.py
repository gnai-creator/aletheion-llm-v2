"""
Avaliacao epistemica completa do AletheionV2.

Mede o delta antes/depois do fine-tuning epistemico:
  - Perplexidade (PPL)
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Brier Score
  - Metricas de tomografia (q1, q2, confidence, phi, vi_severity)

Uso:
    # Avaliar um unico checkpoint
    python scripts/eval_epistemic.py --checkpoint checkpoints/backbone/step_106000.pt

    # Comparar backbone vs fine-tuned (o resultado principal do paper)
    python scripts/eval_epistemic.py \
        --backbone checkpoints/350m_4xh100/step_106000.pt \
        --finetuned checkpoints/350m_epistemic_finetune/step_15000.pt

    # Gerar plots de calibracao
    python scripts/eval_epistemic.py --checkpoint ckpt.pt --plot

Saida:
    Tabela comparativa + reliability diagram + JSON detalhado.
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Carrega modelo de um checkpoint."""
    print(f"[LOAD] Carregando {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in ckpt:
        config = ckpt["config"]
        if isinstance(config, dict):
            config = AletheionV2Config(**config)
    else:
        config = AletheionV2Config()

    model = AletheionV2Model(config)

    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)

    model = model.to(device).eval()
    step = ckpt.get("global_step", "unknown")
    params = sum(p.numel() for p in model.parameters())
    print(f"[LOAD] {params:,} params, step={step}")
    return model, config, step


def prepare_eval_tokens(config, data_dir=None, max_tokens=2_000_000):
    """
    Prepara tokens de avaliacao.

    IMPORTANTE: Usa WikiText-103 test por default (out-of-distribution).
    O modelo foi treinado em FineWeb-Edu, entao WikiText-103 e LAMBADA
    sao datasets que o modelo NUNCA viu. Isso garante que ECE e Brier
    nao sao artificialmente otimistas.

    Se --eval-data for passado, usa o ultimo shard como held-out split
    (in-distribution). Reportar ambos e ideal para o paper.
    """
    results = {}

    # WikiText-103 test (OOD - preferido para calibracao)
    try:
        import tiktoken
        from datasets import load_dataset
        print("[DATA] Baixando WikiText-103 test (out-of-distribution)...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join([x["text"] for x in ds if x["text"].strip()])
        enc = tiktoken.get_encoding("gpt2")
        wikitext_tokens = np.array(enc.encode(text), dtype=np.int64)
        wikitext_tokens = wikitext_tokens[:max_tokens]
        print(f"[DATA] WikiText-103: {len(wikitext_tokens):,} tokens (OOD)")
        results["wikitext"] = wikitext_tokens
    except Exception as e:
        print(f"[WARN] WikiText-103 indisponivel: {e}")

    # Held-out shard (in-distribution)
    if data_dir:
        from glob import glob
        shards = sorted(glob(os.path.join(data_dir, "train_shard_*.bin")))
        if shards:
            eval_shard = shards[-1]
            print(f"[DATA] Held-out shard: {os.path.basename(eval_shard)} (in-distribution)")
            tokens = np.memmap(eval_shard, dtype=np.uint16, mode="r")
            tokens = tokens[:max_tokens].astype(np.int64)
            print(f"[DATA] Held-out: {len(tokens):,} tokens (ID)")
            results["held_out"] = tokens

    if not results:
        print("[ERRO] Sem dados de avaliacao disponveis")
        sys.exit(1)

    return results


@torch.no_grad()
def eval_full(model, tokens, seq_len=1024, batch_size=4, device="cuda",
              with_tomography=False, n_bins=15, max_seqs=2000):
    """
    Avaliacao completa: PPL + ECE + tomografia.

    ECE: Divide predicoes em bins de confianca. Para cada bin, compara
    a confianca media com a acuracia real. ECE = media ponderada dos gaps.
    """
    model.eval()

    # Acumuladores
    total_loss = 0.0
    total_tokens = 0

    # Para ECE: coleta (confidence, correct) por token
    all_confidences = []
    all_corrects = []

    # Para tomografia
    tomo_stats = {
        "q1": [], "q2": [], "confidence": [], "phi": [],
        "vi_severity": [], "temperature": [],
    }

    n_seqs = min(len(tokens) // (seq_len + 1), max_seqs)
    if n_seqs == 0:
        return {"error": "insufficient tokens"}

    print(f"[EVAL] {n_seqs} sequences, tomography={'ON' if with_tomography else 'OFF'}")

    for i in range(0, n_seqs, batch_size):
        batch_seqs = min(batch_size, n_seqs - i)
        input_ids = []
        labels = []

        for j in range(batch_seqs):
            idx = (i + j) * (seq_len + 1)
            seq = tokens[idx : idx + seq_len + 1]
            input_ids.append(seq[:seq_len])
            labels.append(seq[1 : seq_len + 1])

        input_ids = torch.tensor(np.array(input_ids), dtype=torch.long, device=device)
        labels_t = torch.tensor(np.array(labels), dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids, return_tomography=with_tomography)
            logits = output.logits  # [B, T, V]

            # Loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels_t.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += labels_t.numel()

            # ECE: confianca do softmax = max prob
            probs = F.softmax(logits, dim=-1)  # [B, T, V]
            max_probs, preds = probs.max(dim=-1)  # [B, T]
            correct = (preds == labels_t).float()  # [B, T]

            all_confidences.append(max_probs.cpu().float().reshape(-1))
            all_corrects.append(correct.cpu().float().reshape(-1))

            # Tomografia
            if with_tomography and output.tomography is not None:
                tomo = output.tomography
                for key in tomo_stats:
                    val = getattr(tomo, key, None)
                    if val is not None and isinstance(val, torch.Tensor):
                        tomo_stats[key].append(val.cpu().float().mean().item())

        if (i // batch_size) % 100 == 0 and i > 0:
            ppl_so_far = math.exp(total_loss / total_tokens)
            print(f"  [{i}/{n_seqs}] PPL={ppl_so_far:.2f}")

    # --- Calcula metricas ---
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    all_confidences = torch.cat(all_confidences)
    all_corrects = torch.cat(all_corrects)

    # ECE (Expected Calibration Error)
    ece, bin_data = compute_ece(all_confidences, all_corrects, n_bins)

    # MCE (Maximum Calibration Error) — reporta qual bin tem o pior gap
    mce = 0.0
    mce_bin = -1
    for b in bin_data:
        if b["count"] > 0 and b["gap"] > mce:
            mce = b["gap"]
            mce_bin = b["bin"]

    # Brier Score: media de (prob_correct - 1)^2 para token correto
    # Simplificado: mean((max_prob - correct)^2)
    brier = ((all_confidences - all_corrects) ** 2).mean().item()

    # Overconfidence ratio: % de tokens onde confianca > acuracia do bin
    overconf_count = 0
    underconf_count = 0
    for b in bin_data:
        if b["count"] > 0:
            if b["conf"] > b["acc"]:
                overconf_count += b["count"]
            else:
                underconf_count += b["count"]
    total_binned = overconf_count + underconf_count
    overconf_ratio = overconf_count / total_binned if total_binned > 0 else 0.0

    # Tomografia media
    tomo_means = {}
    for key, vals in tomo_stats.items():
        if vals:
            tomo_means[key] = round(float(np.mean(vals)), 4)

    results = {
        "perplexity": round(perplexity, 2),
        "ce_loss": round(avg_loss, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "mce_bin": mce_bin,
        "mce_bin_range": f"{bin_data[mce_bin]['lo']:.2f}-{bin_data[mce_bin]['hi']:.2f}" if mce_bin >= 0 else "N/A",
        "brier_score": round(brier, 4),
        "overconfidence_ratio": round(overconf_ratio, 4),
        "mean_confidence": round(all_confidences.mean().item(), 4),
        "mean_accuracy": round(all_corrects.mean().item(), 4),
        "total_tokens": total_tokens,
        "calibration_bins": bin_data,
        "tomography": tomo_means,
    }

    return results


def compute_ece(confidences, corrects, n_bins=15):
    """
    Expected Calibration Error.

    Divide predicoes em bins por confianca. Para cada bin:
      acc(b) = fracao de predicoes corretas
      conf(b) = confianca media
      ECE = sum |B_b|/n * |acc(b) - conf(b)|
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_data = []

    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum().item()

        if count > 0:
            acc = corrects[mask].mean().item()
            conf = confidences[mask].mean().item()
            ece += (count / n) * abs(acc - conf)
        else:
            acc = 0.0
            conf = (lo + hi) / 2

        bin_data.append({
            "bin": i,
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "count": int(count),
            "acc": round(acc, 4),
            "conf": round(conf, 4),
            "gap": round(abs(acc - conf), 4),
        })

    return ece, bin_data


def plot_comparison(backbone_results, finetuned_results, output_dir):
    """Gera plots comparativos backbone vs fine-tuned."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib nao disponivel, pulando plots")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    # --- 1. Reliability Diagram (side by side) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, results, title, color in [
        (ax1, backbone_results, "Backbone (CE only)", "#58a6ff"),
        (ax2, finetuned_results, "After Epistemic Fine-Tuning", "#3fb950"),
    ]:
        bins = results["calibration_bins"]
        confs = [b["conf"] for b in bins if b["count"] > 0]
        accs = [b["acc"] for b in bins if b["count"] > 0]
        counts = [b["count"] for b in bins if b["count"] > 0]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1, label="Perfect")

        # Bars
        bin_width = 1.0 / len(bins)
        for b in bins:
            if b["count"] > 0:
                mid = (b["lo"] + b["hi"]) / 2
                # Gap (error)
                if b["conf"] > b["acc"]:
                    ax.bar(mid, b["conf"] - b["acc"], bottom=b["acc"],
                           width=bin_width * 0.8, color="#f85149", alpha=0.4)
                else:
                    ax.bar(mid, b["acc"] - b["conf"], bottom=b["conf"],
                           width=bin_width * 0.8, color="#3fb950", alpha=0.4)
                # Accuracy bar
                ax.bar(mid, b["acc"], width=bin_width * 0.8, color=color, alpha=0.7)

        ece = results["ece"]
        ppl = results["perplexity"]
        ax.set_title(f"{title}\nECE={ece:.4f}  PPL={ppl:.1f}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc="upper left")

    fig.suptitle("AletheionV2 350M — Reliability Diagram", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] reliability_diagram.png")

    # --- 2. Bar chart comparison ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    metrics = [
        ("PPL", "perplexity", "lower is better"),
        ("ECE", "ece", "lower is better"),
        ("Brier", "brier_score", "lower is better"),
        ("Overconf %", "overconfidence_ratio", "lower is better"),
    ]

    for ax, (label, key, note) in zip(axes, metrics):
        bb = backbone_results.get(key, 0)
        ft = finetuned_results.get(key, 0)
        bars = ax.bar(["Backbone", "Fine-tuned"], [bb, ft],
                       color=["#58a6ff", "#3fb950"], alpha=0.8)

        # Value labels
        for bar, val in zip(bars, [bb, ft]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}" if val < 1 else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color="#c9d1d9")

        # Delta
        if bb > 0:
            delta = ((ft - bb) / bb) * 100
            sign = "+" if delta > 0 else ""
            delta_color = "#f85149" if delta > 0 else "#3fb950"
            if key in ("perplexity",) and delta > 0:
                delta_color = "#f0883e"  # PPL increase is acceptable
            ax.text(0.5, 0.95, f"{sign}{delta:.1f}%",
                    transform=ax.transAxes, ha="center", fontsize=12,
                    fontweight="bold", color=delta_color)

        ax.set_title(f"{label}\n({note})", fontsize=11)
        ax.grid(True, axis="y")

    fig.suptitle("AletheionV2 350M — Backbone vs Epistemic Fine-Tuning",
                 fontsize=15, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(out / "comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] comparison_bars.png")

    # --- 3. Tomography radar (if available) ---
    if backbone_results.get("tomography") and finetuned_results.get("tomography"):
        tomo_keys = sorted(set(backbone_results["tomography"]) & set(finetuned_results["tomography"]))
        if len(tomo_keys) >= 3:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            angles = np.linspace(0, 2 * np.pi, len(tomo_keys), endpoint=False).tolist()
            angles += angles[:1]

            bb_vals = [backbone_results["tomography"].get(k, 0) for k in tomo_keys] + \
                      [backbone_results["tomography"].get(tomo_keys[0], 0)]
            ft_vals = [finetuned_results["tomography"].get(k, 0) for k in tomo_keys] + \
                      [finetuned_results["tomography"].get(tomo_keys[0], 0)]

            ax.plot(angles, bb_vals, "o-", color="#58a6ff", linewidth=2, label="Backbone")
            ax.fill(angles, bb_vals, color="#58a6ff", alpha=0.1)
            ax.plot(angles, ft_vals, "o-", color="#3fb950", linewidth=2, label="Fine-tuned")
            ax.fill(angles, ft_vals, color="#3fb950", alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(tomo_keys, fontsize=9)
            ax.set_title("Epistemic Tomography Profile", fontsize=13, fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)

            fig.tight_layout()
            fig.savefig(out / "tomography_radar.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] tomography_radar.png")


def print_results(results, label=""):
    """Imprime resultados formatados."""
    prefix = f"  [{label}]" if label else " "
    print(f"{prefix} PPL:          {results['perplexity']:.2f}")
    print(f"{prefix} CE Loss:      {results['ce_loss']:.4f}")
    print(f"{prefix} ECE:          {results['ece']:.4f}")
    print(f"{prefix} MCE:          {results['mce']:.4f}  (bin {results.get('mce_bin', '?')}: {results.get('mce_bin_range', '?')})")
    print(f"{prefix} Brier Score:  {results['brier_score']:.4f}")
    print(f"{prefix} Overconf:     {results['overconfidence_ratio']:.2%}")
    print(f"{prefix} Mean Conf:    {results['mean_confidence']:.4f}")
    print(f"{prefix} Mean Acc:     {results['mean_accuracy']:.4f}")
    if results.get("tomography"):
        print(f"{prefix} Tomography:")
        for k, v in results["tomography"].items():
            print(f"{prefix}   {k}: {v:.4f}")


def print_comparison_table(bb, ft):
    """Imprime tabela LaTeX-ready para o paper."""
    print("\n" + "=" * 70)
    print("  RESULTADO PRINCIPAL — DELTA ANTES/DEPOIS")
    print("=" * 70)

    header = f"  {'Metric':<25} {'Backbone':>12} {'Fine-tuned':>12} {'Delta':>10}"
    print(header)
    print("  " + "-" * 65)

    rows = [
        ("Perplexity (PPL)", "perplexity", ".2f", False),
        ("CE Loss", "ce_loss", ".4f", False),
        ("ECE", "ece", ".4f", True),
        ("MCE", "mce", ".4f", True),
        ("Brier Score", "brier_score", ".4f", True),
        ("Overconfidence", "overconfidence_ratio", ".2%", True),
        ("Mean Confidence", "mean_confidence", ".4f", False),
        ("Mean Accuracy", "mean_accuracy", ".4f", False),
    ]

    for label, key, fmt, lower_better in rows:
        bv = bb.get(key, 0)
        fv = ft.get(key, 0)

        if fmt == ".2%":
            bs = f"{bv:.2%}"
            fs = f"{fv:.2%}"
        else:
            bs = f"{bv:{fmt}}"
            fs = f"{fv:{fmt}}"

        if bv > 0:
            delta_pct = ((fv - bv) / bv) * 100
            sign = "+" if delta_pct > 0 else ""
            ds = f"{sign}{delta_pct:.1f}%"
            if lower_better:
                if delta_pct < 0:
                    ds += " !"  # improvement marker
        else:
            ds = "N/A"

        print(f"  {label:<25} {bs:>12} {fs:>12} {ds:>10}")

    print("  " + "-" * 65)
    print()

    # LaTeX table
    print("  % LaTeX table for paper:")
    print("  % \\begin{tabular}{lrrr}")
    print("  % \\toprule")
    print("  % \\textbf{Metric} & \\textbf{Backbone} & \\textbf{Fine-tuned} & \\textbf{$\\Delta$} \\\\")
    print("  % \\midrule")
    for label, key, fmt, _ in rows:
        bv = bb.get(key, 0)
        fv = ft.get(key, 0)
        if bv > 0:
            delta_pct = ((fv - bv) / bv) * 100
            sign = "+" if delta_pct > 0 else ""
            ds = f"{sign}{delta_pct:.1f}\\%"
        else:
            ds = "---"
        if fmt == ".2%":
            bs, fs = f"{bv:.2%}", f"{fv:.2%}"
        else:
            bs, fs = f"{bv:{fmt}}", f"{fv:{fmt}}"
        print(f"  % {label} & {bs} & {fs} & {ds} \\\\")
    print("  % \\bottomrule")
    print("  % \\end{tabular}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Avaliacao epistemica AletheionV2")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint unico")
    parser.add_argument("--backbone", default=None, help="Checkpoint do backbone (antes)")
    parser.add_argument("--finetuned", default=None, help="Checkpoint do fine-tuned (depois)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-data", default=None, help="Diretorio dos shards de dados")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seqs", type=int, default=2000)
    parser.add_argument("--n-bins", type=int, default=15, help="Bins para ECE")
    parser.add_argument("--plot", action="store_true", help="Gerar plots")
    parser.add_argument("--output", default=None, help="Arquivo JSON de saida")
    parser.add_argument("--plot-dir", default="plots/epistemic_eval")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        args.device = "cpu"
        print("[WARN] CUDA nao disponivel, usando CPU")

    # Modo comparacao
    if args.backbone and args.finetuned:
        token_sets = prepare_eval_tokens(AletheionV2Config(), args.eval_data)

        # Usa WikiText (OOD) como principal, held-out como secundario
        primary_name = "wikitext" if "wikitext" in token_sets else list(token_sets.keys())[0]
        tokens = token_sets[primary_name]
        print(f"\n[EVAL] Dataset principal: {primary_name}")

        # Backbone
        print("\n" + "=" * 60)
        print("  AVALIANDO BACKBONE")
        print("=" * 60)
        model_bb, config_bb, step_bb = load_model(args.backbone, args.device)
        results_bb = eval_full(
            model_bb, tokens, config_bb.max_seq_len, args.batch_size,
            args.device, with_tomography=False, n_bins=args.n_bins,
            max_seqs=args.max_seqs,
        )
        results_bb["checkpoint"] = args.backbone
        results_bb["step"] = step_bb
        results_bb["phase"] = "backbone"
        results_bb["eval_dataset"] = primary_name
        print_results(results_bb, "Backbone")
        del model_bb
        torch.cuda.empty_cache()

        # Fine-tuned
        print("\n" + "=" * 60)
        print("  AVALIANDO FINE-TUNED")
        print("=" * 60)
        model_ft, config_ft, step_ft = load_model(args.finetuned, args.device)
        results_ft = eval_full(
            model_ft, tokens, config_ft.max_seq_len, args.batch_size,
            args.device, with_tomography=True, n_bins=args.n_bins,
            max_seqs=args.max_seqs,
        )
        results_ft["checkpoint"] = args.finetuned
        results_ft["step"] = step_ft
        results_ft["phase"] = "finetuned"
        results_ft["eval_dataset"] = primary_name
        print_results(results_ft, "Fine-tuned")

        # Avalia tambem no held-out (in-distribution) se disponivel
        results_ft_id = None
        if "held_out" in token_sets and primary_name != "held_out":
            print("\n  [BONUS] Avaliando fine-tuned no held-out (in-distribution)...")
            results_ft_id = eval_full(
                model_ft, token_sets["held_out"], config_ft.max_seq_len,
                args.batch_size, args.device, with_tomography=True,
                n_bins=args.n_bins, max_seqs=args.max_seqs,
            )
            results_ft_id["eval_dataset"] = "held_out"
            print(f"  [held-out] PPL={results_ft_id['perplexity']:.2f} ECE={results_ft_id['ece']:.4f}")

        del model_ft
        torch.cuda.empty_cache()

        # Comparacao
        print_comparison_table(results_bb, results_ft)

        # MCE detail
        print(f"  MCE detail (backbone): bin {results_bb['mce_bin']} ({results_bb['mce_bin_range']}), gap={results_bb['mce']:.4f}")
        print(f"  MCE detail (fine-tuned): bin {results_ft['mce_bin']} ({results_ft['mce_bin_range']}), gap={results_ft['mce']:.4f}")

        # Plots
        if args.plot:
            print("\n[PLOT] Gerando plots comparativos...")
            plot_comparison(results_bb, results_ft, args.plot_dir)

        # Salva JSON
        output_path = args.output or "eval_epistemic_comparison.json"
        combined = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eval_dataset": primary_name,
            "note": "ECE/Brier measured on out-of-distribution data (WikiText-103 test). Model trained on FineWeb-Edu.",
            "backbone": results_bb,
            "finetuned": results_ft,
        }
        if results_ft_id:
            combined["finetuned_in_distribution"] = results_ft_id
        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n[SAVE] {output_path}")

    # Modo checkpoint unico
    elif args.checkpoint:
        token_sets = prepare_eval_tokens(AletheionV2Config(), args.eval_data)
        primary_name = "wikitext" if "wikitext" in token_sets else list(token_sets.keys())[0]
        tokens = token_sets[primary_name]

        print("\n" + "=" * 60)
        print(f"  AVALIACAO EPISTEMICA - AletheionV2 350M ({primary_name})")
        print("=" * 60)

        model, config, step = load_model(args.checkpoint, args.device)

        # Tenta com tomografia
        try:
            results = eval_full(
                model, tokens, config.max_seq_len, args.batch_size,
                args.device, with_tomography=True, n_bins=args.n_bins,
                max_seqs=args.max_seqs,
            )
        except Exception as e:
            print(f"[WARN] Tomografia falhou ({e}), avaliando sem...")
            results = eval_full(
                model, tokens, config.max_seq_len, args.batch_size,
                args.device, with_tomography=False, n_bins=args.n_bins,
                max_seqs=args.max_seqs,
            )

        results["checkpoint"] = args.checkpoint
        results["step"] = step
        results["eval_dataset"] = primary_name

        print_results(results)
        print(f"\n  MCE detail: bin {results['mce_bin']} ({results['mce_bin_range']}), gap={results['mce']:.4f}")

        # Referencia
        print("\n  Referencia (350M params):")
        print("  GPT-2 Medium: PPL ~22.8, ECE ~0.10-0.15")
        print("  OPT-350M:     PPL ~22.0, ECE ~0.08-0.12")
        print("  (ECE tipico de LLMs sem calibracao: 0.10-0.20)")
        print("  (ECE apos temperature scaling: 0.03-0.06)")

        # Salva
        output_path = args.output or os.path.join(
            os.path.dirname(args.checkpoint), "epistemic_eval.json"
        )
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVE] {output_path}")

    else:
        print("[ERRO] Use --checkpoint ou --backbone + --finetuned")
        sys.exit(1)


if __name__ == "__main__":
    main()
