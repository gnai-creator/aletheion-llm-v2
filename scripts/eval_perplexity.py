"""
Avaliacao de perplexidade do AletheionV2.

Calcula perplexidade em datasets padrao (WikiText-103, LAMBADA, PTB)
e no proprio eval split do dataset de treino.

Uso:
    python scripts/eval_perplexity.py --checkpoint checkpoints/350m_4xh100/best_model.pt
    python scripts/eval_perplexity.py --checkpoint checkpoints/350m_4xh100/step_100000.pt --datasets wikitext lambada
    python scripts/eval_perplexity.py --checkpoint checkpoints/350m_4xh100/step_100000.pt --eval-data data/350m_rtx4090

Saida:
    Tabela com perplexidade por dataset + JSON com resultados detalhados.
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Carrega modelo de um checkpoint."""
    print(f"[LOAD] Carregando {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extrai config
    if "config" in ckpt:
        config = ckpt["config"]
        if isinstance(config, dict):
            config = AletheionV2Config(**config)
    else:
        config = AletheionV2Config()

    model = AletheionV2Model(config)

    # Carrega pesos (trata DDP/FSDP prefix)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)

    model = model.to(device).eval()
    print(f"[LOAD] Modelo: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"[LOAD] Step: {ckpt.get('global_step', 'N/A')}")
    return model, config


@torch.no_grad()
def eval_perplexity_on_tokens(model, tokens, seq_len=1024, batch_size=8, device="cuda"):
    """Calcula perplexidade sobre uma sequencia de tokens."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Cria batches de sequencias
    n_seqs = len(tokens) // (seq_len + 1)
    if n_seqs == 0:
        return float("inf")

    n_seqs = min(n_seqs, 5000)  # max 5000 seqs para nao demorar

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
        labels = torch.tensor(np.array(labels), dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids, return_tomography=False)
            loss = torch.nn.functional.cross_entropy(
                output.logits.view(-1, output.logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def eval_on_local_data(model, data_dir, config, device="cuda"):
    """Avalia no eval split dos dados locais (shards)."""
    from glob import glob

    shards = sorted(glob(os.path.join(data_dir, "train_shard_*.bin")))
    if not shards:
        return None

    # Usa ultimo shard como eval (nao visto no treino se save_total_limit < total)
    eval_shard = shards[-1]
    print(f"[EVAL] Usando shard de eval: {os.path.basename(eval_shard)}")

    tokens = np.memmap(eval_shard, dtype=np.uint16, mode="r")
    tokens = tokens.astype(np.int64)

    ppl = eval_perplexity_on_tokens(
        model, tokens,
        seq_len=config.max_seq_len,
        batch_size=8,
        device=device,
    )
    return ppl


def eval_on_wikitext(model, config, device="cuda"):
    """Avalia no WikiText-103 test set."""
    try:
        import tiktoken
        from datasets import load_dataset

        print("[EVAL] Baixando WikiText-103...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join([x["text"] for x in ds if x["text"].strip()])

        enc = tiktoken.get_encoding("gpt2")
        tokens = np.array(enc.encode(text), dtype=np.int64)
        print(f"[EVAL] WikiText-103: {len(tokens):,} tokens")

        ppl = eval_perplexity_on_tokens(
            model, tokens,
            seq_len=config.max_seq_len,
            batch_size=8,
            device=device,
        )
        return ppl
    except Exception as e:
        print(f"[WARN] WikiText-103 falhou: {e}")
        return None


def eval_on_lambada(model, config, device="cuda"):
    """Avalia no LAMBADA test set."""
    try:
        import tiktoken
        from datasets import load_dataset

        print("[EVAL] Baixando LAMBADA...")
        ds = load_dataset("lambada", split="test")
        text = "\n\n".join([x["text"] for x in ds])

        enc = tiktoken.get_encoding("gpt2")
        tokens = np.array(enc.encode(text), dtype=np.int64)
        print(f"[EVAL] LAMBADA: {len(tokens):,} tokens")

        ppl = eval_perplexity_on_tokens(
            model, tokens,
            seq_len=config.max_seq_len,
            batch_size=8,
            device=device,
        )
        return ppl
    except Exception as e:
        print(f"[WARN] LAMBADA falhou: {e}")
        return None


def eval_on_ptb(model, config, device="cuda"):
    """Avalia no Penn Treebank test set."""
    try:
        import tiktoken
        from datasets import load_dataset

        print("[EVAL] Baixando Penn Treebank...")
        ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
        text = "\n".join([x["sentence"] for x in ds])

        enc = tiktoken.get_encoding("gpt2")
        tokens = np.array(enc.encode(text), dtype=np.int64)
        print(f"[EVAL] PTB: {len(tokens):,} tokens")

        ppl = eval_perplexity_on_tokens(
            model, tokens,
            seq_len=config.max_seq_len,
            batch_size=8,
            device=device,
        )
        return ppl
    except Exception as e:
        print(f"[WARN] PTB falhou: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Avaliacao de perplexidade AletheionV2")
    parser.add_argument("--checkpoint", required=True, help="Caminho do checkpoint .pt")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--eval-data", default=None, help="Diretorio dos shards de dados")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["local", "wikitext", "lambada"],
        choices=["local", "wikitext", "lambada", "ptb"],
        help="Datasets para avaliar",
    )
    parser.add_argument("--output", default=None, help="Arquivo JSON de saida")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        args.device = "cpu"
        print("[WARN] CUDA nao disponivel, usando CPU")

    model, config = load_model(args.checkpoint, args.device)

    results = {
        "checkpoint": args.checkpoint,
        "step": None,
        "params": sum(p.numel() for p in model.parameters()),
        "perplexity": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Extrai step do checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    results["step"] = ckpt.get("global_step", "unknown")

    print("\n" + "=" * 60)
    print("  AVALIACAO DE PERPLEXIDADE - AletheionV2 350M")
    print("=" * 60)

    # Avalia em cada dataset
    if "local" in args.datasets and args.eval_data:
        t0 = time.time()
        ppl = eval_on_local_data(model, args.eval_data, config, args.device)
        dt = time.time() - t0
        if ppl is not None:
            results["perplexity"]["eval_split"] = round(ppl, 2)
            print(f"\n  Eval split:    PPL = {ppl:.2f}  ({dt:.1f}s)")

    if "wikitext" in args.datasets:
        t0 = time.time()
        ppl = eval_on_wikitext(model, config, args.device)
        dt = time.time() - t0
        if ppl is not None:
            results["perplexity"]["wikitext_103"] = round(ppl, 2)
            print(f"  WikiText-103:  PPL = {ppl:.2f}  ({dt:.1f}s)")

    if "lambada" in args.datasets:
        t0 = time.time()
        ppl = eval_on_lambada(model, config, args.device)
        dt = time.time() - t0
        if ppl is not None:
            results["perplexity"]["lambada"] = round(ppl, 2)
            print(f"  LAMBADA:       PPL = {ppl:.2f}  ({dt:.1f}s)")

    if "ptb" in args.datasets:
        t0 = time.time()
        ppl = eval_on_ptb(model, config, args.device)
        dt = time.time() - t0
        if ppl is not None:
            results["perplexity"]["ptb"] = round(ppl, 2)
            print(f"  PTB:           PPL = {ppl:.2f}  ({dt:.1f}s)")

    print("\n" + "=" * 60)

    # Contexto: perplexidades de referencia para 350M params
    print("\n  Referencia (350M params, ~7B tokens):")
    print("  GPT-2 Small (124M):  WikiText-103 PPL ~29.4")
    print("  GPT-2 Medium (355M): WikiText-103 PPL ~22.8")
    print("  OPT-350M:            WikiText-103 PPL ~22.0")
    print("=" * 60)

    # Salva JSON
    output_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "perplexity_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Resultados salvos em {output_path}")


if __name__ == "__main__":
    main()
