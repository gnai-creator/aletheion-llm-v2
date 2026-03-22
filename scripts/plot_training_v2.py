"""
Visualizacao de treinamento AletheionV2 - formato pipe-separated.

Parseia logs no formato:
    2026-03-17 04:40:28,699 | ce=9.945789 | step=50 | stp=1.494911 | ...

Uso:
    python scripts/plot_training_v2.py checkpoints/50m_backbone/train.log -o eval_results/50m/main_backbone
    python scripts/plot_training_v2.py checkpoints/50m_backbone/train.log -o eval_results/50m/main_backbone --title "50M Backbone"
"""

import re
import sys
import math
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("[ERRO] matplotlib nao instalado. Instale com: pip install matplotlib")
    sys.exit(1)


STYLE = {
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
}


def apply_style():
    """Aplica tema dark nos graficos."""
    plt.rcParams.update(STYLE)


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Suaviza valores com media movel."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def parse_pipe_log(log_path: str) -> dict:
    """
    Parseia log pipe-separated key=value.

    Se houver multiplas runs (restart), mantém apenas a ultima
    run completa (a com mais steps).
    """
    runs = []
    current_run = []

    with open(log_path) as f:
        for line in f:
            if "=== Training started" in line:
                if current_run:
                    runs.append(current_run)
                current_run = []
                continue

            if " | " not in line:
                continue

            parts = line.split(" | ")
            entry = {}
            for part in parts:
                part = part.strip()
                if "=" in part and not part.startswith("2026"):
                    key, val = part.split("=", 1)
                    try:
                        entry[key.strip()] = float(val.strip())
                    except ValueError:
                        pass

            if entry and "step" in entry:
                current_run.append(entry)

    if current_run:
        runs.append(current_run)

    if not runs:
        print("[ERRO] Nenhum step encontrado no log.")
        sys.exit(1)

    # Pega a run mais longa
    best_run = max(runs, key=len)
    print(f"[PARSE] {len(runs)} run(s) encontrada(s), usando a maior ({len(best_run)} entries)")

    # Converte para dict de arrays
    keys = sorted(best_run[0].keys())
    data = {k: np.array([e.get(k, 0.0) for e in best_run]) for k in keys}
    return data


def fmt_step(x, p):
    """Formata step como 1K, 2K, etc."""
    if x >= 1000:
        return f"{x/1000:.0f}K"
    return f"{x:.0f}"


def plot_ce_loss(data: dict, out: Path, title_prefix: str):
    """CE Loss curve."""
    fig, ax = plt.subplots(figsize=(14, 6))
    steps = data["step"]
    ce = data["ce"]

    ax.plot(steps, ce, alpha=0.25, color="#58a6ff", linewidth=0.5)
    if len(steps) > 20:
        s = smooth(ce, 20)
        ax.plot(steps[19:], s, color="#58a6ff", linewidth=2.5, label="CE Loss (smooth)")

    current_ce = float(ce[-1])
    ax.axhline(y=current_ce, color="#ff7b72", linestyle=":", alpha=0.8, linewidth=1.5)
    ax.text(
        steps[-1] * 0.98, current_ce + 0.03,
        f"Final = {current_ce:.3f} (PPL {math.exp(current_ce):.1f})",
        color="#ff7b72", fontsize=10, fontweight="bold", alpha=0.9, ha="right",
    )

    tokens_b = data["tokens_seen"][-1] / 1e9 if "tokens_seen" in data else 0
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(
        f"{title_prefix} -- CE Loss  (step {int(steps[-1]):,}, {tokens_b:.2f}B tokens)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    fig.tight_layout()
    fig.savefig(out / "ce_loss.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] ce_loss.png")


def plot_ce_vs_tokens(data: dict, out: Path, title_prefix: str):
    """CE Loss vs tokens processados."""
    if "tokens_seen" not in data:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    tokens_b = data["tokens_seen"] / 1e9
    ce = data["ce"]

    ax.plot(tokens_b, ce, alpha=0.25, color="#58a6ff", linewidth=0.5)
    if len(tokens_b) > 20:
        s = smooth(ce, 20)
        ax.plot(tokens_b[19:], s, color="#58a6ff", linewidth=2.5, label="CE Loss")

    current_ce = float(ce[-1])
    ax.axhline(y=current_ce, color="#ff7b72", linestyle=":", alpha=0.8, linewidth=1.5)
    ax.text(
        tokens_b[-1] * 0.98, current_ce + 0.03,
        f"Final = {current_ce:.3f}",
        color="#ff7b72", fontsize=10, fontweight="bold", alpha=0.9, ha="right",
    )

    ax.set_xlabel("Tokens (Billions)")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"{title_prefix} -- CE vs Tokens", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "ce_vs_tokens.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] ce_vs_tokens.png")


def plot_all_losses(data: dict, out: Path, title_prefix: str):
    """CE + STP losses."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    steps = data["step"]
    w = 20

    # CE
    ax1.plot(steps, data["ce"], alpha=0.25, color="#58a6ff", linewidth=0.5)
    if len(steps) > w:
        ax1.plot(steps[w-1:], smooth(data["ce"], w), color="#58a6ff", linewidth=2.5, label="CE")
    ax1.set_ylabel("Cross-Entropy")
    ax1.set_title(f"{title_prefix} -- Training Losses", fontsize=14, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # STP
    if "stp" in data:
        ax2.plot(steps, data["stp"], alpha=0.25, color="#f0883e", linewidth=0.5)
        if len(steps) > w:
            ax2.plot(steps[w-1:], smooth(data["stp"], w), color="#f0883e", linewidth=2.5, label="STP")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("STP Loss")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    fig.tight_layout()
    fig.savefig(out / "all_losses.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] all_losses.png")


def plot_lr_schedule(data: dict, out: Path, title_prefix: str):
    """Learning rate schedule."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(data["step"], data["lr"], color="#d2a8ff", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{title_prefix} -- LR Schedule (Warmup Cosine)", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.1e}"))
    fig.tight_layout()
    fig.savefig(out / "lr_schedule.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] lr_schedule.png")


def plot_grad_norm(data: dict, out: Path, title_prefix: str):
    """Gradient norm."""
    fig, ax = plt.subplots(figsize=(14, 4))
    steps = data["step"]
    ax.plot(steps, data["grad_norm"], alpha=0.25, color="#f778ba", linewidth=0.5)
    if len(steps) > 20:
        ax.plot(steps[19:], smooth(data["grad_norm"], 20), color="#f778ba", linewidth=2.5, label="Grad Norm (smooth)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"{title_prefix} -- Gradient Norm", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    fig.tight_layout()
    fig.savefig(out / "grad_norm.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] grad_norm.png")


def plot_throughput(data: dict, out: Path, title_prefix: str):
    """Throughput."""
    if "tokens_per_sec" not in data:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    tok_s = data["tokens_per_sec"] / 1000
    ax.plot(data["step"], tok_s, color="#3fb950", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("K tokens/s")
    ax.set_title(f"{title_prefix} -- Throughput", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ymin = max(0, tok_s.min() - 5)
    ymax = tok_s.max() + 5
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] throughput.png")


def plot_perplexity(data: dict, out: Path, title_prefix: str):
    """Perplexidade derivada da CE loss."""
    fig, ax = plt.subplots(figsize=(14, 6))
    steps = data["step"]
    ppl = np.exp(np.clip(data["ce"], 0, 20))

    ax.plot(steps, ppl, alpha=0.25, color="#bc8cff", linewidth=0.5)
    if len(steps) > 20:
        ax.plot(steps[19:], smooth(ppl, 20), color="#bc8cff", linewidth=2.5, label="Train PPL")

    current_ppl = float(ppl[-1])
    ax.axhline(y=current_ppl, color="#ff7b72", linestyle=":", alpha=0.8, linewidth=1.5)
    ax.text(
        steps[-1] * 0.98, current_ppl * 1.02,
        f"Final = {current_ppl:.1f}",
        color="#ff7b72", fontsize=10, fontweight="bold", alpha=0.9, ha="right",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"{title_prefix} -- Train Perplexity", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    if ppl.max() / max(ppl.min(), 1) > 50:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "perplexity.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] perplexity.png")


def plot_dashboard(data: dict, out: Path, title_prefix: str):
    """Dashboard com todas as metricas."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    steps = data["step"]
    w = 20

    # CE Loss
    ax = axes[0, 0]
    ax.plot(steps, data["ce"], alpha=0.2, color="#58a6ff", linewidth=0.5)
    if len(steps) > w:
        ax.plot(steps[w-1:], smooth(data["ce"], w), color="#58a6ff", linewidth=2)
    ax.set_title("CE Loss", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # STP Loss
    ax = axes[0, 1]
    if "stp" in data:
        ax.plot(steps, data["stp"], alpha=0.2, color="#f0883e", linewidth=0.5)
        if len(steps) > w:
            ax.plot(steps[w-1:], smooth(data["stp"], w), color="#f0883e", linewidth=2)
    ax.set_title("STP Loss", fontsize=12, fontweight="bold")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # LR
    ax = axes[0, 2]
    ax.plot(steps, data["lr"], color="#d2a8ff", linewidth=2)
    ax.set_title("Learning Rate", fontsize=12, fontweight="bold")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.0e}"))

    # Perplexity
    ax = axes[1, 0]
    ppl = np.exp(np.clip(data["ce"], 0, 20))
    ax.plot(steps, ppl, alpha=0.2, color="#bc8cff", linewidth=0.5)
    if len(steps) > w:
        ax.plot(steps[w-1:], smooth(ppl, w), color="#bc8cff", linewidth=2)
    ax.set_title("Perplexity", fontsize=12, fontweight="bold")
    ax.set_ylabel("PPL")
    ax.set_xlabel("Step")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    if ppl.max() / max(ppl.min(), 1) > 50:
        ax.set_yscale("log")

    # Grad Norm
    ax = axes[1, 1]
    ax.plot(steps, data["grad_norm"], alpha=0.2, color="#f778ba", linewidth=0.5)
    if len(steps) > w:
        ax.plot(steps[w-1:], smooth(data["grad_norm"], w), color="#f778ba", linewidth=2)
    ax.set_title("Gradient Norm", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Throughput
    ax = axes[1, 2]
    if "tokens_per_sec" in data:
        tok_s = data["tokens_per_sec"] / 1000
        ax.plot(steps, tok_s, color="#3fb950", linewidth=2)
        ymin = max(0, tok_s.min() - 5)
        ymax = tok_s.max() + 5
        ax.set_ylim(ymin, ymax)
    ax.set_title("Throughput (K tok/s)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    tokens_b = data["tokens_seen"][-1] / 1e9 if "tokens_seen" in data else 0
    fig.suptitle(
        f"{title_prefix} -- Training Dashboard  "
        f"(step {int(steps[-1]):,}, {tokens_b:.2f}B tokens)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] dashboard.png")


def print_summary(data: dict, title_prefix: str):
    """Imprime resumo do treino."""
    steps = data["step"]
    tokens_b = data["tokens_seen"][-1] / 1e9 if "tokens_seen" in data else 0

    print(f"\n{'='*55}")
    print(f"  {title_prefix} -- Training Summary")
    print(f"{'='*55}")
    print(f"  Steps:       {int(steps[-1]):,}")
    print(f"  Tokens:      {tokens_b:.2f}B")
    print(f"  CE (start):  {data['ce'][0]:.4f}")
    print(f"  CE (final):  {data['ce'][-1]:.4f}")
    print(f"  CE (min):    {data['ce'].min():.4f}")
    print(f"  PPL (final): {math.exp(data['ce'][-1]):.1f}")
    print(f"  PPL (min):   {math.exp(data['ce'].min()):.1f}")
    if "stp" in data:
        print(f"  STP (avg):   {data['stp'].mean():.4f}")
    print(f"  LR (final):  {data['lr'][-1]:.2e}")
    if "tokens_per_sec" in data:
        print(f"  Throughput:  {data['tokens_per_sec'].mean()/1000:.1f}K tok/s")
    print(f"  Grad norm:   {data['grad_norm'][-1]:.3f} (avg {data['grad_norm'].mean():.3f})")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="Plot AletheionV2 training (pipe-separated logs)")
    parser.add_argument("log", help="Caminho do train.log (pipe-separated)")
    parser.add_argument("--output", "-o", default=None, help="Diretorio de saida")
    parser.add_argument("--title", "-t", default=None, help="Prefixo do titulo (ex: 'AletheionV2 50M')")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERRO] Arquivo nao encontrado: {log_path}")
        sys.exit(1)

    out = Path(args.output) if args.output else log_path.parent / "plots"
    out.mkdir(parents=True, exist_ok=True)

    title_prefix = args.title or f"AletheionV2 ({log_path.parent.name})"

    apply_style()

    data = parse_pipe_log(str(log_path))

    required = ["step", "ce", "lr", "grad_norm"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"[ERRO] Campos obrigatorios ausentes: {missing}")
        sys.exit(1)

    print(f"[PLOT] Gerando graficos em {out}/\n")

    plot_ce_loss(data, out, title_prefix)
    plot_ce_vs_tokens(data, out, title_prefix)
    plot_all_losses(data, out, title_prefix)
    plot_lr_schedule(data, out, title_prefix)
    plot_grad_norm(data, out, title_prefix)
    plot_throughput(data, out, title_prefix)
    plot_perplexity(data, out, title_prefix)
    plot_dashboard(data, out, title_prefix)

    print_summary(data, title_prefix)
    print(f"\n  8 plots salvos em {out}/")


if __name__ == "__main__":
    main()
