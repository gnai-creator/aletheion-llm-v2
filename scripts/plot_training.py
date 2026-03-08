"""
Visualizacao de treinamento AletheionV2.

Gera graficos a partir de logs de treinamento (texto ou JSON).

Uso:
    # A partir do log de texto do cloud training
    python scripts/plot_training.py checkpoints/350m_4xh100/cloud_train.log

    # A partir de log JSON
    python scripts/plot_training.py --json checkpoints/training_log.json

    # Salvar em diretorio especifico
    python scripts/plot_training.py checkpoints/350m_4xh100/cloud_train.log -o plots/my_run
"""

import re
import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[ERRO] matplotlib nao instalado. Instale com: pip install matplotlib")
    sys.exit(1)


# --- Dark theme ---
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
    plt.rcParams.update(STYLE)


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def parse_text_log(log_path: str):
    """Extrai metricas do log de texto (cloud_train.log)."""
    pattern = re.compile(
        r"step=(\d+)/(\d+)\s+"
        r"loss=([\d.]+)\s+"
        r"ce=([\d.]+)\s+"
        r"stp=([\d.]+)\s+"
        r"lr=([\d.e+-]+)\s+"
        r"gnorm=([\d.]+)\s+"
        r"tok/s=(\d+)\s+"
        r"tokens=([\d,]+)"
    )

    data = {
        "step": [], "total_steps": [],
        "loss": [], "ce": [], "stp": [],
        "lr": [], "gnorm": [], "tok_s": [], "tokens": [],
    }

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data["step"].append(int(m.group(1)))
                data["total_steps"].append(int(m.group(2)))
                data["loss"].append(float(m.group(3)))
                data["ce"].append(float(m.group(4)))
                data["stp"].append(float(m.group(5)))
                data["lr"].append(float(m.group(6)))
                data["gnorm"].append(float(m.group(7)))
                data["tok_s"].append(int(m.group(8)))
                data["tokens"].append(int(m.group(9).replace(",", "")))

    for k in data:
        data[k] = np.array(data[k])

    return data


def parse_json_log(log_path: str):
    """Carrega log JSON e converte para formato interno."""
    with open(log_path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        data = {}
        for entry in raw:
            for key, val in entry.items():
                if key not in data:
                    data[key] = []
                data[key].append(val)
        for k in data:
            data[k] = np.array(data[k])
        return data
    elif isinstance(raw, dict):
        for k in raw:
            if isinstance(raw[k], list):
                raw[k] = np.array(raw[k])
        return raw
    return {}


def fmt_step(x, p):
    return f"{x/1000:.0f}K"


def plot_ce_loss(data, out):
    """CE Loss curve with reference lines."""
    fig, ax = plt.subplots(figsize=(14, 6))
    steps = data["step"]
    total = data["total_steps"][0] if len(data["total_steps"]) > 0 else 106811
    ce = data["ce"]

    ax.plot(steps, ce, alpha=0.25, color="#58a6ff", linewidth=0.5)
    if len(steps) > 20:
        s = smooth(ce, 20)
        ax.plot(steps[19:], s, color="#58a6ff", linewidth=2.5, label="CE Loss (smooth)")

    # Reference lines
    refs = [
        (3.5, "GPT-2 Medium (355M)", "#3fb950"),
        (3.3, "OPT-350M", "#f0883e"),
    ]
    for val, label, color in refs:
        if ce.min() < val + 1.0:
            ax.axhline(y=val, color=color, linestyle="--", alpha=0.5, linewidth=1)
            ax.text(steps[1], val + 0.03, label, color=color, fontsize=9, alpha=0.7)

    # AletheionV2 current position
    current_ce = float(ce[-1])
    ax.axhline(y=current_ce, color="#ff7b72", linestyle=":", alpha=0.8, linewidth=1.5)
    ax.text(steps[-1] * 0.98, current_ce + 0.03, f"AletheionV2 350M = {current_ce:.2f}", color="#ff7b72", fontsize=10, fontweight="bold", alpha=0.9, ha="right")

    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(
        f"AletheionV2 350M — CE Loss  (step {steps[-1]:,}/{total:,}, {steps[-1]/total*100:.1f}%)",
        fontsize=14, fontweight="bold"
    )
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    fig.tight_layout()
    fig.savefig(out / "ce_loss.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] ce_loss.png")


def plot_ce_vs_tokens(data, out):
    """CE Loss vs tokens processed (scaling law view)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    tokens_b = data["tokens"] / 1e9
    ce = data["ce"]

    ax.plot(tokens_b, ce, alpha=0.25, color="#58a6ff", linewidth=0.5)
    if len(tokens_b) > 20:
        s = smooth(ce, 20)
        ax.plot(tokens_b[19:], s, color="#58a6ff", linewidth=2.5, label="CE Loss")

    refs = [
        (3.5, "GPT-2 Medium (355M)", "#3fb950"),
        (3.3, "OPT-350M", "#f0883e"),
    ]
    for val, label, color in refs:
        if ce.min() < val + 1.0:
            ax.axhline(y=val, color=color, linestyle="--", alpha=0.5, linewidth=1)
            ax.text(0.05, val + 0.03, label, color=color, fontsize=9, alpha=0.7)

    # AletheionV2 current position
    current_ce = float(ce[-1])
    ax.axhline(y=current_ce, color="#ff7b72", linestyle=":", alpha=0.8, linewidth=1.5)
    ax.text(tokens_b[-1] * 0.98, current_ce + 0.03, f"AletheionV2 350M = {current_ce:.2f}", color="#ff7b72", fontsize=10, fontweight="bold", alpha=0.9, ha="right")

    ax.set_xlabel("Tokens (Billions)")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("AletheionV2 350M — CE vs Tokens Processed", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "ce_vs_tokens.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] ce_vs_tokens.png")


def plot_all_losses(data, out):
    """CE + STP losses on separate axes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    steps = data["step"]

    # CE
    ax1.plot(steps, data["ce"], alpha=0.25, color="#58a6ff", linewidth=0.5)
    if len(steps) > 20:
        ax1.plot(steps[19:], smooth(data["ce"], 20), color="#58a6ff", linewidth=2.5, label="CE")
    ax1.set_ylabel("Cross-Entropy")
    ax1.set_title("AletheionV2 350M — Training Losses", fontsize=14, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # STP
    ax2.plot(steps, data["stp"], alpha=0.25, color="#f0883e", linewidth=0.5)
    if len(steps) > 20:
        ax2.plot(steps[19:], smooth(data["stp"], 20), color="#f0883e", linewidth=2.5, label="STP")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("STP Loss")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    fig.tight_layout()
    fig.savefig(out / "all_losses.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] all_losses.png")


def plot_lr_schedule(data, out):
    """Learning rate schedule."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(data["step"], data["lr"], color="#d2a8ff", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Warmup Cosine)", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.1e}"))
    fig.tight_layout()
    fig.savefig(out / "lr_schedule.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] lr_schedule.png")


def plot_grad_norm(data, out):
    """Gradient norm over time."""
    fig, ax = plt.subplots(figsize=(14, 4))
    steps = data["step"]
    ax.plot(steps, data["gnorm"], alpha=0.25, color="#f778ba", linewidth=0.5)
    if len(steps) > 20:
        ax.plot(steps[19:], smooth(data["gnorm"], 20), color="#f778ba", linewidth=2.5, label="Grad Norm (smooth)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    fig.tight_layout()
    fig.savefig(out / "grad_norm.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] grad_norm.png")


def plot_throughput(data, out):
    """Throughput over time."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(data["step"], data["tok_s"] / 1000, color="#3fb950", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("K tokens/s")
    ax.set_title("Throughput (4x H100 DDP)", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ymin = max(0, data["tok_s"].min() / 1000 - 10)
    ymax = data["tok_s"].max() / 1000 + 10
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] throughput.png")


def plot_perplexity(data, out):
    """Perplexity derived from CE loss."""
    fig, ax = plt.subplots(figsize=(14, 6))
    steps = data["step"]
    ppl = np.exp(np.clip(data["ce"], 0, 20))

    ax.plot(steps, ppl, alpha=0.25, color="#bc8cff", linewidth=0.5)
    if len(steps) > 20:
        ax.plot(steps[19:], smooth(ppl, 20), color="#bc8cff", linewidth=2.5, label="Train PPL")

    # References
    refs = [
        (math.exp(3.5), "GPT-2 Medium ~33", "#3fb950"),
        (math.exp(3.3), "OPT-350M ~27", "#f0883e"),
    ]
    for val, label, color in refs:
        if ppl.min() < val * 2:
            ax.axhline(y=val, color=color, linestyle="--", alpha=0.5, linewidth=1)
            ax.text(steps[1], val * 1.02, label, color=color, fontsize=9, alpha=0.7)

    # AletheionV2 current position
    current_ppl = float(ppl[-1])
    ax.axhline(y=current_ppl, color="#ff7b72", linestyle=":", alpha=0.8, linewidth=1.5)
    ax.text(steps[-1] * 0.98, current_ppl * 1.02, f"AletheionV2 350M = {current_ppl:.1f}", color="#ff7b72", fontsize=10, fontweight="bold", alpha=0.9, ha="right")

    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("AletheionV2 350M — Train Perplexity", fontsize=14, fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    if ppl.max() / ppl.min() > 50:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "perplexity.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] perplexity.png")


def plot_dashboard(data, out):
    """Single dashboard with all metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    steps = data["step"]
    total = data["total_steps"][0] if len(data["total_steps"]) > 0 else 106811
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
    if ppl.max() / ppl.min() > 50:
        ax.set_yscale("log")

    # Grad Norm
    ax = axes[1, 1]
    ax.plot(steps, data["gnorm"], alpha=0.2, color="#f778ba", linewidth=0.5)
    if len(steps) > w:
        ax.plot(steps[w-1:], smooth(data["gnorm"], w), color="#f778ba", linewidth=2)
    ax.set_title("Gradient Norm", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Throughput
    ax = axes[1, 2]
    ax.plot(steps, data["tok_s"] / 1000, color="#3fb950", linewidth=2)
    ax.set_title("Throughput (K tok/s)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    ymin = max(0, data["tok_s"].min() / 1000 - 10)
    ymax = data["tok_s"].max() / 1000 + 10
    ax.set_ylim(ymin, ymax)

    fig.suptitle(
        f"AletheionV2 350M — Training Dashboard  "
        f"(step {steps[-1]:,}/{total:,} = {steps[-1]/total*100:.1f}%)",
        fontsize=16, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] dashboard.png")


def print_summary(data):
    steps = data["step"]
    total = data["total_steps"][0] if len(data["total_steps"]) > 0 else 106811
    tokens_b = data["tokens"] / 1e9

    print(f"\n{'='*55}")
    print(f"  AletheionV2 350M Training Summary")
    print(f"{'='*55}")
    print(f"  Steps:       {steps[-1]:,} / {total:,} ({steps[-1]/total*100:.1f}%)")
    print(f"  Tokens:      {tokens_b[-1]:.2f}B / 7.00B")
    print(f"  CE (start):  {data['ce'][0]:.4f}")
    print(f"  CE (now):    {data['ce'][-1]:.4f}")
    print(f"  CE (min):    {data['ce'].min():.4f}")
    print(f"  PPL (now):   {math.exp(data['ce'][-1]):.1f}")
    print(f"  PPL (min):   {math.exp(data['ce'].min()):.1f}")
    print(f"  STP (avg):   {data['stp'].mean():.4f}")
    print(f"  LR (now):    {data['lr'][-1]:.2e}")
    print(f"  Throughput:  {data['tok_s'].mean()/1000:.1f}K tok/s")
    print(f"  Grad norm:   {data['gnorm'][-1]:.2f} (avg {data['gnorm'].mean():.2f})")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="Plot AletheionV2 training curves")
    parser.add_argument("log", nargs="?", default=None, help="Path to cloud_train.log (text)")
    parser.add_argument("--json", default=None, help="Path to JSON training log")
    parser.add_argument("--output", "-o", default="plots/350m_4xh100", help="Output directory")
    args = parser.parse_args()

    if not args.log and not args.json:
        print("[ERRO] Passe o caminho do log: python scripts/plot_training.py <log>")
        sys.exit(1)

    apply_style()

    if args.json:
        data = parse_json_log(args.json)
    else:
        data = parse_text_log(args.log)

    if len(data.get("step", [])) == 0:
        print("[ERRO] Nenhum step encontrado no log.")
        sys.exit(1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[PARSE] {len(data['step'])} steps parsed")
    print(f"[PLOT] Gerando graficos em {out}/\n")

    plot_ce_loss(data, out)
    plot_ce_vs_tokens(data, out)
    plot_all_losses(data, out)
    plot_lr_schedule(data, out)
    plot_grad_norm(data, out)
    plot_throughput(data, out)
    plot_perplexity(data, out)
    plot_dashboard(data, out)

    print_summary(data)
    print(f"\n  8 plots salvos em {out}/")


if __name__ == "__main__":
    main()
