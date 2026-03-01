"""
Visualizacao de treinamento AletheionV2.

Gera graficos a partir de logs de treinamento (JSON ou checkpoints).

Uso:
    # A partir do log JSON gerado pelo trainer
    python scripts/plot_training.py --log checkpoints/training_log.json

    # A partir de multiplos logs (comparacao)
    python scripts/plot_training.py --log run1/log.json run2/log.json --labels "1M" "10M"

    # Gerar graficos sinteticos de demonstracao
    python scripts/plot_training.py --demo

    # Salvar em diretorio especifico
    python scripts/plot_training.py --log checkpoints/training_log.json --output plots/
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[ERRO] matplotlib nao instalado. Instale com: pip install matplotlib")
    sys.exit(1)


# Paleta de cores consistente
COLORS = {
    "loss": "#2196F3",
    "eval_loss": "#FF9800",
    "lr": "#4CAF50",
    "q1": "#9C27B0",
    "q2": "#E91E63",
    "confidence": "#00BCD4",
    "phi": "#FF5722",
    "vi_severity": "#795548",
    "temperature": "#607D8B",
    "perplexity": "#3F51B5",
    "tokens_per_sec": "#8BC34A",
    "grad_norm": "#FFC107",
}

STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.3,
}


def apply_style():
    """Aplica estilo escuro aos graficos."""
    plt.rcParams.update(STYLE)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def smooth(values, window=10):
    """Suaviza curva com media movel."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_training_log(path):
    """Carrega log de treinamento JSON."""
    with open(path) as f:
        data = json.load(f)
    return data


def extract_metrics(log_data):
    """Extrai metricas de um log de treinamento.

    Suporta dois formatos:
    1. Lista de dicts (um por step)
    2. Dict com listas (historico do trainer)
    """
    if isinstance(log_data, list):
        # Formato: [{step, loss, lr, ...}, ...]
        metrics = {}
        for entry in log_data:
            for key, val in entry.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(val)
        return metrics
    elif isinstance(log_data, dict):
        # Formato: {train_losses: [...], eval_losses: [...], ...}
        return log_data
    return {}


def plot_loss_curves(metrics, output_dir, label=""):
    """Grafico 1: Curvas de loss (train + eval)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    prefix = f"{label} - " if label else ""

    # Train loss
    if "train_losses" in metrics:
        losses = metrics["train_losses"]
        steps = range(len(losses))
        ax.plot(steps, losses, color=COLORS["loss"], alpha=0.3, linewidth=0.5)
        if len(losses) > 20:
            smoothed = smooth(losses, min(50, len(losses) // 5))
            ax.plot(
                range(len(smoothed)), smoothed,
                color=COLORS["loss"], linewidth=2, label="Train Loss (smooth)"
            )
        else:
            ax.plot(steps, losses, color=COLORS["loss"], linewidth=2, label="Train Loss")
    elif "loss" in metrics:
        losses = metrics["loss"]
        steps = metrics.get("step", range(len(losses)))
        ax.plot(steps, losses, color=COLORS["loss"], alpha=0.3, linewidth=0.5)
        if len(losses) > 20:
            smoothed = smooth(losses, min(50, len(losses) // 5))
            ax.plot(
                range(len(smoothed)), smoothed,
                color=COLORS["loss"], linewidth=2, label="Train Loss (smooth)"
            )

    # Eval loss
    if "eval_losses" in metrics:
        evals = metrics["eval_losses"]
        eval_steps = metrics.get("eval_steps", range(len(evals)))
        ax.plot(
            eval_steps, evals,
            color=COLORS["eval_loss"], linewidth=2, marker="o",
            markersize=4, label="Eval Loss"
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{prefix}Curvas de Loss")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] loss_curves.png")


def plot_learning_rate(metrics, output_dir, label=""):
    """Grafico 2: Learning rate schedule."""
    fig, ax = plt.subplots(figsize=(12, 4))

    prefix = f"{label} - " if label else ""

    lr_key = "lr" if "lr" in metrics else "learning_rates"
    if lr_key in metrics:
        lrs = metrics[lr_key]
        steps = metrics.get("step", range(len(lrs)))
        ax.plot(steps, lrs, color=COLORS["lr"], linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{prefix}Learning Rate Schedule")
        ax.grid(True)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(output_dir / "learning_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] learning_rate.png")


def plot_epistemic_metrics(metrics, output_dir, label=""):
    """Grafico 3: Metricas epistemicas (q1, q2, confidence, phi)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    prefix = f"{label} - " if label else ""

    # Q1 (incerteza aleatoria)
    ax = axes[0, 0]
    if "avg_q1" in metrics:
        vals = metrics["avg_q1"]
        ax.plot(range(len(vals)), vals, color=COLORS["q1"], linewidth=1.5, alpha=0.5)
        if len(vals) > 20:
            s = smooth(vals, min(30, len(vals) // 5))
            ax.plot(range(len(s)), s, color=COLORS["q1"], linewidth=2)
    ax.set_title(f"{prefix}Q1 (Incerteza Aleatoria)")
    ax.set_ylabel("Q1")
    ax.set_ylim(0, 1)
    ax.grid(True)

    # Q2 (incerteza epistemica)
    ax = axes[0, 1]
    if "avg_q2" in metrics:
        vals = metrics["avg_q2"]
        ax.plot(range(len(vals)), vals, color=COLORS["q2"], linewidth=1.5, alpha=0.5)
        if len(vals) > 20:
            s = smooth(vals, min(30, len(vals) // 5))
            ax.plot(range(len(s)), s, color=COLORS["q2"], linewidth=2)
    ax.set_title(f"{prefix}Q2 (Incerteza Epistemica)")
    ax.set_ylabel("Q2")
    ax.set_ylim(0, 1)
    ax.grid(True)

    # Confidence (MAD)
    ax = axes[1, 0]
    if "avg_confidence" in metrics:
        vals = metrics["avg_confidence"]
        ax.plot(range(len(vals)), vals, color=COLORS["confidence"], linewidth=1.5, alpha=0.5)
        if len(vals) > 20:
            s = smooth(vals, min(30, len(vals) // 5))
            ax.plot(range(len(s)), s, color=COLORS["confidence"], linewidth=2)
    ax.set_title(f"{prefix}Confidence (MAD)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    ax.grid(True)

    # Phi (saude do manifold)
    ax = axes[1, 1]
    if "avg_phi" in metrics:
        vals = metrics["avg_phi"]
        ax.plot(range(len(vals)), vals, color=COLORS["phi"], linewidth=1.5, alpha=0.5)
        if len(vals) > 20:
            s = smooth(vals, min(30, len(vals) // 5))
            ax.plot(range(len(s)), s, color=COLORS["phi"], linewidth=2)
    ax.set_title(f"{prefix}Phi (Saude do Manifold)")
    ax.set_ylabel("Phi")
    ax.set_ylim(0, 1)
    ax.grid(True)

    for ax in axes.flat:
        ax.set_xlabel("Step")

    fig.suptitle(f"{prefix}Metricas Epistemicas", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "epistemic_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] epistemic_metrics.png")


def plot_loss_components(metrics, output_dir, label=""):
    """Grafico 4: Componentes da loss composta."""
    fig, ax = plt.subplots(figsize=(12, 6))

    prefix = f"{label} - " if label else ""

    components = {
        "ce_loss": ("CE Loss", "#2196F3"),
        "varo_loss": ("VARO Loss", "#9C27B0"),
        "vi_loss": ("VI Reg", "#FF5722"),
        "mad_loss": ("MAD Cal", "#00BCD4"),
        "metric_loss": ("Metric Reg", "#FFC107"),
    }

    found = False
    for key, (name, color) in components.items():
        if key in metrics:
            vals = metrics[key]
            if len(vals) > 20:
                s = smooth(vals, min(30, len(vals) // 5))
                ax.plot(range(len(s)), s, color=color, linewidth=2, label=name)
            else:
                ax.plot(range(len(vals)), vals, color=color, linewidth=2, label=name)
            found = True

    if found:
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{prefix}Componentes da Loss Composta")
        ax.legend()
        ax.grid(True)
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_dir / "loss_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] loss_components.png")


def plot_throughput(metrics, output_dir, label=""):
    """Grafico 5: Throughput (tokens/segundo)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    prefix = f"{label} - " if label else ""

    # Tokens/segundo
    ax = axes[0]
    if "tokens_per_sec" in metrics:
        vals = metrics["tokens_per_sec"]
        ax.plot(range(len(vals)), vals, color=COLORS["tokens_per_sec"], linewidth=1, alpha=0.5)
        if len(vals) > 10:
            s = smooth(vals, min(20, len(vals) // 3))
            ax.plot(range(len(s)), s, color=COLORS["tokens_per_sec"], linewidth=2)
        ax.set_ylabel("Tokens/s")
    ax.set_title(f"{prefix}Throughput")
    ax.set_xlabel("Step")
    ax.grid(True)

    # Gradient norm
    ax = axes[1]
    if "grad_norm" in metrics:
        vals = metrics["grad_norm"]
        ax.plot(range(len(vals)), vals, color=COLORS["grad_norm"], linewidth=1, alpha=0.5)
        if len(vals) > 10:
            s = smooth(vals, min(20, len(vals) // 3))
            ax.plot(range(len(s)), s, color=COLORS["grad_norm"], linewidth=2)
        ax.set_ylabel("Grad Norm")
    ax.set_title(f"{prefix}Gradient Norm")
    ax.set_xlabel("Step")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] throughput.png")


def plot_perplexity(metrics, output_dir, label=""):
    """Grafico 6: Perplexidade."""
    fig, ax = plt.subplots(figsize=(12, 5))

    prefix = f"{label} - " if label else ""

    # Calcula perplexidade a partir da loss
    loss_key = "train_losses" if "train_losses" in metrics else "loss"
    if loss_key in metrics:
        losses = metrics[loss_key]
        perp = [np.exp(min(l, 20)) for l in losses]  # cap para evitar overflow

        ax.plot(range(len(perp)), perp, color=COLORS["perplexity"], alpha=0.3, linewidth=0.5)
        if len(perp) > 20:
            s = smooth(perp, min(50, len(perp) // 5))
            ax.plot(range(len(s)), s, color=COLORS["perplexity"], linewidth=2, label="Train PPL")

    if "eval_losses" in metrics:
        evals = metrics["eval_losses"]
        eval_perp = [np.exp(min(l, 20)) for l in evals]
        eval_steps = metrics.get("eval_steps", range(len(evals)))
        ax.plot(
            eval_steps, eval_perp,
            color=COLORS["eval_loss"], linewidth=2, marker="o",
            markersize=4, label="Eval PPL"
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexidade")
    ax.set_title(f"{prefix}Perplexidade")
    ax.legend()
    ax.grid(True)
    if ax.get_ylim()[1] > 1000:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_dir / "perplexity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] perplexity.png")


def plot_drm_manifold(metrics, output_dir, label=""):
    """Grafico 7: Metricas do DRM manifold."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    prefix = f"{label} - " if label else ""

    # Distancia geodesica media
    ax = axes[0]
    if "avg_geodesic_distance" in metrics:
        vals = metrics["avg_geodesic_distance"]
        ax.plot(range(len(vals)), vals, color="#E91E63", linewidth=1.5)
    ax.set_title(f"{prefix}Distancia Geodesica")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distancia")
    ax.grid(True)

    # Dimensao direcional (dim_D)
    ax = axes[1]
    if "avg_dim_d" in metrics:
        vals = metrics["avg_dim_d"]
        ax.plot(range(len(vals)), vals, color="#9C27B0", linewidth=1.5)
    ax.set_title(f"{prefix}Dim D (Direcional)")
    ax.set_xlabel("Step")
    ax.set_ylabel("dim_D")
    ax.set_ylim(0, 6)
    ax.grid(True)

    # VI severity
    ax = axes[2]
    if "avg_vi_severity" in metrics:
        vals = metrics["avg_vi_severity"]
        ax.plot(range(len(vals)), vals, color=COLORS["vi_severity"], linewidth=1.5)
    ax.set_title(f"{prefix}VI Severity")
    ax.set_xlabel("Step")
    ax.set_ylabel("Severity")
    ax.set_ylim(0, 1)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "drm_manifold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] drm_manifold.png")


def plot_comparison(all_metrics, labels, output_dir):
    """Grafico 8: Comparacao entre escalas."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_metrics)))

    # Loss final por escala
    ax = axes[0, 0]
    final_losses = []
    for i, (m, lbl) in enumerate(zip(all_metrics, labels)):
        loss_key = "train_losses" if "train_losses" in m else "loss"
        if loss_key in m:
            final_losses.append(m[loss_key][-1])
        else:
            final_losses.append(0)
    ax.bar(labels, final_losses, color=colors)
    ax.set_title("Loss Final por Escala")
    ax.set_ylabel("Loss")
    ax.grid(True, axis="y")

    # Perplexidade final por escala
    ax = axes[0, 1]
    final_perp = [np.exp(min(l, 20)) for l in final_losses]
    ax.bar(labels, final_perp, color=colors)
    ax.set_title("Perplexidade Final por Escala")
    ax.set_ylabel("Perplexidade")
    ax.grid(True, axis="y")

    # Curvas de loss sobrepostas
    ax = axes[1, 0]
    for i, (m, lbl) in enumerate(zip(all_metrics, labels)):
        loss_key = "train_losses" if "train_losses" in m else "loss"
        if loss_key in m:
            vals = m[loss_key]
            # Normaliza steps para [0, 1] para comparacao
            x = np.linspace(0, 1, len(vals))
            if len(vals) > 20:
                s = smooth(vals, min(30, len(vals) // 5))
                x_s = np.linspace(0, 1, len(s))
                ax.plot(x_s, s, color=colors[i], linewidth=2, label=lbl)
            else:
                ax.plot(x, vals, color=colors[i], linewidth=2, label=lbl)
    ax.set_title("Curvas de Loss Normalizadas")
    ax.set_xlabel("Progresso (%)")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # Confidence final por escala
    ax = axes[1, 1]
    final_conf = []
    for m in all_metrics:
        if "avg_confidence" in m:
            final_conf.append(m["avg_confidence"][-1])
        else:
            final_conf.append(0)
    ax.bar(labels, final_conf, color=colors)
    ax.set_title("Confidence Final por Escala")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y")

    fig.suptitle("Comparacao entre Escalas", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] comparison.png")


def plot_dashboard(metrics, output_dir, label=""):
    """Grafico 9: Dashboard completo (tudo em um)."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.3)

    prefix = f"{label} - " if label else ""

    def _plot(ax, key, title, color, ylim=None):
        if key in metrics:
            vals = metrics[key]
            ax.plot(range(len(vals)), vals, color=color, alpha=0.3, linewidth=0.5)
            if len(vals) > 20:
                s = smooth(vals, min(30, len(vals) // 5))
                ax.plot(range(len(s)), s, color=color, linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.grid(True)
        if ylim:
            ax.set_ylim(ylim)

    # Linha 1: Loss, Perplexidade, LR, Throughput
    loss_key = "train_losses" if "train_losses" in metrics else "loss"
    ax = fig.add_subplot(gs[0, 0])
    _plot(ax, loss_key, "Train Loss", COLORS["loss"])

    ax = fig.add_subplot(gs[0, 1])
    if loss_key in metrics:
        perp = [np.exp(min(l, 20)) for l in metrics[loss_key]]
        metrics["_perp"] = perp
    _plot(ax, "_perp", "Perplexidade", COLORS["perplexity"])

    ax = fig.add_subplot(gs[0, 2])
    lr_key = "lr" if "lr" in metrics else "learning_rates"
    _plot(ax, lr_key, "Learning Rate", COLORS["lr"])

    ax = fig.add_subplot(gs[0, 3])
    _plot(ax, "tokens_per_sec", "Tokens/s", COLORS["tokens_per_sec"])

    # Linha 2: Q1, Q2, Confidence, Phi
    ax = fig.add_subplot(gs[1, 0])
    _plot(ax, "avg_q1", "Q1 (Aleatoria)", COLORS["q1"], (0, 1))

    ax = fig.add_subplot(gs[1, 1])
    _plot(ax, "avg_q2", "Q2 (Epistemica)", COLORS["q2"], (0, 1))

    ax = fig.add_subplot(gs[1, 2])
    _plot(ax, "avg_confidence", "Confidence (MAD)", COLORS["confidence"], (0, 1))

    ax = fig.add_subplot(gs[1, 3])
    _plot(ax, "avg_phi", "Phi (Manifold)", COLORS["phi"], (0, 1))

    # Linha 3: Loss components, Grad norm, DRM, VI
    ax = fig.add_subplot(gs[2, 0])
    for key, (name, color) in [
        ("ce_loss", ("CE", "#2196F3")),
        ("varo_loss", ("VARO", "#9C27B0")),
        ("vi_loss", ("VI", "#FF5722")),
        ("mad_loss", ("MAD", "#00BCD4")),
    ]:
        if key in metrics:
            vals = metrics[key]
            if len(vals) > 20:
                s = smooth(vals, min(20, len(vals) // 5))
                ax.plot(range(len(s)), s, linewidth=1.5, label=name, color=color)
    ax.set_title("Loss Components", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True)
    if any(k in metrics for k in ["ce_loss", "varo_loss"]):
        ax.set_yscale("log")

    ax = fig.add_subplot(gs[2, 1])
    _plot(ax, "grad_norm", "Grad Norm", COLORS["grad_norm"])

    ax = fig.add_subplot(gs[2, 2])
    _plot(ax, "avg_geodesic_distance", "Geodesic Dist", "#E91E63")

    ax = fig.add_subplot(gs[2, 3])
    _plot(ax, "avg_vi_severity", "VI Severity", COLORS["vi_severity"], (0, 1))

    # Limpa metrica temporaria
    metrics.pop("_perp", None)

    fig.suptitle(f"{prefix}AletheionV2 Training Dashboard", fontsize=16, y=1.01)
    fig.savefig(output_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] dashboard.png")


def generate_demo_data(num_steps=2000):
    """Gera dados sinteticos para demonstracao."""
    steps = np.arange(num_steps)
    progress = steps / num_steps

    # Loss: decai exponencialmente com ruido
    base_loss = 10.0 * np.exp(-3.0 * progress) + 2.0
    noise = np.random.normal(0, 0.1, num_steps) * (1 - 0.5 * progress)
    train_losses = (base_loss + noise).clip(1.0).tolist()

    # Eval loss (a cada 100 steps)
    eval_steps = list(range(0, num_steps, 100))
    eval_losses = [
        float(10.0 * np.exp(-3.0 * s / num_steps) + 2.1 + np.random.normal(0, 0.05))
        for s in eval_steps
    ]

    # LR: warmup + cosine decay
    warmup = 200
    lr_max = 3e-4
    lrs = []
    for s in steps:
        if s < warmup:
            lrs.append(float(lr_max * s / warmup))
        else:
            p = (s - warmup) / (num_steps - warmup)
            lrs.append(float(lr_max * 0.5 * (1 + np.cos(np.pi * p))))

    # Q1: comeca alto, estabiliza
    q1 = (0.7 - 0.3 * progress + np.random.normal(0, 0.02, num_steps)).clip(0, 1).tolist()

    # Q2: comeca alto, cai conforme modelo aprende
    q2 = (0.8 - 0.5 * progress + np.random.normal(0, 0.03, num_steps)).clip(0, 1).tolist()

    # Confidence: cresce com treinamento
    conf = (0.2 + 0.6 * progress + np.random.normal(0, 0.02, num_steps)).clip(0, 1).tolist()

    # Phi: estabiliza acima do critico
    phi = (0.3 + 0.4 * progress + np.random.normal(0, 0.03, num_steps)).clip(0, 1).tolist()

    # Loss components
    ce = train_losses
    varo = (0.5 * np.exp(-2 * progress) + np.random.normal(0, 0.01, num_steps)).clip(0).tolist()
    vi_loss = (0.3 * np.exp(-1.5 * progress) + np.random.normal(0, 0.005, num_steps)).clip(0).tolist()
    mad = (0.4 * np.exp(-2.5 * progress) + np.random.normal(0, 0.01, num_steps)).clip(0).tolist()

    # Throughput
    base_tps = 15000
    tps = (base_tps + 2000 * np.random.randn(num_steps)).clip(5000).tolist()

    # Grad norm
    gn = (2.0 * np.exp(-progress) + 0.5 + np.random.normal(0, 0.1, num_steps)).clip(0).tolist()

    # DRM
    geo_dist = (3.0 - 2.0 * progress + np.random.normal(0, 0.1, num_steps)).clip(0).tolist()
    dim_d = (1.5 + 2.0 * progress + np.random.normal(0, 0.1, num_steps)).clip(1, 5).tolist()
    vi_sev = (0.7 - 0.5 * progress + np.random.normal(0, 0.03, num_steps)).clip(0, 1).tolist()

    return {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "eval_steps": eval_steps,
        "learning_rates": lrs,
        "avg_q1": q1,
        "avg_q2": q2,
        "avg_confidence": conf,
        "avg_phi": phi,
        "ce_loss": ce,
        "varo_loss": varo,
        "vi_loss": vi_loss,
        "mad_loss": mad,
        "tokens_per_sec": tps,
        "grad_norm": gn,
        "avg_geodesic_distance": geo_dist,
        "avg_dim_d": dim_d,
        "avg_vi_severity": vi_sev,
    }


def generate_all_plots(metrics, output_dir, label=""):
    """Gera todos os graficos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PLOT] Gerando graficos em {output_dir}/")

    plot_loss_curves(metrics, output_dir, label)
    plot_learning_rate(metrics, output_dir, label)
    plot_epistemic_metrics(metrics, output_dir, label)
    plot_loss_components(metrics, output_dir, label)
    plot_throughput(metrics, output_dir, label)
    plot_perplexity(metrics, output_dir, label)
    plot_drm_manifold(metrics, output_dir, label)
    plot_dashboard(metrics, output_dir, label)

    print(f"\n[OK] 8 graficos gerados em {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualizacao de treinamento AletheionV2")
    parser.add_argument(
        "--log", nargs="+", default=[],
        help="Caminho(s) para log(s) JSON de treinamento"
    )
    parser.add_argument(
        "--labels", nargs="+", default=[],
        help="Labels para cada log (para comparacao)"
    )
    parser.add_argument(
        "--output", default="plots",
        help="Diretorio de saida dos graficos"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Gera graficos com dados sinteticos de demonstracao"
    )
    parser.add_argument(
        "--demo-compare", action="store_true",
        help="Gera graficos de comparacao com dados sinteticos"
    )

    args = parser.parse_args()
    apply_style()

    if args.demo:
        print("[DEMO] Gerando dados sinteticos...")
        metrics = generate_demo_data(2000)
        generate_all_plots(metrics, args.output, label="Demo 125M")

        # Salva dados sinteticos para referencia
        demo_path = Path(args.output) / "demo_data.json"
        with open(demo_path, "w") as f:
            json.dump(metrics, f)
        print(f"[OK] Dados sinteticos salvos em {demo_path}")
        return

    if args.demo_compare:
        print("[DEMO] Gerando dados sinteticos para comparacao...")
        scales = ["1M", "10M", "50M", "125M"]
        all_metrics = []
        for i, scale in enumerate(scales):
            np.random.seed(42 + i)
            m = generate_demo_data(500 * (i + 1))
            # Modelos maiores tem loss menor
            factor = 1.0 - 0.15 * i
            m["train_losses"] = [l * factor for l in m["train_losses"]]
            all_metrics.append(m)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        apply_style()
        plot_comparison(all_metrics, scales, output_dir)

        # Gera dashboard individual para cada escala
        for m, scale in zip(all_metrics, scales):
            scale_dir = output_dir / scale
            scale_dir.mkdir(exist_ok=True)
            generate_all_plots(m, scale_dir, label=scale)

        print(f"\n[OK] Comparacao + dashboards individuais em {output_dir}/")
        return

    if not args.log:
        print("[ERRO] Use --log <arquivo.json> ou --demo para graficos de demonstracao")
        print("       Use --demo-compare para comparacao entre escalas")
        sys.exit(1)

    if len(args.log) == 1:
        # Log unico
        metrics = extract_metrics(load_training_log(args.log[0]))
        label = args.labels[0] if args.labels else ""
        generate_all_plots(metrics, args.output, label)
    else:
        # Multiplos logs (comparacao)
        all_metrics = []
        labels = args.labels if args.labels else [
            Path(p).stem for p in args.log
        ]
        for log_path in args.log:
            m = extract_metrics(load_training_log(log_path))
            all_metrics.append(m)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(all_metrics, labels, output_dir)

        # Graficos individuais tambem
        for m, lbl in zip(all_metrics, labels):
            ind_dir = output_dir / lbl
            ind_dir.mkdir(exist_ok=True)
            generate_all_plots(m, ind_dir, label=lbl)


if __name__ == "__main__":
    main()
