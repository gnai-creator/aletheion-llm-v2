"""
Visualizacao dos resultados de Voronoi foliation.

Gera: T^2 overlay, UMAP, espectro eigenvalues, heatmap coerencia,
Reeb graph, diagrama persistencia, dashboard resumo.

Uso:
    .venv/bin/python scripts/plot_foliation.py \
        --results-dir eval_results/foliation
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DARK_THEME = {
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

COLORS = [
    "#58a6ff", "#3fb950", "#f0883e", "#f85149", "#bc8cff",
    "#79c0ff", "#56d364", "#d29922", "#ff7b72", "#d2a8ff",
    "#a5d6ff", "#7ee787", "#e3b341", "#ffa198", "#e8d5ff",
    "#39d353", "#db6d28", "#da3633", "#8b949e", "#c9d1d9",
    "#1f6feb", "#238636", "#9e6a03", "#cf222e", "#8957e5",
    "#388bfd", "#2ea043", "#bb8009", "#a40e26", "#6e40c9",
]


def get_plt():
    """Retorna matplotlib.pyplot com backend Agg e tema dark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(DARK_THEME)
    return plt


def plot_torus_overlay(
    vectors: np.ndarray, labels: np.ndarray, path: str, max_pts: int = 50000,
) -> None:
    """Celulas Voronoi na projecao toroidal q1 x q2."""
    plt = get_plt()
    if len(vectors) > max_pts:
        idx = np.random.choice(len(vectors), max_pts, replace=False)
        vectors, labels = vectors[idx], labels[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, lbl in enumerate(np.unique(labels)):
        m = labels == lbl
        ax.scatter(vectors[m, 0], vectors[m, 1], c=COLORS[i % len(COLORS)],
                   s=1, alpha=0.4, rasterized=True)
    ax.set(xlabel="q1 (aleatoric)", ylabel="q2 (epistemic)", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Voronoi Cells on T^2 (q1 x q2)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] %s", path)


def plot_umap(
    vectors: np.ndarray, labels: np.ndarray, path: str, max_pts: int = 30000,
) -> None:
    """UMAP 2D colorido por celula Voronoi e confidence."""
    try:
        import umap
    except ImportError:
        logger.warning("[WARN] umap-learn nao instalado, pulando UMAP")
        return

    plt = get_plt()
    if len(vectors) > max_pts:
        idx = np.random.choice(len(vectors), max_pts, replace=False)
        vectors, labels = vectors[idx], labels[idx]

    logger.info("[UMAP] Computando embedding 2D (%d pontos)...", len(vectors))
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(vectors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    for i, lbl in enumerate(np.unique(labels)):
        m = labels == lbl
        ax1.scatter(emb[m, 0], emb[m, 1], c=COLORS[i % len(COLORS)], s=2, alpha=0.5, rasterized=True)
    ax1.set_title("UMAP: Voronoi Cells", fontsize=12, fontweight="bold")

    sc = ax2.scatter(emb[:, 0], emb[:, 1], c=vectors[:, 2], cmap="viridis", s=2, alpha=0.5, rasterized=True)
    fig.colorbar(sc, ax=ax2, label="confidence")
    ax2.set_title("UMAP: Confidence", fontsize=12, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set(xlabel="UMAP-1", ylabel="UMAP-2")
    fig.suptitle("AletheionV2 - Epistemic Manifold", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] %s", path)


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray, eff_dims: np.ndarray, path: str,
) -> None:
    """Espectro de eigenvalues e distribuicao de dimensionalidade."""
    plt = get_plt()
    k, d = eigenvalues.shape
    x = np.arange(1, d + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ev_norm = eigenvalues / (eigenvalues.sum(axis=1, keepdims=True) + 1e-10)
    for i in range(k):
        ax1.plot(x, ev_norm[i], "-o", color=COLORS[i % len(COLORS)], alpha=0.4, markersize=3, linewidth=1)
    ax1.plot(x, ev_norm.mean(axis=0), "w-o", linewidth=3, markersize=6, label="Mean")
    ax1.set(xlabel="Component Index", ylabel="Normalized Eigenvalue", xticks=x)
    ax1.set_title("Eigenvalue Spectrum per Cell", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ud, uc = np.unique(eff_dims, return_counts=True)
    ax2.bar(ud, uc, color=[COLORS[(d - 1) % len(COLORS)] for d in ud], alpha=0.8, edgecolor="#30363d")
    for dim, cnt in zip(ud, uc):
        ax2.text(dim, cnt + 0.3, str(cnt), ha="center", fontsize=10, fontweight="bold", color="#c9d1d9")
    ax2.set(xlabel="Effective Dimension", ylabel="Number of Cells", xticks=range(1, 6))
    ax2.set_title("Effective Dimensionality", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("AletheionV2 - LTSA Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] %s", path)


def plot_coherence_heatmap(coherence: np.ndarray, path: str) -> None:
    """Heatmap de coerencia tangencial entre celulas."""
    plt = get_plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("coh", ["#3fb950", "#d29922", "#f85149"])

    fig, ax = plt.subplots(figsize=(10, 8))
    display = coherence.copy()
    display[np.isnan(display)] = -1
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=90, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Principal Angle (degrees)")
    ax.set(xlabel="Cell Index", ylabel="Cell Index")
    ax.set_title("Tangent Coherence (green=aligned, red=misaligned)", fontsize=13, fontweight="bold")

    valid = coherence[~np.isnan(coherence)]
    if len(valid) > 0:
        ax.text(0.5, -0.08, f"Mean: {valid.mean():.1f} | Median: {np.median(valid):.1f} | <30: {(valid < 30).mean():.1%}",
                transform=ax.transAxes, ha="center", fontsize=10, color="#8b949e")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] %s", path)


def plot_reeb_graph(reeb_data: Dict, path: str) -> None:
    """Reeb graph como diagrama de nos e arestas."""
    plt = get_plt()
    nodes = reeb_data.get("nodes", [])
    edges = reeb_data.get("edges", [])
    if not nodes:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Posicionar nos
    level_counts: Dict[int, int] = {}
    raw_pos = []
    for node in nodes:
        lvl = node["level"]
        x = level_counts.get(lvl, 0)
        level_counts[lvl] = x + 1
        raw_pos.append((lvl, x, node["level_value"]))

    positions = []
    for lvl, x, y in raw_pos:
        n_lvl = level_counts[lvl]
        x_norm = (x + 0.5) / n_lvl if n_lvl > 1 else 0.5
        positions.append((x_norm, y))

    for src, dst in edges:
        if src < len(positions) and dst < len(positions):
            ax.plot([positions[src][0], positions[dst][0]],
                    [positions[src][1], positions[dst][1]], "-", color="#58a6ff", alpha=0.3)

    sizes = [min(max(n["n_points"] / 10, 10), 200) for n in nodes]
    ax.scatter([p[0] for p in positions], [p[1] for p in positions],
               s=sizes, c="#3fb950", alpha=0.7, edgecolors="#30363d", zorder=5)

    fname = reeb_data.get("func_name", "f")
    ax.set(xlabel="Component (norm)", ylabel=f"{fname} value")
    ax.set_title(f"Reeb Graph: {fname} ({len(nodes)} nodes, {len(edges)} edges)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] %s", path)


def plot_persistence_diagram(hom_data: Dict, path: str) -> None:
    """Diagrama de persistencia."""
    plt = get_plt()
    diagrams = hom_data.get("diagrams", [])
    if not diagrams:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    dim_colors = ["#58a6ff", "#3fb950", "#f0883e"]
    dim_labels = ["H0", "H1", "H2"]
    all_finite = []

    for dim, dgm in enumerate(diagrams):
        dgm = np.array(dgm)
        if len(dgm) == 0:
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) > 0:
            all_finite.extend(finite.flatten().tolist())
            c = dim_colors[dim] if dim < len(dim_colors) else "#8b949e"
            lbl = dim_labels[dim] if dim < len(dim_labels) else f"H{dim}"
            ax.scatter(finite[:, 0], finite[:, 1], c=c, s=20, alpha=0.7,
                       label=f"{lbl} ({len(dgm)})", zorder=3)

    if all_finite:
        lim = max(all_finite) * 1.1
        ax.plot([0, lim], [0, lim], "w--", alpha=0.3)
        ax.set(xlim=(0, lim), ylim=(0, lim))

    ax.set(xlabel="Birth", ylabel="Death", aspect="equal")
    ax.set_title(f"Persistence Diagram\n{hom_data.get('topology', '')}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] %s", path)


def plot_summary(results: Dict, path: str) -> None:
    """Dashboard resumo."""
    plt = get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    F = results.get("foliation_score", 0.0)
    ax = axes[0, 0]
    ax.barh(["F Score"], [F], color="#3fb950" if F > 0.3 else "#f0883e", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_title(f"Foliation Score: {F:.4f}", fontsize=13, fontweight="bold")
    ax.axvline(0.3, color="#f85149", ls="--", alpha=0.5)

    ari = results.get("stability", {}).get("mean_ari", 0)
    ax = axes[0, 1]
    ax.barh(["ARI"], [ari], color="#58a6ff", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_title(f"Stability ARI: {ari:.3f}", fontsize=13, fontweight="bold")
    ax.axvline(0.7, color="#3fb950", ls="--", alpha=0.5)

    ax = axes[1, 0]
    coh = results.get("coherence", {})
    null = results.get("null_models", {})
    names = ["Model"]
    vals = [coh.get("coherent_fraction", 0)]
    for nm, nd in null.items():
        if isinstance(nd, dict) and "coherent_fraction" in nd:
            names.append(nm.capitalize())
            vals.append(nd["coherent_fraction"])
    ax.bar(names, vals, color=["#3fb950"] + ["#8b949e"] * (len(names) - 1), alpha=0.8)
    ax.set_ylabel("Coherent Fraction")
    ax.set_title("Model vs Nulls", fontsize=13, fontweight="bold")

    ax = axes[1, 1]
    hom = results.get("homology", {})
    lines = [f"Topology: {hom.get('topology', 'N/A')}"]
    for key in ["H0", "H1", "H2"]:
        h = hom.get("homology", {}).get(key, {})
        lines.append(f"  {key}: {h.get('n_features', 0)} feat, {h.get('long_bars', 0)} long")
    ax.text(0.1, 0.5, "\n".join(lines), transform=ax.transAxes, fontsize=12,
            va="center", fontfamily="monospace", color="#c9d1d9")
    ax.set_title("Persistent Homology", fontsize=13, fontweight="bold")
    ax.set(xticks=[], yticks=[])

    fig.suptitle("AletheionV2 - Voronoi Foliation Summary", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[OK] %s", path)


def main() -> None:
    """Gera todas as visualizacoes de foliation."""
    parser = argparse.ArgumentParser(description="Plots de Voronoi foliation")
    parser.add_argument("--results-dir", default="eval_results/foliation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    rd = Path(args.results_dir)
    od = Path(args.output_dir) if args.output_dir else rd / "plots"
    od.mkdir(parents=True, exist_ok=True)

    vp, lp = rd / "finetuned_vectors.npy", rd / "voronoi_labels.npy"
    if not vp.exists() or not lp.exists():
        logger.error("[ERROR] Arquivos nao encontrados: %s, %s", vp, lp)
        sys.exit(1)

    vectors = np.load(str(vp))
    labels = np.load(str(lp))
    logger.info("[LOAD] Vectors: %s, Labels: %s", vectors.shape, labels.shape)

    plot_torus_overlay(vectors, labels, str(od / "torus_overlay.png"))
    plot_umap(vectors, labels, str(od / "umap_cells.png"))

    ev_path, ed_path = rd / "eigenvalues.npy", rd / "eff_dims.npy"
    if ev_path.exists() and ed_path.exists():
        plot_eigenvalue_spectrum(np.load(str(ev_path)), np.load(str(ed_path)),
                                str(od / "eigenvalue_spectrum.png"))

    coh_path = rd / "coherence_matrix.npy"
    if coh_path.exists():
        plot_coherence_heatmap(np.load(str(coh_path)), str(od / "coherence_heatmap.png"))

    for fn in ["confidence", "phi"]:
        rp = rd / f"reeb_{fn}.json"
        if rp.exists():
            with open(str(rp)) as f:
                plot_reeb_graph(json.load(f), str(od / f"reeb_{fn}.png"))

    hp = rd / "homology.json"
    if hp.exists():
        with open(str(hp)) as f:
            plot_persistence_diagram(json.load(f), str(od / "persistence_diagram.png"))

    rp = rd / "foliation_results.json"
    if rp.exists():
        with open(str(rp)) as f:
            plot_summary(json.load(f), str(od / "foliation_summary.png"))

    logger.info("[OK] Todas as visualizacoes em %s", od)


if __name__ == "__main__":
    main()
