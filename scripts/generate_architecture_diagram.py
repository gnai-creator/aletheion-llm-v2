"""
Gera diagrama de arquitetura do AletheionV2.

Organizado em 3 blocos independentes:
  1. draw_backbone(ax, x, y, w, h) — caixa preta com stack do transformer
  2. draw_epistemic(ax, x, y, w, h) — caixa tracejada com todos os sub-blocos
  3. draw_footer(ax, x, y, w) — composite loss + branches + resources

Cada bloco recebe suas coordenadas e nao depende dos outros.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Cores
WHITE = "#ffffff"
BLACK = "#1a1a1a"
GRAY_BG = "#2d2d2d"
GRAY_LIGHT = "#e8e8e8"
GRAY_MED = "#888888"
GRAY_DARK = "#555555"
PURPLE = "#9b59b6"
INDIGO = "#5c6bc0"
TEAL = "#1abc9c"
GREEN = "#4caf50"
MAGENTA = "#e91e63"
AMBER = "#ff9800"
CYAN = "#00bcd4"
ORANGE = "#e67e22"


# ============================================================================
# Primitivas
# ============================================================================

def rect(ax, x, y, w, h, label, color=WHITE, border=BLACK, tc=BLACK,
         fs=7, zorder=2):
    """Bloco retangular com label centralizado."""
    r = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03",
        facecolor=color, edgecolor=border, linewidth=0.9, zorder=zorder,
    )
    ax.add_patch(r)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center", fontsize=fs, color=tc,
        fontweight="bold", fontfamily="sans-serif", zorder=zorder + 1,
    )


def container(ax, x, y, w, h, color, border, lw=1.5, ls="-", zorder=0):
    """Caixa de fundo (container)."""
    r = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=border, linewidth=lw,
        linestyle=ls, zorder=zorder,
    )
    ax.add_patch(r)


# ============================================================================
# Stack generico para epistemic sub-blocos
# ============================================================================

def draw_stack(ax, x, y_top, w, rows, gap=0.08):
    """Desenha uma stack vertical de blocos.

    Args:
        ax: matplotlib axes
        x: left x
        y_top: top y (primeiro bloco comeca aqui)
        w: largura total disponivel
        rows: lista de dicts, cada um com:
            - "cols": lista de dicts com "label", "color", "border"
            - "h": altura dos blocos nesta row
            - "header": texto de header acima da row (opcional)
            - "header_color": cor do header (opcional)
            - "fs": fontsize (opcional, default 6)
        gap: espaco vertical entre rows

    Returns:
        y_bottom: coordenada Y apos o ultimo bloco
    """
    y = y_top

    for row in rows:
        # Header opcional
        if "header" in row:
            ax.text(
                x + 0.02, y, row["header"],
                ha="left", va="bottom",
                fontsize=5, fontweight="bold",
                color=row.get("header_color", GRAY_MED),
            )
            y -= 0.12

        h = row.get("h", 0.32)
        fs = row.get("fs", 6)
        cols = row["cols"]
        n = len(cols)
        col_gap = 0.12
        col_w = (w - col_gap * (n - 1)) / n

        for i, col in enumerate(cols):
            cx = x + i * (col_w + col_gap)
            rect(
                ax, cx, y - h, col_w, h,
                col["label"],
                color=col.get("color", WHITE),
                border=col.get("border", BLACK),
                tc=col.get("tc", BLACK),
                fs=fs,
            )

        y -= h + gap

    return y


# ============================================================================
# Bloco 1: Backbone (caixa preta)
# ============================================================================

def draw_backbone(ax, x, y, w, h):
    """Desenha o backbone do transformer na area (x, y, w, h).

    y e o BOTTOM da caixa, h e a altura total.
    """
    top = y + h

    # Container escuro com mais contraste
    container(ax, x, y, w, h, "#1a1a2e", "#444466", lw=2.0)

    # Stack interna
    cx = x + 0.15
    cw = w - 0.3
    mid = cx + cw / 2

    # Cores com contraste
    C_MAIN = "#3a3a5c"
    C_MAIN_B = "#7070aa"
    C_SMALL = "#2e2e4a"
    C_SMALL_B = "#5a5a8a"

    blocks = [
        ("Token Embedding",      0.36, C_MAIN, C_MAIN_B, 7),
        ("Multi-Head Attention",  0.36, C_MAIN, C_MAIN_B, 7),
        ("RMSNorm (fp32)",       0.26, C_SMALL, C_SMALL_B, 6),
        ("Feed-Forward",         0.36, C_MAIN, C_MAIN_B, 7),
        ("RMSNorm (fp32)",       0.26, C_SMALL, C_SMALL_B, 6),
        ("Final LayerNorm",      0.26, C_SMALL, C_SMALL_B, 6),
        ("Linear Output",        0.36, C_MAIN, C_MAIN_B, 7),
    ]

    # Distribuir uniformemente na altura
    total_bh = sum(bh_i for _, bh_i, _, _, _ in blocks)
    margin_top = 0.45  # space for input label
    margin_bot = 0.35  # space for logits label
    avail = h - margin_top - margin_bot
    n_gaps = len(blocks) - 1
    g = (avail - total_bh) / max(n_gaps, 1)

    # Input label
    ax.text(mid, top - 0.15, "input_ids [B, T]",
            ha="center", fontsize=6, color="#bbbbdd", fontweight="bold")

    cy = top - margin_top

    for i, (label, bh_i, bg, bd, fs) in enumerate(blocks):
        cy -= bh_i
        rect(ax, cx, cy, cw, bh_i, label, bg, bd, WHITE, fs)

        # Seta entre blocos
        if i < len(blocks) - 1:
            ax.annotate(
                "", xy=(mid, cy - g + 0.02), xytext=(mid, cy - 0.02),
                arrowprops=dict(arrowstyle="-|>", color="#8888bb", lw=0.8),
            )

        # + circle para residual (apos Attention e FFN)
        if label in ("Multi-Head Attention", "Feed-Forward"):
            py = cy - g / 2
            ax.plot(mid, py, "o", color="#1a1a2e", markersize=8,
                    markeredgecolor="#aaaacc", markeredgewidth=1.0, zorder=5)
            ax.text(mid, py, "+", ha="center", va="center",
                    fontsize=7, color="#ccccee", fontweight="bold", zorder=6)

        cy -= g

    # logits label
    ax.text(mid, y + 0.12, "logits [B, T, 50257]",
            ha="center", fontsize=5.5, color="#9999bb")

    # 24x annotation
    ax.text(x + w + 0.12, top - 1.0, "24x",
            ha="left", fontsize=11, fontweight="bold", color=GRAY_DARK)
    ax.text(x + w + 0.12, top - 1.25, "layers",
            ha="left", fontsize=6, color=GRAY_MED)


# ============================================================================
# Bloco 2: EpistemicHead (caixa tracejada roxa)
# ============================================================================

def draw_epistemic(ax, x, y, w, h):
    """Desenha o EpistemicHead na area (x, y, w, h).

    y e o BOTTOM, h e a altura total.
    Blocos distribuidos uniformemente pela altura disponivel.
    """
    top = y + h

    # Container tracejado
    container(ax, x, y, w, h, "#faf5ff", PURPLE, lw=1.8, ls="--", zorder=0)

    # Titulo
    ax.text(x + w / 2, top - 0.12, "EpistemicHead (fp32)",
            ha="center", fontsize=10, fontweight="bold", color=PURPLE, zorder=5)
    ax.text(x + w / 2, top - 0.3,
            "~2.2M params (1.8%) | 30+ fields per token",
            ha="center", fontsize=5.5, color=GRAY_MED, zorder=5)

    # Area interna
    ix = x + 0.12
    iw = w - 0.24

    # Todas as rows com tipo (para DRM box)
    all_rows = [
        # Core
        {"cols": [
            {"label": "Q1Gate (aleatoric)", "color": "#ede7f6", "border": PURPLE},
            {"label": "Q2Gate (epistemic)", "color": "#ede7f6", "border": PURPLE},
        ], "h": 0.3, "fs": 6, "section": "core"},

        {"cols": [
            {"label": "AdaptiveTemperature", "color": "#ede7f6", "border": PURPLE},
        ], "h": 0.26, "fs": 6, "section": "core"},

        # DRM
        {"cols": [
            {"label": "ManifoldEmbedding [B,T,5]", "color": "#e8eaf6", "border": INDIGO},
        ], "h": 0.3, "fs": 6, "section": "drm"},

        {"cols": [
            {"label": "MetricNet G(x)", "color": "#e8eaf6", "border": INDIGO},
            {"label": "LearnableMetric G", "color": "#e8eaf6", "border": INDIGO},
        ], "h": 0.28, "fs": 5.5, "section": "drm"},

        {"cols": [
            {"label": "DirectionalField", "color": "#e8eaf6", "border": INDIGO},
            {"label": "GeodesicDistance", "color": "#e8eaf6", "border": INDIGO},
        ], "h": 0.26, "fs": 5.5, "section": "drm"},

        # Post-DRM
        {"cols": [
            {"label": "MADConfidence (BayesianTau)", "color": "#e0f2f1", "border": TEAL},
        ], "h": 0.28, "fs": 6, "section": "post"},

        {"cols": [
            {"label": "PhiField", "color": "#e8f5e9", "border": GREEN},
            {"label": "IntentionalityVector", "color": "#e8f5e9", "border": GREEN},
        ], "h": 0.26, "fs": 5.5, "section": "post"},

        # Tier 1
        {"header": "Tier 1", "header_color": MAGENTA, "cols": [
            {"label": "EidosDecay", "color": "#fce4ec", "border": MAGENTA},
            {"label": "Filosofia3", "color": "#fce4ec", "border": MAGENTA},
            {"label": "SelfModel", "color": "#fce4ec", "border": MAGENTA},
        ], "h": 0.26, "fs": 5.5, "section": "tier"},

        # Tier 2
        {"header": "Tier 2", "header_color": AMBER, "cols": [
            {"label": "Grounding", "color": "#fff8e1", "border": AMBER},
            {"label": "Plasticity", "color": "#fff8e1", "border": AMBER},
            {"label": "MPL/Frontier", "color": "#fff8e1", "border": AMBER},
        ], "h": 0.26, "fs": 5.5, "section": "tier"},

        # Tier 3
        {"header": "Tier 3", "header_color": CYAN, "cols": [
            {"label": "MOPsi", "color": "#e0f7fa", "border": CYAN},
            {"label": "CausalState", "color": "#e0f7fa", "border": CYAN},
            {"label": "Metacognitive", "color": "#e0f7fa", "border": CYAN},
        ], "h": 0.26, "fs": 5.5, "section": "tier"},

        # GravityField
        {"cols": [
            {"label": "GravityField", "color": "#f3e5f5", "border": PURPLE},
        ], "h": 0.26, "fs": 6, "section": "post"},

        # Output
        {"cols": [
            {"label": "EpistemicTomography (30+ fields)", "color": GRAY_LIGHT, "border": BLACK},
        ], "h": 0.3, "fs": 6, "section": "output"},
    ]

    # Calcular gap automaticamente para distribuir na altura
    margin_top = 0.5   # espaço para titulo
    margin_bot = 0.2   # margem inferior
    header_extra = sum(0.14 for r in all_rows if "header" in r)
    total_block_h = sum(r["h"] for r in all_rows) + header_extra
    avail = h - margin_top - margin_bot
    n_gaps = len(all_rows) - 1
    gap = (avail - total_block_h) / max(n_gaps, 1)

    # Desenhar rows com gap uniforme
    cy = top - margin_top
    drm_top_y = None
    drm_bot_y = None

    for row in all_rows:
        # Track DRM boundaries
        if row.get("section") == "drm" and drm_top_y is None:
            drm_top_y = cy + 0.06

        # Header
        if "header" in row:
            ax.text(ix + 0.02, cy, row["header"],
                    ha="left", va="bottom", fontsize=5, fontweight="bold",
                    color=row.get("header_color", GRAY_MED))
            cy -= 0.14

        # Cols
        rh = row["h"]
        fs = row.get("fs", 6)
        cols = row["cols"]
        n = len(cols)
        col_gap = 0.12
        col_w = (iw - col_gap * (n - 1)) / n

        for i, col in enumerate(cols):
            cx_col = ix + i * (col_w + col_gap)
            rect(ax, cx_col, cy - rh, col_w, rh,
                 col["label"], col.get("color", WHITE),
                 col.get("border", BLACK), col.get("tc", BLACK), fs)

        if row.get("section") == "drm":
            drm_bot_y = cy - rh - 0.04

        cy -= rh + gap

    # Draw DRM dotted container
    if drm_top_y and drm_bot_y:
        container(ax, ix - 0.04, drm_bot_y, iw + 0.08,
                  drm_top_y - drm_bot_y,
                  "#e8eaf605", INDIGO, lw=0.8, ls=":", zorder=1)
        ax.text(ix + iw / 2, drm_top_y - 0.02, "DRM",
                ha="center", va="bottom", fontsize=6, fontweight="bold",
                color=INDIGO)


# ============================================================================
# Bloco 3: Footer (loss + annotations)
# ============================================================================

def draw_footer(ax, x, y_top, w):
    """Desenha CompositeLoss + branches + resources.

    y_top e o topo do footer area.
    """
    # CompositeLoss container (taller to fit config line inside)
    ly = y_top
    lh = 0.65
    container(ax, x, ly - lh, w, lh, "#fff3e0", ORANGE, lw=1.3)
    ax.text(x + w / 2, ly - 0.08, "CompositeLoss (14 + STP)",
            ha="center", fontsize=7, fontweight="bold", color=ORANGE)

    losses = [
        "CE", "VARO", "VI", "MAD", "metric", "eidos",
        "conflict", "consc", "ground", "plast", "front",
        "mopsi", "contr", "STP",
    ]
    n = len(losses)
    lw_each = (w - 0.4) / n
    for i, name in enumerate(losses):
        lx = x + 0.2 + i * (lw_each + 0.01)
        rect(ax, lx, ly - lh + 0.2, lw_each - 0.02, 0.18, name,
             "#ffe0b2", ORANGE, BLACK, 4.5)

    # Config line (inside the loss box)
    ax.text(x + w / 2, ly - lh + 0.08,
            "warmup=0.30 | ramp=0.80 | decay_k=0.0001 | grad_clip=0.5",
            ha="center", fontsize=5, color=GRAY_MED)

    # Branches
    ay = ly - lh - 0.3
    ax.text(x, ay, "Branches:", ha="left", fontsize=6.5, fontweight="bold")
    for i, (nm, c, d) in enumerate([
        ("main", GRAY_MED, "G=diag(tau) -- ECE 0.0176"),
        ("full_mahalanobis", "#4a90d9", "G constant 5x5 SPD"),
        ("real_geodesic", INDIGO, "G(x) via MetricNet"),
        ("gravitational_objective", PURPLE, "G(x, gravity_field)"),
    ]):
        by = ay - 0.18 - i * 0.16
        ax.plot(x + 0.08, by, "s", color=c, markersize=4)
        ax.text(x + 0.2, by, nm, ha="left", va="center",
                fontsize=5, color=c, fontweight="bold")
        ax.text(x + 2.2, by, d, ha="left", va="center",
                fontsize=4.5, color=GRAY_DARK)

    # Resources
    rx = x + w * 0.5
    ax.text(rx, ay, "Resources:", ha="left", fontsize=6.5, fontweight="bold")
    for i, line in enumerate([
        "Total: 354M params",
        "Epistemic: ~2.2M (1.8%)",
        "MetricNet: ~700 params",
        "fp32 (epistemic) + bf16 (backbone)",
        "5x H200 SXM: ~6h / 1B tokens",
    ]):
        ax.text(rx, ay - 0.18 - i * 0.16, line, ha="left",
                fontsize=5, color=GRAY_DARK)


# ============================================================================
# Main
# ============================================================================

def main():
    fig, ax = plt.subplots(1, 1, figsize=(13, 12))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(6.5, 11.7, "AletheionV2 (354M)", ha="center",
            fontsize=18, fontweight="bold")
    ax.text(6.5, 11.4, "Decoder-only LLM with integrated epistemic tomography",
            ha="center", fontsize=7, color=GRAY_MED)

    # Layout: top area for backbone + epistemic, bottom for footer
    top_y = 11.1   # top of both boxes
    top_h = 8.3    # height of both boxes
    bot_y = top_y - top_h  # = 3.3

    # Backbone: left column
    draw_backbone(ax, 0.2, bot_y, 2.8, top_h)

    # Arrows from backbone to epistemic
    ax.annotate("", xy=(3.5, 7.5), xytext=(3.0, 7.5),
                arrowprops=dict(arrowstyle="-|>", color="#9b59b666", lw=1.0))
    ax.text(3.05, 7.65, "hidden_states", fontsize=4.5,
            color=GRAY_MED, style="italic")

    # EpistemicHead: right column (same height)
    draw_epistemic(ax, 3.5, bot_y, 9.2, top_h)

    # Footer: below both
    draw_footer(ax, 0.2, bot_y - 0.3, 12.5)

    # Save
    fig.savefig("docs/architecture_aletheion_v2.png", dpi=360,
                bbox_inches="tight", facecolor=WHITE, edgecolor="none")
    plt.close(fig)
    print("[OK] docs/architecture_aletheion_v2.png")


if __name__ == "__main__":
    main()
