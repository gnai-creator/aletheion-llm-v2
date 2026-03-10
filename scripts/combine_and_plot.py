#!/usr/bin/env python3
"""Combine fine-tuning logs from H200 cloud + RTX 4090 local and plot."""

import json
import re
import sys
from pathlib import Path

LOG_DIR = Path("checkpoints/350m_epistemic_finetune")


def parse_h200_log(path):
    """Parse cloud_train_5xh200_fp32.log (stdout format)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("step="):
                continue
            m = re.match(
                r"step=(\d+)/\d+\s+loss=([\d.]+)\s+ce=([\d.]+)"
                r"(?:\s+stp=([\d.]+))?\s+lr=([\d.e+-]+)\s+gnorm=([\d.]+)"
                r"\s+tok/s=(\d+)(?:\s+phi=([\d.]+))?(?:\s+conf=([\d.]+))?"
                r"\s+tokens=([\d,]+)",
                line,
            )
            if m:
                entry = {
                    "step": int(m.group(1)),
                    "total": float(m.group(2)),
                    "ce": float(m.group(3)),
                    "stp": float(m.group(4)) if m.group(4) else 0,
                    "lr": float(m.group(5)),
                    "grad_norm": float(m.group(6)),
                    "tokens_per_sec": float(m.group(7)),
                    "avg_phi": float(m.group(8)) if m.group(8) else None,
                    "avg_confidence": float(m.group(9)) if m.group(9) else None,
                    "tokens_seen": int(m.group(10).replace(",", "")),
                    "source": "5xH200",
                }
                entries.append(entry)
    return entries


def parse_train_log(path):
    """Parse train.log (pipe-separated key=value format)."""
    entries = []
    with open(path) as f:
        for line in f:
            if "step=" not in line or "Config:" in line or "Training started" in line:
                continue
            parts = line.strip().split(" | ")
            entry = {"source": "RTX4090"}
            for part in parts:
                if "=" in part:
                    key, val = part.split("=", 1)
                    key = key.strip()
                    try:
                        entry[key] = float(val)
                    except ValueError:
                        entry[key] = val
            if "step" in entry:
                entry["step"] = int(entry["step"])
                entries.append(entry)
    return entries


def combine_logs():
    """Combine all logs, deduplicate by step (prefer later source)."""
    all_entries = []

    # H200 cloud log
    h200_log = LOG_DIR / "cloud_train_5xh200_fp32.log"
    if h200_log.exists():
        entries = parse_h200_log(h200_log)
        print(f"H200 log: {len(entries)} entries (steps {entries[0]['step']}-{entries[-1]['step']})" if entries else "H200 log: empty")
        all_entries.extend(entries)

    # RTX 4090 train.log
    train_log = LOG_DIR / "train.log"
    if train_log.exists():
        entries = parse_train_log(train_log)
        print(f"RTX4090 log: {len(entries)} entries (steps {entries[0]['step']}-{entries[-1]['step']})" if entries else "RTX4090 log: empty")
        all_entries.extend(entries)

    # Deduplicate: keep latest source per step
    by_step = {}
    for e in all_entries:
        s = e["step"]
        if s not in by_step:
            by_step[s] = e
        else:
            # Prefer RTX4090 (more detailed) over H200
            if e["source"] == "RTX4090":
                by_step[s] = e

    combined = sorted(by_step.values(), key=lambda x: x["step"])
    print(f"Combined: {len(combined)} entries (steps {combined[0]['step']}-{combined[-1]['step']})")
    return combined


def save_combined(entries, path):
    """Save combined log as JSON."""
    # Transpose to dict of lists
    keys = set()
    for e in entries:
        keys.update(e.keys())

    log = {}
    for key in sorted(keys):
        vals = [e.get(key, None) for e in entries]
        log[key] = vals

    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"Saved: {path}")
    return log


def plot_training(log, output_dir):
    """Plot training metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = log["step"]
    sources = log.get("source", ["unknown"] * len(steps))

    # Color by source
    colors_map = {"5xH200": "#2196F3", "RTX4090": "#FF9800"}
    colors = [colors_map.get(s, "gray") for s in sources]

    def get_vals(key):
        return [v if v is not None else float("nan") for v in log.get(key, [])]

    # 1. Loss (total + CE)
    fig, ax = plt.subplots(figsize=(14, 6))
    total = get_vals("total")
    ce = get_vals("ce")
    ax.scatter(steps, total, c=colors, s=8, alpha=0.7, label="total loss")
    ax.plot(steps, ce, color="red", alpha=0.5, linewidth=0.8, label="CE loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (Fine-tuning Epistemic fp32)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Add source annotation
    h200_steps = [s for s, src in zip(steps, sources) if src == "5xH200"]
    rtx_steps = [s for s, src in zip(steps, sources) if src == "RTX4090"]
    if h200_steps and rtx_steps:
        ax.axvline(x=rtx_steps[0], color="orange", linestyle="--", alpha=0.5, label="H200→RTX4090")
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'loss.png'}")

    # 2. Grad norm
    fig, ax = plt.subplots(figsize=(14, 4))
    gnorm = get_vals("grad_norm")
    ax.scatter(steps, gnorm, c=colors, s=8, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "grad_norm.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'grad_norm.png'}")

    # 3. Phi + Confidence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    phi = get_vals("avg_phi")
    conf = get_vals("avg_confidence")
    ax1.scatter(steps, phi, c=colors, s=8, alpha=0.7)
    ax1.set_ylabel("phi (integration)")
    ax1.set_title("Epistemic Metrics")
    ax1.grid(True, alpha=0.3)
    ax2.scatter(steps, conf, c=colors, s=8, alpha=0.7)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("confidence")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "epistemic.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'epistemic.png'}")

    # 4. Throughput
    fig, ax = plt.subplots(figsize=(14, 4))
    tps = get_vals("tokens_per_sec")
    ax.scatter(steps, tps, c=colors, s=8, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "throughput.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'throughput.png'}")

    # 5. STP loss
    fig, ax = plt.subplots(figsize=(14, 4))
    stp = get_vals("stp")
    ax.scatter(steps, stp, c=colors, s=8, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("STP Loss")
    ax.set_title("STP (Smooth Transition Penalty)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "stp.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'stp.png'}")

    # 6. Learning rate
    fig, ax = plt.subplots(figsize=(14, 4))
    lr = get_vals("lr")
    ax.scatter(steps, lr, c=colors, s=8, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "lr.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'lr.png'}")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    combined = combine_logs()
    out_json = LOG_DIR / "combined_finetune_log.json"
    log = save_combined(combined, out_json)

    plot_dir = LOG_DIR / "plots_finetune"
    plot_training(log, plot_dir)
