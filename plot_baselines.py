"""
Generate baseline-vs-proposed comparison plots from results/{run_name}/metrics.json.

Run AFTER training + evaluate.py for each run you want plotted. Missing runs
are skipped with a warning, so this is safe to run partway through a sweep.

Usage:
    python plot_baselines.py
    python plot_baselines.py --results_dir results --out_dir results/plots
    python plot_baselines.py --hop_ms 10 --uar_floor 0.25  # 4-class chance = 0.25

Produces (under --out_dir):
    baselines_latency_accuracy.png   latency curves, baselines vs proposed
    baselines_uar_bar.png            final UAR bar chart with chance line
    baselines_confusion_grid.png     row-normalized confusion matrices
    baselines_first_correct.png      per-emotion first-correct-frame distribution
    baselines_summary_table.txt      plain-text summary

The runs are grouped:
  BASELINES    = Sharan, square 3×3, BiLSTM, last-frame loss
  PROPOSED     = baseline (anisotropic), +Harmonic, +FreqPos, +Both
Each group gets its own colour family so the two are visually separable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Run registry — single source of truth for what to plot and how.
# ---------------------------------------------------------------------------
@dataclass
class RunSpec:
    label: str
    run_name: str
    group: str        # "baseline" or "proposed"
    color: str
    linestyle: str


RUNS: list[RunSpec] = [
    # Baselines (warm palette)
    RunSpec("Sharan CNN-BiLSTM",  "baseline_sharan",            "baseline", "#7f1d1d", "-"),
    RunSpec("Square 3×3 kernel",   "baseline_square_kernel",     "baseline", "#c2410c", "--"),
    RunSpec("BiLSTM (causal off)", "baseline_bilstm",            "baseline", "#a16207", "-."),
    RunSpec("Last-frame loss",     "baseline_last_frame_loss",   "baseline", "#9d174d", ":"),
    # Proposed family (cool palette)
    RunSpec("Proposed (baseline)", "ravdess_baseline",            "proposed", "#1e3a8a", "-"),
    RunSpec("Proposed +Harmonic",  "ravdess_harmonic",            "proposed", "#0e7490", "--"),
    RunSpec("Proposed +FreqPos",   "ravdess_freqpos",             "proposed", "#0f766e", "-."),
    RunSpec("Proposed +Both",      "ravdess_harmonic_freqpos",    "proposed", "#15803d", ":"),
]

EMOTIONS = ["happy", "sad", "angry", "neutral"]


def load_runs(results_dir: str) -> dict[str, dict]:
    """Return {label: metrics} for whichever runs have a metrics.json on disk."""
    loaded = {}
    for r in RUNS:
        p = os.path.join(results_dir, r.run_name, "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                loaded[r.label] = {**json.load(f), "_spec": r}
        else:
            print(f"  [skip] {r.label}: no metrics.json at {p}")
    if not loaded:
        sys.exit(f"\nNo metrics.json found under {results_dir}/. "
                 f"Run train.py + evaluate.py first.")
    return loaded


# ---------------------------------------------------------------------------
# Plot 1 — latency vs accuracy
# ---------------------------------------------------------------------------
def plot_latency_accuracy(loaded: dict, out_path: str, hop_ms_default: float, uar_floor: float):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, m in loaded.items():
        r: RunSpec = m["_spec"]
        curve = np.array(m["latency_curve"])
        run_hop = m.get("effective_hop_ms", hop_ms_default)
        t_ms = np.arange(len(curve)) * run_hop
        lw = 2.4 if r.group == "proposed" else 1.8
        alpha = 1.0 if r.group == "proposed" else 0.85
        ax.plot(t_ms, curve, color=r.color, linestyle=r.linestyle,
                linewidth=lw, alpha=alpha,
                label=f"{label}  (UAR={m['uar']:.3f})")

    ax.axhline(uar_floor, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(5, uar_floor + 0.01, f"chance ({uar_floor:.0%})",
            color="grey", fontsize=9)
    ax.axvline(500, color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(510, 0.03, "500 ms", color="grey", fontsize=9)
    # Reference line for human <100ms recognition claim from slide 4
    ax.axvline(100, color="#9333ea", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(105, 0.95, "human ~100 ms", color="#9333ea", fontsize=9)

    ax.set_xlabel("Elapsed time (ms)", fontsize=12)
    ax.set_ylabel("Frame accuracy", fontsize=12)
    ax.set_title("Latency–Accuracy: baselines vs proposed", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — UAR bar chart (grouped, baselines vs proposed)
# ---------------------------------------------------------------------------
def plot_uar_bar(loaded: dict, out_path: str, uar_floor: float):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    names, uars, colors, groups = [], [], [], []
    for label, m in loaded.items():
        r: RunSpec = m["_spec"]
        names.append(label)
        uars.append(m["uar"])
        colors.append(r.color)
        groups.append(r.group)

    # Stable group ordering: baselines first, then proposed, then by descending UAR within
    order = sorted(range(len(names)),
                   key=lambda i: (0 if groups[i] == "baseline" else 1, -uars[i]))
    names = [names[i] for i in order]
    uars = [uars[i] for i in order]
    colors = [colors[i] for i in order]
    groups = [groups[i] for i in order]

    xs = np.arange(len(names))
    bars = ax.bar(xs, uars, color=colors, alpha=0.9,
                  edgecolor="white", linewidth=1.5, width=0.65)
    for x, u in zip(xs, uars):
        ax.text(x, u + 0.012, f"{u:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Visual divider between baseline and proposed groups
    n_baseline = sum(1 for g in groups if g == "baseline")
    if 0 < n_baseline < len(groups):
        ax.axvline(n_baseline - 0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(n_baseline / 2 - 0.5, 1.02, "Baselines",
                ha="center", fontsize=11, color="#7f1d1d", fontweight="bold")
        ax.text(n_baseline + (len(groups) - n_baseline) / 2 - 0.5, 1.02, "Proposed",
                ha="center", fontsize=11, color="#1e3a8a", fontweight="bold")

    ax.axhline(uar_floor, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(len(names) - 0.4, uar_floor + 0.008,
            f"chance ({uar_floor:.0%})", color="grey", fontsize=9, ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Final-frame UAR", fontsize=12)
    ax.set_title("Final UAR — baselines vs proposed", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — confusion matrices grid
# ---------------------------------------------------------------------------
def plot_confusion_grid(loaded: dict, out_path: str):
    n = len(loaded)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for idx, (label, m) in enumerate(loaded.items()):
        ax = axes[idx // n_cols, idx % n_cols]
        cm = np.array(m["confusion_matrix"], dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sums

        ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(EMOTIONS, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(EMOTIONS, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{label}\nUAR={m['uar']:.3f}", fontsize=10)
        for i in range(4):
            for j in range(4):
                v = cm_norm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.55 else "black", fontsize=8)

    # Blank out unused axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle("Confusion matrices (row-normalized recall)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — per-emotion first-correct-frame (latency by emotion)
# ---------------------------------------------------------------------------
def plot_first_correct(loaded: dict, out_path: str, hop_ms_default: float):
    """Bar chart per emotion across runs, showing median first-correct frame in ms.
       Highlights "which emotions does each model commit to fastest"."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = list(loaded.keys())
    n_runs = len(labels)
    x = np.arange(len(EMOTIONS))
    width = 0.8 / max(n_runs, 1)

    for i, label in enumerate(labels):
        m = loaded[label]
        r: RunSpec = m["_spec"]
        run_hop = m.get("effective_hop_ms", hop_ms_default)
        stats = m.get("first_correct_frame_stats", {})
        medians = []
        for e in EMOTIONS:
            v = stats.get(e, {}).get("median_frame")
            medians.append((v * run_hop) if v is not None else np.nan)
        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(x + offset, medians, width=width, color=r.color, alpha=0.9,
               edgecolor="white", linewidth=0.8, label=label)

    ax.set_xticks(x); ax.set_xticklabels(EMOTIONS)
    ax.set_ylabel("Median first-correct frame (ms)", fontsize=12)
    ax.set_title("Per-emotion commit latency — lower = the model commits earlier",
                 fontsize=12, fontweight="bold")
    ax.axhline(100, color="#9333ea", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(0.02, 110, "human ~100 ms", color="#9333ea", fontsize=9,
            transform=ax.get_yaxis_transform())
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plain-text summary table
# ---------------------------------------------------------------------------
def write_summary_table(loaded: dict, out_path: str):
    lines = []
    lines.append(f"{'Run':<30}  {'Group':<9}  {'UAR':>7}  {'WAcc':>7}")
    lines.append("-" * 60)
    for label, m in loaded.items():
        r: RunSpec = m["_spec"]
        wacc = m.get("weighted_accuracy", float("nan"))
        lines.append(f"{label:<30}  {r.group:<9}  {m['uar']:>7.4f}  {wacc:>7.4f}")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"  wrote {out_path}")
    print()
    print(text)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing per-run subdirs with metrics.json")
    parser.add_argument("--out_dir", default="results/plots",
                        help="Where to write generated plots")
    parser.add_argument("--hop_ms", type=float, default=10.0,
                        help="Hop length in ms (for converting frame index to time)")
    parser.add_argument("--uar_floor", type=float, default=0.25,
                        help="Chance level UAR (0.25 for balanced 4-class)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    loaded = load_runs(args.results_dir)
    print(f"Loaded {len(loaded)} runs:")
    for label in loaded:
        print(f"  - {label}")
    print()

    plot_latency_accuracy(loaded,
                          os.path.join(args.out_dir, "baselines_latency_accuracy.png"),
                          args.hop_ms, args.uar_floor)
    plot_uar_bar(loaded,
                 os.path.join(args.out_dir, "baselines_uar_bar.png"),
                 args.uar_floor)
    plot_confusion_grid(loaded,
                        os.path.join(args.out_dir, "baselines_confusion_grid.png"))
    plot_first_correct(loaded,
                       os.path.join(args.out_dir, "baselines_first_correct.png"),
                       args.hop_ms)
    write_summary_table(loaded,
                        os.path.join(args.out_dir, "baselines_summary_table.txt"))


if __name__ == "__main__":
    main()
