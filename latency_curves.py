"""Latency–accuracy curves for the AnisotropicCRNN.

Per-frame class-balanced UAR vs time-elapsed, per-class recall curves, K-stable-
correct latency per class, and time-to-N%-of-plateau UAR. Designed for held-out
test sets long enough that the per-frame denominator stays constant out to
max_frames (e.g. the MSP-Podcast ≥5s subset).

Usage:
    python latency_curves.py \\
      --ckpt runs/<run>/best.pt \\
      --test data/processed/<dataset>/test.npz
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import SERDataset
from train import build_model
from utils.metrics import unweighted_avg_recall

HOP_MS = 10.0
EMOTIONS = ["happy", "sad", "angry", "neutral"]
DEFAULT_CFG = {
    "num_classes": 4,
    "n_mels": 128,
    "sample_rate": 16000,
    "win_length_ms": 25,
    "model": {
        "conv_channels": [1, 8, 16, 32, 64],
        "kernel_freq": 32,
        "lstm_hidden": 128,
        "lstm_layers": 1,
        "dropout": 0.3,
        "norm": "layer",
    },
}


def per_frame_predictions(model, ds):
    """Run inference; return list of (preds_1d, true_label, T_frames)."""
    out = []
    with torch.no_grad():
        for i in range(len(ds)):
            spec, lab = ds[i]
            logits = model(spec.unsqueeze(0))[0]  # [T, C]
            out.append((logits.argmax(-1).numpy(), int(lab), int(logits.shape[0])))
    return out


def class_balanced_uar_per_frame(pf, max_frames, num_classes=4):
    labels = np.array([p[1] for p in pf])
    correct = np.full((len(pf), max_frames), np.nan)
    for i, (preds, _, T) in enumerate(pf):
        u = min(T, max_frames)
        correct[i, :u] = (preds[:u] == labels[i]).astype(float)
    uar = np.full(max_frames, np.nan)
    for t in range(max_frames):
        recalls = []
        for c in range(num_classes):
            mask = (labels == c) & ~np.isnan(correct[:, t])
            if mask.any():
                recalls.append(correct[mask, t].mean())
        if recalls:
            uar[t] = float(np.mean(recalls))
    return uar


def per_class_recall_per_frame(pf, max_frames, num_classes=4):
    labels = np.array([p[1] for p in pf])
    rec = np.full((num_classes, max_frames), np.nan)
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        correct = np.full((len(idx), max_frames), np.nan)
        for j, i in enumerate(idx):
            preds, _, T = pf[i]
            u = min(T, max_frames)
            correct[j, :u] = (preds[:u] == c).astype(float)
        for t in range(max_frames):
            col = correct[:, t]
            m = ~np.isnan(col)
            if m.any():
                rec[c, t] = col[m].mean()
    return rec


def stable_correct_ms(pf, K, num_classes=4):
    """First frame at which prediction == true for K consecutive frames.

    Returns dict[class_idx] -> list of ms (one per clip with that label that
    reaches the K-stable condition somewhere in its run-time).
    """
    out = {c: [] for c in range(num_classes)}
    for preds, lab, T in pf:
        correct = (preds == lab).astype(int)
        run, start = 0, None
        for t in range(T):
            if correct[t]:
                if run == 0:
                    start = t
                run += 1
                if run >= K:
                    out[lab].append(start * HOP_MS)
                    break
            else:
                run, start = 0, None
    return out


def time_to_plateau_fraction(uar, plateau_window_frames, fractions):
    tail = uar[-plateau_window_frames:]
    tail = tail[~np.isnan(tail)]
    if len(tail) == 0:
        return None, {f: None for f in fractions}
    plateau = float(tail.mean())
    result = {}
    for f in fractions:
        target = f * plateau
        hits = np.where((~np.isnan(uar)) & (uar >= target))[0]
        result[f] = int(hits[0]) if len(hits) else None
    return plateau, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", default=None, help="output PNG path")
    p.add_argument("--max_frames", type=int, default=500)
    p.add_argument("--min_clip_frames", type=int, default=0,
                   help="drop clips shorter than this many frames so the per-frame "
                        "denominator stays constant across [0, max_frames)")
    p.add_argument("--plateau_window_ms", type=float, default=100.0)
    p.add_argument("--K", type=int, default=5, help="frames of consecutive-correct for stable metric")
    args = p.parse_args()

    output = args.output or f"results/latency_{os.path.basename(os.path.dirname(args.test))}.png"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"loading model from {args.ckpt} ...")
    m = build_model(DEFAULT_CFG).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    m.load_state_dict(ck["model_state"])
    print(f"  ckpt epoch={ck.get('epoch', '?')} saved val_uar={ck.get('val_uar', '?')}")

    print(f"loading test set from {args.test} ...")
    ds = SERDataset(args.test)
    print(f"  N={len(ds)} clips")

    print("running per-frame inference ...")
    pf = per_frame_predictions(m, ds)
    if args.min_clip_frames > 0:
        before = len(pf)
        pf = [p for p in pf if p[2] >= args.min_clip_frames]
        print(f"  filtered to clips >= {args.min_clip_frames} frames "
              f"({args.min_clip_frames * HOP_MS:.0f} ms): {before} -> {len(pf)}")
        if not pf:
            raise SystemExit("no clips passed the duration filter")
    final_argmax_uar = unweighted_avg_recall(
        np.array([p[1] for p in pf]),
        np.array([p[0][-1] for p in pf]),
    )
    print(f"  final-frame argmax UAR = {final_argmax_uar:.4f}")

    # Denominator constancy check
    n_at = np.array([sum(1 for p in pf if p[2] > t) for t in range(args.max_frames)])
    print(f"  denominator @ t=0: {n_at[0]}; @ t={args.max_frames - 1}: {n_at[-1]}")
    if n_at[-1] < 0.8 * n_at[0]:
        print("  WARNING: denominator dropped >20% across the window — late-tail curve will be noisy")

    uar = class_balanced_uar_per_frame(pf, args.max_frames)
    rec = per_class_recall_per_frame(pf, args.max_frames)

    sc = stable_correct_ms(pf, args.K)
    print(f"\n=== Median ms to K={args.K}-stable-correct (per class) ===")
    for c, name in enumerate(EMOTIONS):
        vals = sc[c]
        if vals:
            med = float(np.median(vals))
            print(f"  {name:8s} n={len(vals):4d}  median={med:6.0f} ms  (range {min(vals):.0f}-{max(vals):.0f})")
        else:
            print(f"  {name:8s} n=0  no clip reaches {args.K} consecutive correct")

    plateau_frames = max(1, int(args.plateau_window_ms / HOP_MS))
    fractions = [0.5, 0.75, 0.9, 0.95]
    plateau, t_to = time_to_plateau_fraction(uar, plateau_frames, fractions)
    print(f"\n=== Time to N% of plateau UAR ({plateau:.4f}, mean of last {args.plateau_window_ms:.0f} ms) ===")
    for f in fractions:
        t = t_to[f]
        print(f"  {int(f * 100):>2}%: t={t * HOP_MS:>6.0f} ms ({t} frames)" if t is not None
              else f"  {int(f * 100):>2}%: never reached")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    t_axis = np.arange(args.max_frames) * HOP_MS
    ax.plot(t_axis, uar, color="black", lw=2.5, label="overall UAR (class-balanced)")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for c, (name, col) in enumerate(zip(EMOTIONS, colors)):
        ax.plot(t_axis, rec[c], color=col, lw=1.5, alpha=0.85, label=f"recall: {name}")
    ax.axhline(0.25, ls=":", c="gray", label="chance (4 classes)")
    if plateau is not None:
        ax.axhline(plateau, ls="--", c="black", alpha=0.4, label=f"plateau ≈ {plateau:.3f}")
    t90 = t_to.get(0.9)
    if t90 is not None:
        ax.axvline(t90 * HOP_MS, ls="-.", c="red", alpha=0.6,
                   label=f"90% of plateau @ {t90 * HOP_MS:.0f} ms")
    ax.set_xlabel("audio elapsed (ms)")
    ax.set_ylabel("per-frame accuracy / class-balanced UAR")
    ax.set_title(f"Latency–accuracy curve  (N={len(ds)}, ckpt @ epoch {ck.get('epoch', '?')})")
    ax.set_xlim(0, args.max_frames * HOP_MS)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
