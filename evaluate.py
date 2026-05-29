"""
Evaluate a trained AnisotropicCRNN and produce latency-accuracy curves.

Usage:
    python evaluate.py --config configs/crnn_ravdess.yaml --checkpoint runs/ravdess/best.pt
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SERDataset
from models.crnn import AnisotropicCRNN
from train import build_model
from utils.metrics import (
    confusion_matrix,
    first_correct_frame,
    per_frame_accuracy,
    unweighted_avg_recall,
    weighted_accuracy,
)


def evaluate(cfg_path: str, checkpoint_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]
    data_root = cfg["data_root"]
    emotions = cfg["emotions"]  # e.g. [happy, sad, angry, neutral]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Detect temporal downsampling (e.g. Sharan pools 8x). Lets the latency curve
    # x-axis stay in real-time milliseconds regardless of model.
    hop_ms = cfg.get("hop_length_ms", 10.0)
    n_mels = cfg.get("n_mels", 128)
    with torch.no_grad():
        probe = torch.zeros(1, 1, n_mels, 64, device=device)
        probe_out = model(probe)
    temporal_ratio = 64.0 / probe_out.shape[1]
    effective_hop_ms = float(hop_ms * temporal_ratio)
    print(f"Effective hop: {effective_hop_ms:.1f} ms/frame  (model T-ratio = {temporal_ratio:.2f})")

    test_set = SERDataset(os.path.join(data_root, "test.npz"))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for spec, label in tqdm(test_loader, desc="Evaluating"):
            spec = spec.to(device)
            logits = model(spec).squeeze(0).cpu().numpy()  # [T, C]
            all_logits.append(logits)
            all_labels.append(label.item())

    # Final-frame predictions for scalar metrics
    final_preds = np.array([lg[-1].argmax() for lg in all_logits])
    labels_arr = np.array(all_labels)

    uar = unweighted_avg_recall(labels_arr, final_preds)
    wacc = weighted_accuracy(labels_arr, final_preds)
    cm = confusion_matrix(labels_arr, final_preds, labels=list(range(len(emotions))))

    print(f"\nUAR:               {uar:.4f}")
    print(f"Weighted accuracy: {wacc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{e[:4]:>5}" for e in emotions))
    for i, row in enumerate(cm):
        print(f"  {emotions[i][:7]:>7} " + "  ".join(f"{v:>5}" for v in row))

    # Per-frame accuracy curve
    curve = per_frame_accuracy(all_logits, all_labels)

    # Per-emotion first-correct-frame distribution
    first_correct = {e: [] for e in emotions}
    for logits, label in zip(all_logits, all_labels):
        fc = first_correct_frame(logits, label)
        first_correct[emotions[label]].append(fc)

    fc_stats = {}
    for e, vals in first_correct.items():
        valid = [v for v in vals if v is not None]
        fc_stats[e] = {
            "median_frame": float(np.median(valid)) if valid else None,
            "mean_frame": float(np.mean(valid)) if valid else None,
            "never_correct_pct": float(100 * (len(vals) - len(valid)) / len(vals)) if vals else None,
        }
        print(f"\n  {e}: median first-correct frame = {fc_stats[e]['median_frame']}")

    # Save results
    run_name = cfg.get("run_name", dataset_name)
    results_dir = os.path.join("results", run_name)
    os.makedirs(results_dir, exist_ok=True)

    metrics = {
        "uar": uar,
        "weighted_accuracy": wacc,
        "confusion_matrix": cm.tolist(),
        "first_correct_frame_stats": fc_stats,
        "latency_curve": curve.tolist(),   # saved for cross-variant comparison plots
        "effective_hop_ms": effective_hop_ms,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics → results/{run_name}/metrics.json")

    # Plot latency-accuracy curve
    _plot_latency_curve(curve, all_logits, all_labels, emotions, results_dir, cfg,
                        run_name, effective_hop_ms)


def _plot_latency_curve(curve, all_logits, all_labels, emotions, results_dir, cfg,
                        run_name="", effective_hop_ms: float = None):
    hop_ms = effective_hop_ms if effective_hop_ms is not None else cfg.get("hop_length_ms", 10.0)
    frames = np.arange(len(curve))
    time_ms = frames * hop_ms

    # Per-emotion curves
    emotion_curves = {}
    for i, e in enumerate(emotions):
        idxs = [j for j, lbl in enumerate(all_labels) if lbl == i]
        if not idxs:
            continue
        sub_logits = [all_logits[j] for j in idxs]
        sub_labels = [i] * len(idxs)
        emotion_curves[e] = per_frame_accuracy(sub_logits, sub_labels, max_frames=len(curve))

    plt.figure(figsize=(10, 5))
    plt.plot(time_ms, curve, linewidth=2, label="Overall", color="black")
    for e, ec in emotion_curves.items():
        plt.plot(time_ms, ec, linewidth=1.5, linestyle="--", label=e)

    plt.xlabel("Elapsed time (ms)")
    plt.ylabel("Accuracy")
    plt.title("Latency–accuracy curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(results_dir, "latency_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
