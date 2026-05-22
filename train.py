"""
Train AnisotropicCRNN on RAVDESS or ESD-English.

Usage:
    python train.py --config configs/crnn_ravdess.yaml
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SERDataset
from models.crnn import AnisotropicCRNN
from utils.metrics import unweighted_avg_recall


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> AnisotropicCRNN:
    m = cfg["model"]
    return AnisotropicCRNN(
        num_classes=cfg["num_classes"],
        conv_channels=m["conv_channels"],
        kernel_freq=m["kernel_freq"],
        lstm_hidden=m["lstm_hidden"],
        lstm_layers=m["lstm_layers"],
        dropout=m["dropout"],
    )


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for spec, label in tqdm(loader, leave=False):
        spec = spec.to(device)    # [1, 1, 128, T]
        label = label.to(device)  # [1]

        logits = model(spec).squeeze(0)   # [T, C]
        T = logits.shape[0]
        targets = label.expand(T)         # [T]

        loss = F.cross_entropy(logits, targets)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = logits[-1].argmax().item()  # final-frame prediction for UAR
        all_preds.append(pred)
        all_labels.append(label.item())

    avg_loss = total_loss / len(loader)
    uar = unweighted_avg_recall(np.array(all_labels), np.array(all_preds))
    return avg_loss, uar


def train(cfg_path: str):
    cfg = load_config(cfg_path)
    dataset_name = cfg["dataset"]
    data_root = cfg["data_root"]
    t_cfg = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_set = SERDataset(os.path.join(data_root, "train.npz"))
    val_set = SERDataset(os.path.join(data_root, "val.npz"))

    # batch_size=1 — clips have variable T, no padding needed
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    run_dir = os.path.join("runs", dataset_name)
    os.makedirs(run_dir, exist_ok=True)

    best_uar = 0.0
    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss, train_uar = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_uar = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_uar)

        print(
            f"Epoch {epoch:3d}/{t_cfg['epochs']}  "
            f"train_loss={train_loss:.4f}  train_uar={train_uar:.4f}  "
            f"val_loss={val_loss:.4f}  val_uar={val_uar:.4f}"
        )

        if val_uar > best_uar:
            best_uar = val_uar
            ckpt_path = os.path.join(run_dir, "best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_uar": val_uar}, ckpt_path)
            print(f"  → Saved best checkpoint (val_uar={val_uar:.4f})")

    print(f"\nTraining complete. Best val UAR: {best_uar:.4f}")
    print(f"Checkpoint: runs/{dataset_name}/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
