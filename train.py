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
    m = cfg.get("model", {})
    sr = cfg.get("sample_rate", 16000)
    win_ms = cfg.get("win_length_ms", 25.0)
    return AnisotropicCRNN(
        num_classes=cfg["num_classes"],
        conv_channels=m.get("conv_channels", [1, 8, 16, 32, 64]),
        kernel_freq=m.get("kernel_freq", 32),
        lstm_hidden=m.get("lstm_hidden", 128),
        lstm_layers=m.get("lstm_layers", 1),
        dropout=m.get("dropout", 0.3),
        n_mels=cfg.get("n_mels", 128),
        # Component B
        use_freq_pos=m.get("use_freq_pos", False),
        freq_pos_mode=m.get("freq_pos_mode", "concat"),
        pos_dim=m.get("pos_dim", 1),
        pos_init=m.get("pos_init", "learned"),
        # Component A
        use_harmonic_block=m.get("use_harmonic_block", False),
        harmonic_out_ch=m.get("harmonic_out_ch", 8),
        dilation_mode=m.get("dilation_mode", "octave"),
        dilations=m.get("dilations", [1, 2, 4, 8]),
        kernel_h=m.get("kernel_h", 3),
        # mel params for empirical dilations
        sample_rate=sr,
        n_fft=int(sr * win_ms / 1000),
        fmin=cfg.get("fmin", 0.0),
        fmax=cfg.get("fmax", None),
        f0_range=tuple(m.get("f0_range", [80, 300])),
        verbose=m.get("verbose", False),
    )


def spec_augment(spec: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Apply SpecAugment (freq + time masking) to a single spectrogram [1, 1, F, T]."""
    F_bins = spec.shape[2]
    T_bins = spec.shape[3]
    freq_mask_max = cfg.get("freq_mask_max", 20)
    time_mask_max = cfg.get("time_mask_max", 50)
    n_freq = cfg.get("n_freq_masks", 2)
    n_time = cfg.get("n_time_masks", 2)

    out = spec.clone()
    for _ in range(n_freq):
        f = torch.randint(0, freq_mask_max + 1, (1,)).item()
        f0 = torch.randint(0, max(1, F_bins - f), (1,)).item()
        out[:, :, f0:f0 + f, :] = 0.0
    for _ in range(n_time):
        t = torch.randint(0, min(time_mask_max + 1, T_bins), (1,)).item()
        t0 = torch.randint(0, max(1, T_bins - t), (1,)).item()
        out[:, :, :, t0:t0 + t] = 0.0
    return out


def run_epoch(model, loader, optimizer, device, train: bool, augment_cfg: dict = None):
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for spec, label in tqdm(loader, leave=False):
        spec = spec.to(device)    # [1, 1, 128, T]
        label = label.to(device)  # [1]

        if train and augment_cfg:
            spec = spec_augment(spec, augment_cfg)

        logits = model(spec).squeeze(0)   # [T, C]
        T = logits.shape[0]
        targets = label.expand(T)         # [T]

        loss = F.cross_entropy(logits, targets, label_smoothing=0.1)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    augment_cfg = cfg.get("augment", None)
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

    run_dir = os.path.join("runs", cfg.get("run_name", dataset_name))
    os.makedirs(run_dir, exist_ok=True)

    best_uar = 0.0
    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss, train_uar = run_epoch(model, train_loader, optimizer, device, train=True, augment_cfg=augment_cfg)
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
