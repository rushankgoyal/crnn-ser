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
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from data.dataset import SERDataset, collate_pad
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
        norm=m.get("norm", "batch"),
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
        # VAD aux head
        use_vad_head=m.get("use_vad_head", False),
        # mel params for empirical dilations
        sample_rate=sr,
        n_fft=int(sr * win_ms / 1000),
        fmin=cfg.get("fmin", 0.0),
        fmax=cfg.get("fmax", None),
        f0_range=tuple(m.get("f0_range", [80, 300])),
        verbose=m.get("verbose", False),
    )


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    class_weights: torch.Tensor = None,
    label_smoothing: float = 0.0,
    vad_lookup: torch.Tensor = None,
    vad_aux_weight: float = 0.0,
    loss_normalizer: dict = None,
):
    """Run one training/validation epoch.

    Auxiliary VAD regression (optional, multi-task):
      - `vad_lookup`: tensor of shape [num_classes, 3] mapping each class index
        to a (Valence, Arousal, Dominance) target. When the model has a VAD
        head AND vad_lookup is provided AND vad_aux_weight > 0, an additional
        per-frame MSE loss is added: total_loss = CE + vad_aux_weight * MSE.
        Per-class lookup (Russell circumplex) is the cheap variant; per-clip
        continuous IEMOCAP-style VAD takes priority when present in the batch.
      - `loss_normalizer`: when provided (a mutable dict), each loss term is
        divided by its training-time running EMA before summing, so both terms
        contribute ~1.0 to the total throughout training (independent of
        absolute scale and of how each shrinks over epochs). EMAs are updated
        only on training batches. Pass the same dict instance every epoch to
        carry running stats across them. Keys: 'ce_ema', 'vad_ema', 'alpha'.
    """
    model.train(train)
    total_loss = 0.0
    total_ce = 0.0
    total_vad = 0.0
    n_batches = 0
    all_preds, all_labels = [], []
    use_vad_aux = (
        getattr(model, "vad_head", None) is not None and vad_aux_weight > 0
    )
    for batch in tqdm(loader, leave=False):
        # Loader may yield 3-tuples (spec, label, lengths) or 4-tuples
        # (spec, label, lengths, vad) — the latter when the underlying npz has
        # per-clip continuous VAD targets (IEMOCAP+VAD pipeline).
        if len(batch) == 4:
            spec, label, lengths, batch_vad = batch
            batch_vad = batch_vad.to(device)      # [B, 3]
        else:
            spec, label, lengths = batch
            batch_vad = None
        spec = spec.to(device)        # [B, 1, 128, T]
        label = label.to(device)      # [B]
        lengths = lengths.to(device)  # [B]

        logits = model(spec, lengths)        # [B, T, C]
        B, T, C = logits.shape

        # per-frame cross-entropy on real frames only (every real frame gets the utterance label)
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]
        targets = label.unsqueeze(1).expand(B, T)                                   # [B, T]
        per_frame = F.cross_entropy(
            logits.reshape(B * T, C),
            targets.reshape(B * T),
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="none",
        ).reshape(B, T)
        ce_loss = (per_frame * mask).sum() / mask.sum()
        loss = ce_loss

        vad_loss_value = 0.0
        if use_vad_aux:
            # Per-clip VAD targets from the batch (real, continuous IEMOCAP-style)
            # take priority over the per-class Russell lookup.
            if batch_vad is not None:
                vad_target = batch_vad.unsqueeze(1).expand(B, T, 3)
            elif vad_lookup is not None:
                vad_target = vad_lookup[label].unsqueeze(1).expand(B, T, 3)
            else:
                vad_target = None
            if vad_target is not None:
                vad_pred = model.last_vad_pred                              # [B, T, 3]
                vad_per_frame = (vad_pred - vad_target).pow(2).mean(dim=-1) # [B, T]
                vad_loss = (vad_per_frame * mask).sum() / mask.sum()
                vad_loss_value = vad_loss.item()

                if loss_normalizer is not None:
                    # Running-EMA normalization: divide each loss by its
                    # detached running mean so both terms contribute ~1.0 on
                    # average, regardless of absolute scale or training drift.
                    alpha = loss_normalizer.get("alpha", 0.99)
                    ce_val = ce_loss.detach().item()
                    if train:
                        if loss_normalizer.get("ce_ema") is None:
                            loss_normalizer["ce_ema"] = ce_val
                            loss_normalizer["vad_ema"] = vad_loss_value
                        else:
                            loss_normalizer["ce_ema"] = (
                                alpha * loss_normalizer["ce_ema"]
                                + (1 - alpha) * ce_val
                            )
                            loss_normalizer["vad_ema"] = (
                                alpha * loss_normalizer["vad_ema"]
                                + (1 - alpha) * vad_loss_value
                            )
                    ce_ema = loss_normalizer.get("ce_ema") or ce_val
                    vad_ema = loss_normalizer.get("vad_ema") or vad_loss_value
                    eps = 1e-6
                    # vad_aux_weight still applied multiplicatively *after*
                    # normalization to allow asymmetric weighting if desired
                    # (default weight=1.0 ⇒ truly equal contribution).
                    loss = (
                        ce_loss / (ce_ema + eps)
                        + vad_aux_weight * vad_loss / (vad_ema + eps)
                    )
                else:
                    loss = ce_loss + vad_aux_weight * vad_loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_vad += vad_loss_value
        n_batches += 1
        # final real-frame prediction per sequence for UAR
        final = logits[torch.arange(B, device=device), lengths - 1]  # [B, C]
        all_preds.extend(final.argmax(dim=1).cpu().tolist())
        all_labels.extend(label.cpu().tolist())

    avg_loss = total_loss / max(1, n_batches)
    uar = unweighted_avg_recall(np.array(all_labels), np.array(all_preds))
    return avg_loss, uar


def train(cfg_path: str):
    cfg = load_config(cfg_path)
    dataset_name = cfg["dataset"]
    data_root = cfg["data_root"]
    t_cfg = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # data_root may be a single path or a list of paths to combine across datasets
    roots = [data_root] if isinstance(data_root, str) else list(data_root)
    aug = cfg.get("augment", {})
    aug_kwargs = dict(
        augment=aug.get("enabled", False),
        n_freq_masks=aug.get("n_freq_masks", 2),
        freq_mask_param=aug.get("freq_mask_param", 15),
        n_time_masks=aug.get("n_time_masks", 2),
        time_mask_param=aug.get("time_mask_param", 25),
        time_mask_p=aug.get("time_mask_p", 0.2),
    )
    # SpecAugment on train only; val is never augmented.
    train_set = ConcatDataset([SERDataset(os.path.join(r, "train.npz"), **aug_kwargs) for r in roots])
    val_set = ConcatDataset([SERDataset(os.path.join(r, "val.npz")) for r in roots])
    print(f"Train roots: {roots}  (train={len(train_set)}, val={len(val_set)})  augment={aug_kwargs['augment']}")

    # variable-length clips are padded per batch and packed in the LSTM (pad frames masked from the loss)
    batch_size = t_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

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
    print(f"Checkpoint: {run_dir}/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
