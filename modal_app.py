"""Modal app: preprocess RAVDESS + ESD and train AnisotropicCRNN on a GPU.

Data and checkpoints live in a persistent Modal Volume, so preprocessing runs
once and later training runs reuse the cached .npz files.

Usage:
    modal run modal_app.py                          # preprocess (if needed) + train combined
    modal run modal_app.py --no-preprocess-data     # skip preprocess, just train
    modal run modal_app.py --epochs 30 --norm layer
    modal volume get cs231n-ser-data runs/combined_layer/best.pt ./best.pt   # fetch checkpoint
"""
import modal

app = modal.App("cs231n-ser")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "numpy",
        "librosa",
        "soundfile",
        "datasets<4.0.0",
        "pyyaml",
        "tqdm",
        "scikit-learn",
    )
    .add_local_python_source("train", "models", "data", "utils")
)

vol = modal.Volume.from_name("cs231n-ser-data", create_if_missing=True)
DATA = "/data"


@app.function(image=image, volumes={DATA: vol}, cpu=4.0, memory=16384, timeout=60 * 60)
def preprocess():
    import os

    from data.preprocess import preprocess_dataset

    for name in ("ravdess", "esd"):
        out = f"{DATA}/processed/{name}"
        if os.path.exists(f"{out}/train.npz"):
            print(f"[skip] {name} already preprocessed at {out}")
            continue
        preprocess_dataset(out_dir=out, dataset=name, source="hf")
    vol.commit()


@app.function(image=image, volumes={DATA: vol}, cpu=4.0, memory=16384, timeout=60 * 60)
def preprocess_combined_trim(trim_top_db: float = 30.0):
    from data.preprocess import preprocess_combined

    out = f"{DATA}/processed_combined_trim"
    preprocess_combined(out_dir=out, datasets=["ravdess", "esd"], source="hf", trim_top_db=trim_top_db)
    vol.commit()


@app.function(image=image, gpu="A10G", volumes={DATA: vol}, memory=16384, timeout=4 * 60 * 60)
def train_combined(
    epochs: int = 50,
    norm: str = "layer",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    data_dirs: list = None,
    tag: str = None,
):
    import os

    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    from data.dataset import SERDataset
    from train import build_model, run_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if data_dirs is None:
        data_dirs = [f"{DATA}/processed/ravdess", f"{DATA}/processed/esd"]
    if tag is None:
        tag = f"combined_{norm}"

    train_set = ConcatDataset([SERDataset(f"{d}/train.npz") for d in data_dirs])
    val_set = ConcatDataset([SERDataset(f"{d}/val.npz") for d in data_dirs])
    print(f"Data dirs={data_dirs}  train={len(train_set)}  val={len(val_set)}", flush=True)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    cfg = {
        "num_classes": 4,
        "model": {
            "conv_channels": [1, 8, 16, 32, 64],
            "kernel_freq": 32,
            "lstm_hidden": 128,
            "lstm_layers": 1,
            "dropout": 0.3,
            "norm": norm,
        },
    }
    model = build_model(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    run_dir = f"{DATA}/runs/{tag}"
    os.makedirs(run_dir, exist_ok=True)

    best_uar = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_uar = run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss, va_uar = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(va_uar)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={tr_loss:.4f} train_uar={tr_uar:.4f}  "
            f"val_loss={va_loss:.4f} val_uar={va_uar:.4f}",
            flush=True,
        )
        if va_uar > best_uar:
            best_uar = va_uar
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_uar": va_uar},
                f"{run_dir}/best.pt",
            )
            vol.commit()
            print(f"  -> saved best (val_uar={va_uar:.4f})", flush=True)

    print(f"\nDone. Best val UAR: {best_uar:.4f}  ckpt: {run_dir}/best.pt", flush=True)


@app.local_entrypoint()
def main(preprocess_data: bool = True, epochs: int = 50, norm: str = "layer"):
    if preprocess_data:
        preprocess.remote()
    train_combined.remote(epochs=epochs, norm=norm)


@app.local_entrypoint()
def trimmed_combined(epochs: int = 50, norm: str = "layer", trim_top_db: float = 30.0, preprocess_data: bool = True):
    """Regenerate a trimmed + jointly-normalized combined set, then train on it."""
    if preprocess_data:
        preprocess_combined_trim.remote(trim_top_db=trim_top_db)
    train_combined.remote(
        epochs=epochs,
        norm=norm,
        data_dirs=[f"{DATA}/processed_combined_trim"],
        tag=f"combined_trim_{norm}",
    )
