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


@app.function(image=image, volumes={DATA: vol}, cpu=4.0, memory=8192, timeout=2 * 60 * 60)
def cache_msppodcast():
    """One-time: download the full MSP-Podcast parquet shards into the Volume.

    HF rate-limits anonymous streaming hard (multi-minute backoffs on parquet
    pulls). Pre-caching once via `load_dataset` (which uses huggingface_hub with
    built-in retries) lets every subsequent loader read locally with zero network.
    """
    import os
    from datasets import load_dataset

    cache_dir = f"{DATA}/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset("AbstractTTS/PODCAST", split="train", cache_dir=cache_dir)
    print(f"Cached MSP-Podcast: {len(ds)} rows at {cache_dir}", flush=True)
    vol.commit()


@app.function(image=image, cpu=4.0, memory=8192, timeout=30 * 60)
def validate_msppodcast():
    """Metadata-only check (no audio): MSP-Podcast 4-class distribution + episode parse.

    Uses pyarrow directly on the parquet files with true column projection so we
    only fetch the `file` and `major_emotion` columns (tiny) instead of streaming
    the 26.9 GB audio bytes.
    """
    import collections
    import os
    import re

    import pyarrow.dataset as pa_ds
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = "datasets/AbstractTTS/PODCAST@~parquet/default/train"
    all_paths = sorted([p for p in fs.ls(base, detail=False) if p.endswith(".parquet")])
    print(f"Found {len(all_paths)} parquet files at {base}", flush=True)

    # HF rate-limits unauthenticated parquet pulls hard (multi-minute backoffs).
    # Sample a deterministic ~7% subset (~11K rows) — enough to estimate per-class
    # proportions; we extrapolate at the end.
    import random
    rng = random.Random(0)
    sample_n = min(5, len(all_paths))
    paths = rng.sample(all_paths, sample_n)
    print(f"Sampling {sample_n}/{len(all_paths)} files: {[p.rsplit('/',1)[-1] for p in paths]}", flush=True)

    import time
    ds_pa = pa_ds.dataset(paths, format="parquet", filesystem=fs)
    files, emos = [], []
    for i, frag in enumerate(ds_pa.get_fragments()):
        t = frag.to_table(columns=["file", "major_emotion"])
        files.extend(t.column("file").to_pylist())
        emos.extend(t.column("major_emotion").to_pylist())
        print(f"  read {i + 1}/{sample_n} files, {len(files)} rows so far", flush=True)
        time.sleep(0.5)
    print(f"Loaded {len(files)} rows from sample (metadata only)", flush=True)

    scale = len(all_paths) / sample_n  # extrapolation factor (≈13.6)

    LABELS = {"happy", "sad", "angry", "neutral"}
    emo = collections.Counter()
    per_class = collections.Counter()
    episodes = collections.Counter()
    eps_per_class = {k: collections.Counter() for k in LABELS}
    bad = 0
    for f, e in zip(files, emos):
        e = (e or "").lower()
        emo[e] += 1
        if e in LABELS:
            per_class[e] += 1
            m = re.match(r"^MSP-PODCAST_(\d+)_", os.path.basename(f or ""))
            if m:
                episodes[m.group(1)] += 1
                eps_per_class[e][m.group(1)] += 1
            else:
                bad += 1

    print(f"total rows: {len(files)}", flush=True)
    print(f"ALL major_emotion: {dict(emo)}", flush=True)
    print(f"kept 4-class: {sum(per_class.values())}  {dict(per_class)}", flush=True)
    print(f"distinct episodes carrying any kept clip: {len(episodes)}", flush=True)
    print(f"distinct episodes per class: {{ {', '.join(f'{k}:{len(v)}' for k,v in eps_per_class.items())} }}", flush=True)
    print(f"unparsed filenames: {bad}", flush=True)
    rarest = min(per_class.values()) if per_class else 0
    print(f"=== SAMPLE counts (above) — extrapolation factor x{scale:.2f} ===", flush=True)
    print(f"estimated full-dataset 4-class total: {int(sum(per_class.values()) * scale):,}", flush=True)
    print(f"estimated full per-class: {{ {', '.join(f'{k}:{int(v*scale):,}' for k,v in per_class.items())} }}", flush=True)
    est_rarest = int(rarest * scale)
    print(f">>> ESTIMATED balanced subset cap (min per class, full ds) = {est_rarest:,}  -> total {4*est_rarest:,}", flush=True)


@app.function(image=image, timeout=20 * 60)
def validate_iemocap():
    """Metadata-only check (no audio download): IEMOCAP label distribution + speaker parse."""
    import collections
    import os
    import re

    from datasets import load_dataset

    ds = load_dataset("AbstractTTS/IEMOCAP", split="train", streaming=True).select_columns(
        ["file", "major_emotion"]
    )
    LABELS = {"happy", "sad", "angry", "neutral"}
    emo, per_class, spk, bad, n = (
        collections.Counter(), collections.Counter(), collections.Counter(), 0, 0,
    )
    for row in ds:
        n += 1
        e = (row.get("major_emotion") or "").lower()
        emo[e] += 1
        if e in LABELS:
            per_class[e] += 1
            m = re.match(r"(Ses\d+)[FM]_.*_([FM])\d+\.wav$", os.path.basename(row.get("file", "")))
            if m:
                spk[m.group(1) + "_" + m.group(2)] += 1
            else:
                bad += 1
    print(f"total rows: {n}", flush=True)
    print(f"ALL major_emotion: {dict(emo)}", flush=True)
    print(f"kept 4-class: {sum(per_class.values())}  {dict(per_class)}", flush=True)
    print(f"speakers ({len(spk)}): {dict(sorted(spk.items()))}", flush=True)
    print(f"unparsed speakers: {bad}", flush=True)


@app.function(image=image, volumes={DATA: vol}, cpu=4.0, memory=32768, timeout=4 * 60 * 60)
def preprocess_combined_trim(
    out_subdir: str,
    datasets: list,
    trim_top_db: float = 30.0,
    train_only_datasets: list = None,
    naive_split: bool = False,
):
    import os

    from data.preprocess import preprocess_combined

    # Point HF datasets cache at the Volume so the MSP-Podcast loader finds the
    # pre-cached parquet shards (cache_msppodcast must have been run once first).
    os.environ["HF_DATASETS_CACHE"] = f"{DATA}/hf_cache"

    out = f"{DATA}/{out_subdir}"
    preprocess_combined(
        out_dir=out,
        datasets=datasets,
        source="hf",
        trim_top_db=trim_top_db,
        train_only_datasets=train_only_datasets,
        naive_split=naive_split,
    )
    vol.commit()


@app.function(image=image, volumes={DATA: vol}, cpu=4.0, memory=32768, timeout=4 * 60 * 60)
def preprocess_iemocap_vad(out_subdir: str = "processed_iemocap_vad", trim_top_db: float = 30.0):
    """Preprocess IEMOCAP alone with continuous V/A/D targets included."""
    from data.preprocess import preprocess_iemocap_vad as _do
    _do(out_dir=f"{DATA}/{out_subdir}", trim_top_db=trim_top_db)
    vol.commit()


@app.function(image=image, volumes={DATA: vol}, cpu=4.0, memory=16384, timeout=2 * 60 * 60)
def preprocess_external_frozen(
    out_subdir: str,
    dataset: str,
    normalizer_subdir: str,
    trim_top_db: float = 30.0,
    min_duration_s: float = None,
):
    """Preprocess an external corpus (e.g. CREMA-D) using a FROZEN training normalizer.

    `min_duration_s` is forwarded via env var to the MSP-Podcast loader (harmless
    for other datasets); set it to filter to a long-clip subset for latency-curve
    evaluation where the denominator must stay constant across the time axis.
    """
    import os

    from data.preprocess import preprocess_external_frozen as _do

    # MSP-Podcast loader reads parquet from this cache; harmless for other datasets.
    os.environ["HF_DATASETS_CACHE"] = f"{DATA}/hf_cache"
    if min_duration_s is not None:
        os.environ["MSPPODCAST_MIN_DURATION_S"] = str(min_duration_s)

    _do(
        out_dir=f"{DATA}/{out_subdir}",
        dataset=dataset,
        normalizer_dir=f"{DATA}/{normalizer_subdir}",
        source="hf",
        trim_top_db=trim_top_db,
    )
    vol.commit()


@app.function(image=image, gpu="A10G", volumes={DATA: vol}, memory=16384, timeout=4 * 60 * 60)
def train_combined(
    epochs: int = 50,
    norm: str = "layer",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    data_dirs: list = None,
    tag: str = None,
    augment: bool = False,
    batch_size: int = 32,
    use_harmonic_block: bool = False,
    label_smoothing: float = 0.0,
    class_weighted: bool = False,
    use_vad_head: bool = False,
    vad_aux_weight: float = 0.0,
    loss_normalize: bool = False,
):
    import os

    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    from data.dataset import SERDataset, collate_pad
    from train import build_model, run_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if data_dirs is None:
        data_dirs = [f"{DATA}/processed/ravdess", f"{DATA}/processed/esd"]
    if tag is None:
        tag = f"combined_{norm}"

    # SpecAugment on train only; val is never augmented.
    train_set = ConcatDataset([SERDataset(f"{d}/train.npz", augment=augment) for d in data_dirs])
    val_set = ConcatDataset([SERDataset(f"{d}/val.npz") for d in data_dirs])
    print(f"Data dirs={data_dirs}  train={len(train_set)}  val={len(val_set)}  augment={augment}", flush=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    cfg = {
        "num_classes": 4,
        "model": {
            "conv_channels": [1, 8, 16, 32, 64],
            "kernel_freq": 32,
            "lstm_hidden": 128,
            "lstm_layers": 1,
            "dropout": 0.3,
            "norm": norm,
            "use_harmonic_block": use_harmonic_block,
            "use_vad_head": use_vad_head,
            # dilation_mode/dilations/kernel_h/harmonic_out_ch fall back to model defaults
            # (octave [1,2,4,8], kernel_h=3, out_ch=8) when the block is enabled.
        },
    }
    model = build_model(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # Inverse-frequency class weights, computed from the concatenated train labels.
    class_weights_t = None
    if class_weighted:
        import numpy as _np
        all_train_labels = _np.concatenate([sub.labels for sub in train_set.datasets])
        counts = _np.bincount(all_train_labels, minlength=4).astype(_np.float32)
        weights = len(all_train_labels) / (4.0 * _np.maximum(counts, 1.0))
        class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Class counts: {counts.tolist()} -> weights: {[round(w, 3) for w in weights.tolist()]}", flush=True)
    if label_smoothing > 0:
        print(f"Label smoothing: epsilon={label_smoothing}", flush=True)

    # Russell circumplex per-class VAD lookup (rough values from Russell 1980 and
    # later refinements). Indices match _LABEL_MAP: 0=happy, 1=sad, 2=angry, 3=neutral.
    vad_lookup_t = None
    if use_vad_head and vad_aux_weight > 0:
        vad_lookup_t = torch.tensor(
            [
                [+0.80, +0.50, +0.40],  # happy:   high valence, moderate arousal, moderate dominance
                [-0.60, -0.30, -0.30],  # sad:     low valence,  low arousal,      low dominance
                [-0.50, +0.60, +0.30],  # angry:   low valence,  high arousal,     moderate dominance
                [+0.00, +0.00, +0.00],  # neutral: origin
            ],
            dtype=torch.float32,
            device=device,
        )
        print(f"VAD aux head: weight={vad_aux_weight}, "
              f"per-class Russell lookup = {vad_lookup_t.tolist()}", flush=True)

    # Loss normalization state — single dict mutated across epochs/batches by
    # run_epoch (training updates EMAs; val uses but does not update them).
    loss_normalizer = None
    if loss_normalize and use_vad_head and vad_aux_weight > 0:
        loss_normalizer = {"ce_ema": None, "vad_ema": None, "alpha": 0.99}
        print("Loss normalization: ON (EMA-based, each task contributes ~1.0)", flush=True)

    run_dir = f"{DATA}/runs/{tag}"
    os.makedirs(run_dir, exist_ok=True)

    best_uar = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_uar = run_epoch(model, train_loader, optimizer, device, train=True,
                                    class_weights=class_weights_t, label_smoothing=label_smoothing,
                                    vad_lookup=vad_lookup_t, vad_aux_weight=vad_aux_weight,
                                    loss_normalizer=loss_normalizer)
        va_loss, va_uar = run_epoch(model, val_loader, optimizer, device, train=False,
                                    class_weights=class_weights_t, label_smoothing=label_smoothing,
                                    vad_lookup=vad_lookup_t, vad_aux_weight=vad_aux_weight,
                                    loss_normalizer=loss_normalizer)
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
def preprocess_msppodcast_long_test(
    min_duration_s: float = 5.0,
    normalizer_subdir: str = "processed_combined_trim_ravdess_esd_iemocap",
    trim_top_db: float = 30.0,
):
    """Preprocess a long-clip held-out subset of MSP-Podcast (balanced 4-class,
    clips >= `min_duration_s`) under the given training corpus's frozen normalizer.

    Used to build a latency-accuracy test set where the per-frame denominator
    stays constant out to min_duration_s * 100 frames (no late-tail variance).
    Default normalizer = 3-corpus (RAVDESS+ESD+IEMOCAP).
    """
    s_tag = f"{int(min_duration_s)}s"  # "5s"
    out_subdir = f"msppodcast_long{s_tag}_frozen_3c"
    preprocess_external_frozen.remote(
        out_subdir=out_subdir,
        dataset="msppodcast",
        normalizer_subdir=normalizer_subdir,
        trim_top_db=trim_top_db,
        min_duration_s=min_duration_s,
    )


@app.local_entrypoint()
def iemocap_vad_run(
    epochs: int = 50,
    vad_aux_weight: float = 0.1,
    preprocess_data: bool = True,
    trim_top_db: float = 30.0,
    augment: bool = True,
    batch_size: int = 32,
    loss_normalize: bool = False,
    use_vad_head: bool = True,
):
    """IEMOCAP-only training (with or without the VAD aux head).

    Default: VAD aux head ON, using real continuous V/A/D labels
    (EmoVal/EmoAct/EmoDom from the IEMOCAP HF mirror, rescaled to [-1, +1])
    as per-clip aux targets. Pass `--no-use-vad-head` for the no-aux control
    (same data, same hyperparams, no aux head) to isolate the aux contribution.
    """
    out_subdir = "processed_iemocap_vad"
    if preprocess_data:
        preprocess_iemocap_vad.remote(out_subdir=out_subdir, trim_top_db=trim_top_db)
    if use_vad_head and vad_aux_weight > 0:
        vad_suffix = (
            f"_vad{int(vad_aux_weight * 100):02d}"
            + ("norm" if loss_normalize else "")
            + "_real"
        )
    else:
        vad_suffix = "_novad"  # control: no aux head
    tag = (
        f"iemocap_layer_b{batch_size}"
        + ("_aug" if augment else "")
        + vad_suffix
    )
    train_combined.remote(
        epochs=epochs,
        norm="layer",
        data_dirs=[f"{DATA}/{out_subdir}"],
        tag=tag,
        augment=augment,
        batch_size=batch_size,
        use_vad_head=use_vad_head,
        vad_aux_weight=vad_aux_weight if use_vad_head else 0.0,
        loss_normalize=loss_normalize if use_vad_head else False,
    )


@app.local_entrypoint()
def preprocess_held_out_for_2c(
    normalizer_subdir: str = "processed_combined_trim_esd_iemocap",
    trim_top_db: float = 30.0,
):
    """Preprocess CREMA-D + TESS + SAVEE + RAVDESS under the ESD+IEMOCAP frozen
    normalizer, for cross-corpus eval of the new (no-RAVDESS-in-train) checkpoint.
    Note RAVDESS is now a true held-out test set under this normalizer.
    Fans out 4 Modal containers in parallel.
    """
    handles = []
    for dataset in ["cremad", "tess", "savee", "ravdess"]:
        handles.append(preprocess_external_frozen.spawn(
            out_subdir=f"{dataset}_frozen_esdiemocap",
            dataset=dataset,
            normalizer_subdir=normalizer_subdir,
            trim_top_db=trim_top_db,
        ))
    for h in handles:
        h.get()
    print("All 4 preprocesses committed to Volume.")


@app.local_entrypoint()
def preprocess_dusha_test(
    normalizer_subdir: str = "processed_combined_trim_ravdess_esd_iemocap",
    trim_top_db: float = 30.0,
):
    """Preprocess the held-out test split of xbgoose/dusha (Russian SER, 14K clips)
    under a frozen training normalizer for cross-lingual cross-corpus eval.

    Step 0 of the Dusha investigation: does our English-trained 3-corpus model
    have any signal at all on Russian speech? Default normalizer = 3-corpus
    (RAVDESS+ESD+IEMOCAP). Output land in DATA/dusha_test_frozen_3c/test.npz.
    """
    out_subdir = "dusha_test_frozen_3c"
    preprocess_external_frozen.remote(
        out_subdir=out_subdir,
        dataset="dusha",
        normalizer_subdir=normalizer_subdir,
        trim_top_db=trim_top_db,
    )


@app.local_entrypoint()
def eval_small_xcorpus(trim_top_db: float = 30.0):
    """Preprocess TESS + SAVEE under each of the 3 training normalizers (2c/3c/4c)
    in parallel — 6 small Modal containers — for held-out cross-corpus evaluation.
    """
    norm_dirs = {
        "2c": "processed_combined_trim",
        "3c": "processed_combined_trim_ravdess_esd_iemocap",
        "4c": "processed_combined_trim_ravdess_esd_iemocap_msppodcast",
    }
    handles = []
    for dataset in ["tess", "savee"]:
        for tag, norm_dir in norm_dirs.items():
            handles.append(preprocess_external_frozen.spawn(
                out_subdir=f"{dataset}_frozen_{tag}",
                dataset=dataset,
                normalizer_subdir=norm_dir,
                trim_top_db=trim_top_db,
            ))
    for h in handles:
        h.get()  # block until all 6 finish (any failure raises)
    print("All 6 preprocesses committed to Volume.")


@app.local_entrypoint()
def trimmed_combined(
    epochs: int = 50,
    norm: str = "layer",
    trim_top_db: float = 30.0,
    preprocess_data: bool = True,
    augment: bool = False,
    batch_size: int = 32,
    datasets: str = "ravdess,esd",
    use_harmonic_block: bool = False,
    train_only: str = "",
    label_smoothing: float = 0.0,
    class_weighted: bool = False,
    naive_split: bool = False,
    use_vad_head: bool = False,
    vad_aux_weight: float = 0.0,
    loss_normalize: bool = False,
):
    """Regenerate a trimmed + jointly-normalized combined set, then train on it.

    `datasets` is a comma-separated list, e.g. "ravdess,esd,iemocap,msppodcast".
    `train_only` (comma-separated) bypasses the per-corpus speaker-independent
    split for the named datasets — their entire content goes to train (used for
    msppodcast to preserve val comparability with prior runs).

    `naive_split=True` produces a methodology-comparison run: a random clip-level
    80/10/10 split (same speaker can appear in train/val/test). Used ONLY to
    quantify the SI-vs-naive inflation gap; output gets a `_naive` tag suffix
    and writes to a separate processed/ run dir so it never overwrites the SI
    pipeline.
    """
    ds_list = [d.strip() for d in datasets.split(",")]
    train_only_list = [d.strip() for d in train_only.split(",") if d.strip()]
    suffix = "" if ds_list == ["ravdess", "esd"] else "_" + "_".join(ds_list)
    if naive_split:
        suffix += "_naive"
    out_subdir = "processed_combined_trim" + suffix
    if preprocess_data:
        preprocess_combined_trim.remote(
            out_subdir=out_subdir,
            datasets=ds_list,
            trim_top_db=trim_top_db,
            train_only_datasets=train_only_list,
            naive_split=naive_split,
        )
    loss_tag = ""
    if label_smoothing > 0:
        loss_tag += f"_ls{int(label_smoothing * 100):02d}"
    if class_weighted:
        loss_tag += "_cw"
    if use_vad_head and vad_aux_weight > 0:
        loss_tag += f"_vad{int(vad_aux_weight * 100):02d}"
        if loss_normalize:
            loss_tag += "norm"
    tag = (
        f"combined_trim_{norm}_b{batch_size}"
        + ("_aug" if augment else "")
        + suffix
        + ("_hb" if use_harmonic_block else "")
        + loss_tag
    )
    train_combined.remote(
        epochs=epochs,
        norm=norm,
        data_dirs=[f"{DATA}/{out_subdir}"],
        tag=tag,
        augment=augment,
        batch_size=batch_size,
        use_harmonic_block=use_harmonic_block,
        label_smoothing=label_smoothing,
        class_weighted=class_weighted,
        use_vad_head=use_vad_head,
        vad_aux_weight=vad_aux_weight,
        loss_normalize=loss_normalize,
    )
