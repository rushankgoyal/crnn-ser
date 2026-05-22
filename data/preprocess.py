"""
Preprocess raw audio datasets into log-mel spectrogram .npz files.

Usage:
    python data/preprocess.py --dataset ravdess --raw_dir /path/to/RAVDESS --out_dir data/processed/ravdess
    python data/preprocess.py --dataset esd     --raw_dir /path/to/ESD     --out_dir data/processed/esd
"""

import argparse
import os
import re
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Audio → feature
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def compute_log_mel(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    win_length = int(sr * win_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)  # (n_mels, T)


# ---------------------------------------------------------------------------
# Per-bin normalizer
# ---------------------------------------------------------------------------

def fit_normalizer(specs: list) -> tuple:
    """Compute per-bin mean and std from a list of (128, T_i) arrays."""
    all_frames = np.concatenate(specs, axis=1)  # (128, total_T)
    mean = all_frames.mean(axis=1, keepdims=True).astype(np.float32)
    std = all_frames.std(axis=1, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_normalizer(spec: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (spec - mean) / std


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

# RAVDESS emotion codes → 4-way label
# Modality 01=full-AV, 02=video, 03=audio; we use audio (03)
# Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
_RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}
_LABEL_MAP = {"happy": 0, "sad": 1, "angry": 2, "neutral": 3}


def _load_ravdess(raw_dir: str) -> list:
    """Returns list of (audio_path, speaker_id, label_str)."""
    samples = []
    pattern = re.compile(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav")
    for path in sorted(Path(raw_dir).rglob("*.wav")):
        m = pattern.match(path.name)
        if m is None:
            continue
        modality = m.group(1)
        if modality != "03":  # audio-only files
            continue
        emotion_code = m.group(3)
        if emotion_code not in _RAVDESS_EMOTION_MAP:
            continue
        # Speaker ID from parent folder name, e.g. "Actor_01"
        speaker_id = path.parent.name
        samples.append((str(path), speaker_id, _RAVDESS_EMOTION_MAP[emotion_code]))
    return samples


# ESD-English folder structure: ESD/{speaker}/{emotion}/{split}/{file}.wav
# Emotions: Angry, Happy, Neutral, Sad, Surprise
_ESD_EMOTION_MAP = {
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}


def _load_esd(raw_dir: str) -> list:
    samples = []
    for path in sorted(Path(raw_dir).rglob("*.wav")):
        parts = path.parts
        # expect .../ESD/<speaker>/<emotion>/...
        try:
            emotion_idx = next(
                i for i, p in enumerate(parts)
                if p.lower() in _ESD_EMOTION_MAP
            )
            emotion_str = parts[emotion_idx].lower()
            speaker_id = parts[emotion_idx - 1]
        except StopIteration:
            continue
        samples.append((str(path), speaker_id, _ESD_EMOTION_MAP[emotion_str]))
    return samples


# ---------------------------------------------------------------------------
# Speaker-independent split
# ---------------------------------------------------------------------------

def speaker_independent_split(
    samples: list,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Split by speaker so no speaker appears in more than one partition."""
    speakers = sorted(set(s[1] for s in samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    test_spk = set(speakers[:n_test])
    val_spk = set(speakers[n_test: n_test + n_val])

    train, val, test = [], [], []
    for item in samples:
        spk = item[1]
        if spk in test_spk:
            test.append(item)
        elif spk in val_spk:
            val.append(item)
        else:
            train.append(item)
    return train, val, test


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_dataset(
    raw_dir: str,
    out_dir: str,
    dataset: str,
    sr: int = 16000,
    n_mels: int = 128,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {dataset} files in {raw_dir} ...")
    if dataset == "ravdess":
        samples = _load_ravdess(raw_dir)
    elif dataset == "esd":
        samples = _load_esd(raw_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'ravdess' or 'esd'.")

    if not samples:
        raise RuntimeError(f"No matching audio files found in {raw_dir}")

    print(f"Found {len(samples)} clips. Splitting by speaker ...")
    train_s, val_s, test_s = speaker_independent_split(
        samples, val_fraction=val_fraction, test_fraction=test_fraction
    )
    print(f"  train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")

    def compute_specs(split_samples):
        specs, labels = [], []
        for path, _, label_str in tqdm(split_samples):
            audio = load_audio(path, sr=sr)
            spec = compute_log_mel(audio, sr=sr, n_mels=n_mels, win_ms=win_ms, hop_ms=hop_ms)
            specs.append(spec)
            labels.append(_LABEL_MAP[label_str])
        return specs, np.array(labels, dtype=np.int64)

    print("Computing train spectrograms ...")
    train_specs, train_labels = compute_specs(train_s)

    print("Fitting per-bin normalizer on training set ...")
    mean, std = fit_normalizer(train_specs)
    np.save(os.path.join(out_dir, "normalizer_mean.npy"), mean)
    np.save(os.path.join(out_dir, "normalizer_std.npy"), std)

    def normalize_and_save(specs, labels, name):
        norm = [apply_normalizer(s, mean, std) for s in specs]
        X = np.empty(len(norm), dtype=object)
        for i, s in enumerate(norm):
            X[i] = s
        np.savez(os.path.join(out_dir, f"{name}.npz"), X=X, y=labels)
        print(f"  Saved {name}.npz  ({len(labels)} clips)")

    normalize_and_save(train_specs, train_labels, "train")

    print("Computing val spectrograms ...")
    val_specs, val_labels = compute_specs(val_s)
    normalize_and_save(val_specs, val_labels, "val")

    print("Computing test spectrograms ...")
    test_specs, test_labels = compute_specs(test_s)
    normalize_and_save(test_specs, test_labels, "test")

    print(f"\nDone. Files written to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio dataset to log-mel .npz files")
    parser.add_argument("--dataset", required=True, choices=["ravdess", "esd"])
    parser.add_argument("--raw_dir", required=True, help="Root directory of raw audio files")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npz files")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--win_ms", type=float, default=25.0)
    parser.add_argument("--hop_ms", type=float, default=10.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    args = parser.parse_args()

    preprocess_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        dataset=args.dataset,
        sr=args.sr,
        n_mels=args.n_mels,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
