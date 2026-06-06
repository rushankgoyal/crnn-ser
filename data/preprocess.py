"""
Preprocess raw audio datasets into log-mel spectrogram .npz files.

Local source (default):
    python data/preprocess.py --dataset ravdess --raw_dir /path/to/RAVDESS --out_dir data/processed/ravdess
    python data/preprocess.py --dataset esd     --raw_dir /path/to/ESD     --out_dir data/processed/esd

HuggingFace source (no download required):
    python data/preprocess.py --source hf --dataset ravdess --out_dir data/processed/ravdess
    python data/preprocess.py --source hf --dataset esd     --out_dir data/processed/esd
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
# HuggingFace loaders — return (audio_array_16k, speaker_id, label_str)
# ---------------------------------------------------------------------------

def _normalize_audio(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    max_val = np.abs(arr).max()
    if max_val > 0:
        arr = arr / max_val
    return arr


def _load_ravdess_hf() -> list:
    from datasets import load_dataset, Audio as HFAudio
    print("Downloading AbstractTTS/RAVDESS from HuggingFace ...")
    ds = load_dataset("AbstractTTS/RAVDESS", split="train")
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))

    samples = []
    for row in ds:
        emotion = row["emotion"].lower()
        if emotion not in _LABEL_MAP:
            continue
        # Speaker encoded as last two digits of filename: 03-01-01-01-01-01-01.wav
        m = re.search(r"-(\d{2})\.wav$", row["file"])
        speaker_id = f"Actor_{m.group(1)}" if m else row["file"]
        samples.append((_normalize_audio(row["audio"]["array"]), speaker_id, emotion))
    return samples


def _load_esd_hf() -> list:
    from datasets import load_dataset, Audio as HFAudio
    print("Downloading AbstractTTS/ESD_english from HuggingFace ...")
    # streaming avoids materializing the full dataset (10K–100K rows of audio)
    ds = load_dataset("AbstractTTS/ESD_english", split="train", streaming=True)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))

    _ESD_LABEL_MAP = {"angry": "angry", "happy": "happy", "neutral": "neutral", "sad": "sad"}

    samples = []
    for row in tqdm(ds, desc="ESD_english"):
        emotion = row.get("emotion", "").lower()
        if emotion not in _ESD_LABEL_MAP:
            continue
        # ESD has no speaker column; the speaker is the filename prefix, e.g. "0011_000001.wav" -> "0011"
        m = re.match(r"(\d+)_", os.path.basename(row.get("file", "")))
        speaker_id = m.group(1) if m else "unknown"
        samples.append((_normalize_audio(row["audio"]["array"]), speaker_id, emotion))
    return samples


def _load_iemocap_hf(with_vad: bool = False) -> list:
    """Load IEMOCAP from HuggingFace.

    When `with_vad=True`, append a per-clip (V, A, D) target — the
    annotator-averaged Valence/Activation/Dominance from IEMOCAP's EmoVal,
    EmoAct, EmoDom fields (originally 1-5 scale), rescaled to [-1, +1] via
    `(x - 3) / 2`. Used by the IEMOCAP-only VAD aux-head experiment.
    """
    from datasets import load_dataset, Audio as HFAudio
    print("Downloading AbstractTTS/IEMOCAP from HuggingFace ...")
    ds = load_dataset("AbstractTTS/IEMOCAP", split="train", streaming=True)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))

    samples = []
    for row in tqdm(ds, desc="IEMOCAP"):
        # IEMOCAP has no single emotion string; use the categorical majority label.
        # 'excited' is merged into 'happy' (standard IEMOCAP 4-class convention, e.g.
        # Yu et al. 2024 "happy+excited"); all other non-canonical classes are dropped.
        emotion = (row.get("major_emotion") or "").lower()
        if emotion == "excited":
            emotion = "happy"
        if emotion not in _LABEL_MAP:
            continue
        # Speaker = session + speaking-actor gender from filename, e.g.
        # "Ses01F_impro01_F000.wav" -> "Ses01_F" (the canonical 10 IEMOCAP speakers).
        m = re.match(r"(Ses\d+)[FM]_.*_([FM])\d+\.wav$", os.path.basename(row.get("file", "")))
        speaker_id = f"{m.group(1)}_{m.group(2)}" if m else "unknown"
        audio = _normalize_audio(row["audio"]["array"])
        if with_vad:
            v = (float(row.get("EmoVal", 3.0)) - 3.0) / 2.0
            a = (float(row.get("EmoAct", 3.0)) - 3.0) / 2.0
            d = (float(row.get("EmoDom", 3.0)) - 3.0) / 2.0
            samples.append((audio, speaker_id, emotion, (v, a, d)))
        else:
            samples.append((audio, speaker_id, emotion))
    return samples


def _load_msppodcast_hf(seed: int = 0, min_duration_s: float = None) -> list:
    """Load a balanced 4-class subset of MSP-Podcast from a pre-cached local copy.

    Reads `HF_DATASETS_CACHE` (Modal Volume location set by the caller) so no
    network is involved. Keeps clips with `major_emotion` ∈ {happy, sad, angry,
    neutral}, drops everything else, and downsamples each class to
    `min(per_class_counts)`. Speaker is the episode-index proxy parsed from the
    filename (`MSP-PODCAST_<ep>_<utt>.wav`).

    Two modes:
      * `min_duration_s is None` (default): efficient path for the train-only
        balanced subset — balance is decided on metadata then only selected
        clips are decoded.
      * `min_duration_s` set (e.g., 5.0 for held-out latency-curve test sets):
        all ~74K 4-class candidates are decoded, filtered by duration, then
        balanced. Required because duration is only known after audio decode.

    `min_duration_s` may also be supplied via env var `MSPPODCAST_MIN_DURATION_S`.
    """
    import random
    from datasets import load_dataset, Audio as HFAudio

    if min_duration_s is None:
        env_val = os.environ.get("MSPPODCAST_MIN_DURATION_S")
        if env_val:
            min_duration_s = float(env_val)

    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    print(f"Loading AbstractTTS/PODCAST from cache: {cache_dir}")
    if min_duration_s is not None:
        print(f"  long-clip mode: filtering to clips >= {min_duration_s:.1f}s post-decode")
    ds = load_dataset("AbstractTTS/PODCAST", split="train", cache_dir=cache_dir)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))
    sr = 16000  # cast above forces this

    # Pass 1: collect 4-class indices grouped by class (metadata-only)
    per_class_idx = {k: [] for k in _LABEL_MAP}
    for i, e in enumerate(ds["major_emotion"]):
        el = (e or "").lower()
        if el in _LABEL_MAP:
            per_class_idx[el].append(i)
    counts_full = {k: len(v) for k, v in per_class_idx.items()}
    print(f"  4-class candidate counts: {counts_full}  total={sum(counts_full.values())}")

    rng = random.Random(seed)

    if min_duration_s is None:
        # Fast path: balance on metadata, decode selected only.
        cap = min(counts_full.values())
        print(f"  balanced cap = {cap} per class -> {4 * cap} total")
        selected = []
        for k in _LABEL_MAP:
            idxs = per_class_idx[k][:]
            rng.shuffle(idxs)
            selected.extend((i, k) for i in idxs[:cap])
        selected.sort(key=lambda x: x[0])  # sequential row access for efficient parquet reads
        samples = []
        for idx, emotion in tqdm(selected, desc="MSP-Podcast"):
            row = ds[idx]
            m = re.match(r"^MSP-PODCAST_(\d+)_", os.path.basename(row.get("file", "")))
            speaker_id = m.group(1) if m else "unknown"
            samples.append((_normalize_audio(row["audio"]["array"]), speaker_id, emotion))
        return samples

    # Slow path: decode ALL 4-class candidates, filter by duration, then balance.
    all_candidates = []
    for k in _LABEL_MAP:
        all_candidates.extend((i, k) for i in per_class_idx[k])
    all_candidates.sort(key=lambda x: x[0])  # sequential row access
    print(f"  decoding {len(all_candidates)} candidates to filter by duration ...")

    long_per_class = {k: [] for k in _LABEL_MAP}
    for idx, emotion in tqdm(all_candidates, desc="MSP-Podcast (decode+filter)"):
        row = ds[idx]
        audio = row["audio"]["array"]
        if len(audio) / sr < min_duration_s:
            continue
        m = re.match(r"^MSP-PODCAST_(\d+)_", os.path.basename(row.get("file", "")))
        speaker_id = m.group(1) if m else "unknown"
        long_per_class[emotion].append((_normalize_audio(audio), speaker_id, emotion))

    counts_long = {k: len(v) for k, v in long_per_class.items()}
    cap = min(counts_long.values()) if counts_long else 0
    print(f"  post-duration-filter counts: {counts_long}")
    print(f"  balanced cap = {cap} per class -> {4 * cap} total")

    samples = []
    for k in _LABEL_MAP:
        items = long_per_class[k][:]
        rng.shuffle(items)
        samples.extend(items[:cap])
    return samples


# Loader for small acted corpora (TESS, SAVEE) — same schema (`file`, `emotion`),
# `anger`/`happiness`/`sadness` need normalization to our canonical names.
_SAVEE_TESS_NAME_MAP = {"anger": "angry", "happiness": "happy", "sadness": "sad"}


def _load_tess_hf() -> list:
    from datasets import load_dataset, Audio as HFAudio
    print("Downloading AbstractTTS/TESS from HuggingFace ...")
    ds = load_dataset("AbstractTTS/TESS", split="train")
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))

    samples = []
    for row in tqdm(ds, desc="TESS"):
        e = (row.get("emotion") or "").lower()
        e = _SAVEE_TESS_NAME_MAP.get(e, e)
        if e not in _LABEL_MAP:
            continue
        # Speaker = leading prefix from filename, e.g. "OAF_back_angry.wav" -> "OAF" (2 speakers).
        m = re.match(r"^([A-Z]+)_", os.path.basename(row.get("file", "")))
        speaker_id = m.group(1) if m else "unknown"
        samples.append((_normalize_audio(row["audio"]["array"]), speaker_id, e))
    return samples


def _load_savee_hf() -> list:
    from datasets import load_dataset, Audio as HFAudio
    print("Downloading AbstractTTS/SAVEE from HuggingFace ...")
    ds = load_dataset("AbstractTTS/SAVEE", split="train")
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))

    samples = []
    for row in tqdm(ds, desc="SAVEE"):
        e = (row.get("emotion") or "").lower()
        e = _SAVEE_TESS_NAME_MAP.get(e, e)
        if e not in _LABEL_MAP:
            continue
        # Speaker = leading prefix, e.g. "DC_a01.wav" -> "DC" (4 male speakers: DC/JE/JK/KL).
        m = re.match(r"^([A-Z]+)_", os.path.basename(row.get("file", "")))
        speaker_id = m.group(1) if m else "unknown"
        samples.append((_normalize_audio(row["audio"]["array"]), speaker_id, e))
    return samples


def _load_cremad_hf() -> list:
    from datasets import load_dataset, Audio as HFAudio
    print("Downloading AbstractTTS/CREMA-D from HuggingFace ...")
    ds = load_dataset("AbstractTTS/CREMA-D", split="train", streaming=True)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))

    samples = []
    for row in tqdm(ds, desc="CREMA-D"):
        # CREMA-D's major_emotion uses "anger" (not "angry"); normalize to our vocabulary.
        # Drop disgust/fear (not in our 4-class); keep happy/sad/angry/neutral.
        emo_raw = (row.get("major_emotion") or "").lower()
        emotion = "angry" if emo_raw == "anger" else emo_raw
        if emotion not in _LABEL_MAP:
            continue
        # Speaker = leading actor ID from filename, e.g. "1001_DFA_ANG_XX.wav" -> "1001" (91 actors).
        m = re.match(r"^(\d+)_", os.path.basename(row.get("file", "")))
        speaker_id = m.group(1) if m else "unknown"
        samples.append((_normalize_audio(row["audio"]["array"]), speaker_id, emotion))
    return samples


def _load_dusha_hf() -> list:
    """Load the held-out test split of xbgoose/dusha (Russian SER).

    Sber/Salute "Dusha" corpus, ~350h / 300K clips originally; the HF mirror is
    the crowd (acted) subset, 164K total: train=150K, test=14K. We load only
    `test` for cross-lingual cross-corpus eval — the train split is reserved for
    a possible future "Step 1: add Russian to training" experiment.

    HF schema is bare: `audio` + `emotion` only (no speaker, no filename).
    Missing speaker IDs are fine here because `preprocess_external_frozen` does
    not perform a speaker-independent split — every clip goes to test.npz.

    Emotion vocab is {angry, neutral, positive, sad, other}: drop `other`, remap
    `positive -> happy`, keep the other three as-is to match our 4-class space.
    """
    from datasets import load_dataset, Audio as HFAudio
    print("Downloading xbgoose/dusha (test split) from HuggingFace ...")
    ds = load_dataset("xbgoose/dusha", split="test")
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))

    samples = []
    for row in tqdm(ds, desc="Dusha"):
        emo_raw = (row.get("emotion") or "").lower()
        if emo_raw == "positive":
            emotion = "happy"
        elif emo_raw in _LABEL_MAP:
            emotion = emo_raw
        else:
            continue  # drops 'other' and any unexpected label
        samples.append((_normalize_audio(row["audio"]["array"]), "unknown", emotion))
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


def naive_random_split(
    samples: list,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Random clip-level 80/10/10 split, ignoring speaker.

    Each clip is assigned to exactly one partition; the SAME SPEAKER WILL
    APPEAR IN train/val/test. Intentional — used only to quantify the
    speaker-independent vs. naive-split inflation gap for the paper's
    methodology discussion. The standard pipeline uses
    `speaker_independent_split`.
    """
    n = len(samples)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    test_idx = set(idx[:n_test].tolist())
    val_idx = set(idx[n_test: n_test + n_val].tolist())

    train, val, test = [], [], []
    for i, item in enumerate(samples):
        if i in test_idx:
            test.append(item)
        elif i in val_idx:
            val.append(item)
        else:
            train.append(item)
    return train, val, test


# ---------------------------------------------------------------------------
# Shared spec/save helpers
# ---------------------------------------------------------------------------

def _load_samples(dataset: str, source: str, raw_dir: str = None) -> list:
    if source == "hf":
        if dataset == "ravdess":
            return _load_ravdess_hf()
        if dataset == "esd":
            return _load_esd_hf()
        if dataset == "iemocap":
            return _load_iemocap_hf()
        if dataset == "cremad":
            return _load_cremad_hf()
        if dataset == "msppodcast":
            return _load_msppodcast_hf()
        if dataset == "tess":
            return _load_tess_hf()
        if dataset == "savee":
            return _load_savee_hf()
        if dataset == "dusha":
            return _load_dusha_hf()
        raise ValueError(
            f"Unknown dataset: {dataset}. Choose 'ravdess', 'esd', 'iemocap', 'cremad', "
            f"'msppodcast', 'tess', 'savee', or 'dusha'."
        )
    if not raw_dir:
        raise ValueError("raw_dir is required when source='local'")
    print(f"Scanning {dataset} files in {raw_dir} ...")
    if dataset == "ravdess":
        return _load_ravdess(raw_dir)
    if dataset == "esd":
        return _load_esd(raw_dir)
    raise ValueError(f"Unknown dataset: {dataset}. Choose 'ravdess' or 'esd'.")


def _compute_specs(split_samples, sr, n_mels, win_ms, hop_ms, trim_top_db):
    """Convert (audio, _, label_str[, vad]) tuples into specs/labels (and VAD if present).

    Accepts both 3-tuples and 4-tuples. When 4-tuples are passed, returns a
    third array of shape (N, 3) with the per-clip (V, A, D) targets.
    """
    specs, labels, vads = [], [], []
    has_vad = bool(split_samples) and len(split_samples[0]) >= 4
    for item in tqdm(split_samples):
        source_item, _, label_str = item[0], item[1], item[2]
        if isinstance(source_item, np.ndarray):
            audio = source_item
        else:
            audio = load_audio(source_item, sr=sr)
        if trim_top_db is not None:
            trimmed, _ = librosa.effects.trim(audio, top_db=trim_top_db)
            if trimmed.size > 0:  # keep original if the whole clip is below threshold
                audio = trimmed
        spec = compute_log_mel(audio, sr=sr, n_mels=n_mels, win_ms=win_ms, hop_ms=hop_ms)
        specs.append(spec)
        labels.append(_LABEL_MAP[label_str])
        if has_vad:
            vads.append(item[3])
    labels_arr = np.array(labels, dtype=np.int64)
    if has_vad:
        return specs, labels_arr, np.array(vads, dtype=np.float32)
    return specs, labels_arr


def _save_specs(specs, labels, mean, std, out_dir, name, vads: np.ndarray = None):
    norm = [apply_normalizer(s, mean, std) for s in specs]
    # Clips share the freq dim (128) but vary in length, so build the object
    # array element-by-element; np.array(..., dtype=object) tries to broadcast
    # them into a ragged 2D array and fails.
    X = np.empty(len(norm), dtype=object)
    for i, s in enumerate(norm):
        X[i] = s
    if vads is not None:
        np.savez(os.path.join(out_dir, f"{name}.npz"), X=X, y=labels, vad=vads)
        print(f"  Saved {name}.npz  ({len(labels)} clips, with VAD)")
    else:
        np.savez(os.path.join(out_dir, f"{name}.npz"), X=X, y=labels)
        print(f"  Saved {name}.npz  ({len(labels)} clips)")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_dataset(
    out_dir: str,
    dataset: str,
    source: str = "local",
    raw_dir: str = None,
    sr: int = 16000,
    n_mels: int = 128,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    trim_top_db: float = None,
):
    os.makedirs(out_dir, exist_ok=True)

    samples = _load_samples(dataset, source, raw_dir)
    if not samples:
        raise RuntimeError(f"No matching audio files found for {dataset}")

    print(f"Found {len(samples)} clips. Splitting by speaker ...")
    train_s, val_s, test_s = speaker_independent_split(
        samples, val_fraction=val_fraction, test_fraction=test_fraction
    )
    print(f"  train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")

    spec_args = (sr, n_mels, win_ms, hop_ms, trim_top_db)
    print("Computing train spectrograms ...")
    train_specs, train_labels = _compute_specs(train_s, *spec_args)

    print("Fitting per-bin normalizer on training set ...")
    mean, std = fit_normalizer(train_specs)
    np.save(os.path.join(out_dir, "normalizer_mean.npy"), mean)
    np.save(os.path.join(out_dir, "normalizer_std.npy"), std)

    _save_specs(train_specs, train_labels, mean, std, out_dir, "train")

    print("Computing val spectrograms ...")
    val_specs, val_labels = _compute_specs(val_s, *spec_args)
    _save_specs(val_specs, val_labels, mean, std, out_dir, "val")

    print("Computing test spectrograms ...")
    test_specs, test_labels = _compute_specs(test_s, *spec_args)
    _save_specs(test_specs, test_labels, mean, std, out_dir, "test")

    print(f"\nDone. Files written to {out_dir}/")


def preprocess_combined(
    out_dir: str,
    datasets: list,
    source: str = "hf",
    raw_dirs: dict = None,
    sr: int = 16000,
    n_mels: int = 128,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    trim_top_db: float = None,
    train_only_datasets: list = None,
    naive_split: bool = False,
):
    """Combine multiple datasets into one jointly-normalized set.

    Each dataset is split speaker-independently on its own (so every dataset is
    represented in train/val/test), then the splits are concatenated and a SINGLE
    per-bin normalizer is fit on the combined training spectrograms.

    Datasets named in `train_only_datasets` bypass the split and contribute their
    entire content to train (intended for very large corpora we don't want
    mixing into val/test — e.g. MSP-Podcast for comparability with prior runs).

    `naive_split=True` overrides the per-corpus speaker-independent split with
    a random clip-level 80/10/10 split on the concatenated samples (same
    speaker can appear in train/val/test). ONLY used to quantify the
    SI-vs-naive inflation gap for the paper; never the headline protocol.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_dirs = raw_dirs or {}
    train_only = set(train_only_datasets or [])

    if naive_split:
        # Concatenate all samples (including train_only-flagged datasets — they
        # just lose their train-only designation under this comparison protocol)
        # and apply a single clip-level random split. By design, the same speaker
        # may appear in train, val, and test.
        all_samples = []
        for ds in datasets:
            samples = _load_samples(ds, source, raw_dirs.get(ds))
            if not samples:
                raise RuntimeError(f"No matching audio files found for {ds}")
            print(f"  {ds}: loaded {len(samples)} clips")
            all_samples += samples
        train_s, val_s, test_s = naive_random_split(
            all_samples, val_fraction=val_fraction, test_fraction=test_fraction
        )
        print(f"NAIVE-SPLIT Combined (speaker leakage by design): "
              f"train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")
    else:
        train_s, val_s, test_s = [], [], []
        for ds in datasets:
            samples = _load_samples(ds, source, raw_dirs.get(ds))
            if not samples:
                raise RuntimeError(f"No matching audio files found for {ds}")
            if ds in train_only:
                print(f"  {ds}: TRAIN-ONLY, {len(samples)} clips all -> train")
                train_s += samples
                continue
            tr, va, te = speaker_independent_split(
                samples, val_fraction=val_fraction, test_fraction=test_fraction
            )
            print(f"  {ds}: train={len(tr)}  val={len(va)}  test={len(te)}")
            train_s += tr
            val_s += va
            test_s += te
        print(f"Combined: train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")

    spec_args = (sr, n_mels, win_ms, hop_ms, trim_top_db)
    print("Computing train spectrograms ...")
    train_specs, train_labels = _compute_specs(train_s, *spec_args)

    print("Fitting JOINT per-bin normalizer on combined training set ...")
    mean, std = fit_normalizer(train_specs)
    np.save(os.path.join(out_dir, "normalizer_mean.npy"), mean)
    np.save(os.path.join(out_dir, "normalizer_std.npy"), std)

    _save_specs(train_specs, train_labels, mean, std, out_dir, "train")

    print("Computing val spectrograms ...")
    val_specs, val_labels = _compute_specs(val_s, *spec_args)
    _save_specs(val_specs, val_labels, mean, std, out_dir, "val")

    print("Computing test spectrograms ...")
    test_specs, test_labels = _compute_specs(test_s, *spec_args)
    _save_specs(test_specs, test_labels, mean, std, out_dir, "test")

    print(f"\nDone. Files written to {out_dir}/")


def preprocess_iemocap_vad(
    out_dir: str,
    sr: int = 16000,
    n_mels: int = 128,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    trim_top_db: float = 30.0,
):
    """Preprocess IEMOCAP alone with continuous (V, A, D) labels included.

    For the VAD-aux-head experiment: IEMOCAP is the only training corpus, and
    its annotator-averaged continuous V/A/D values (rescaled to [-1, +1]) are
    stored alongside specs and class labels in the .npz files so the training
    loop can use real per-clip VAD targets instead of class-lookup approximations.
    """
    os.makedirs(out_dir, exist_ok=True)
    samples = _load_iemocap_hf(with_vad=True)
    if not samples:
        raise RuntimeError("No IEMOCAP samples loaded")

    print(f"Found {len(samples)} IEMOCAP clips. Splitting by speaker ...")
    train_s, val_s, test_s = speaker_independent_split(
        samples, val_fraction=val_fraction, test_fraction=test_fraction
    )
    print(f"  train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")

    spec_args = (sr, n_mels, win_ms, hop_ms, trim_top_db)
    print("Computing train spectrograms ...")
    train_specs, train_labels, train_vads = _compute_specs(train_s, *spec_args)

    print("Fitting per-bin normalizer on training set ...")
    mean, std = fit_normalizer(train_specs)
    np.save(os.path.join(out_dir, "normalizer_mean.npy"), mean)
    np.save(os.path.join(out_dir, "normalizer_std.npy"), std)

    _save_specs(train_specs, train_labels, mean, std, out_dir, "train", vads=train_vads)

    print("Computing val spectrograms ...")
    val_specs, val_labels, val_vads = _compute_specs(val_s, *spec_args)
    _save_specs(val_specs, val_labels, mean, std, out_dir, "val", vads=val_vads)

    print("Computing test spectrograms ...")
    test_specs, test_labels, test_vads = _compute_specs(test_s, *spec_args)
    _save_specs(test_specs, test_labels, mean, std, out_dir, "test", vads=test_vads)

    print(f"\nDone. Files written to {out_dir}/")


def preprocess_external_frozen(
    out_dir: str,
    dataset: str,
    normalizer_dir: str,
    source: str = "hf",
    raw_dir: str = None,
    sr: int = 16000,
    n_mels: int = 128,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    trim_top_db: float = None,
):
    """Preprocess an external held-out corpus using a FROZEN training normalizer.

    For cross-corpus evaluation: no train/val/test split — every clip goes into
    test.npz. The per-bin mean/std are loaded from `normalizer_dir` (a previous
    training run's output) and applied as-is; nothing is re-fit on the external
    data. This is the strict cross-corpus protocol — see the manuscript notes.
    """
    os.makedirs(out_dir, exist_ok=True)
    samples = _load_samples(dataset, source, raw_dir)
    if not samples:
        raise RuntimeError(f"No samples found for {dataset}")
    print(f"Found {len(samples)} clips. Using all as held-out test (cross-corpus eval).")

    spec_args = (sr, n_mels, win_ms, hop_ms, trim_top_db)
    print("Computing test spectrograms ...")
    specs, labels = _compute_specs(samples, *spec_args)

    mean = np.load(os.path.join(normalizer_dir, "normalizer_mean.npy"))
    std = np.load(os.path.join(normalizer_dir, "normalizer_std.npy"))
    print(f"Applied frozen normalizer from {normalizer_dir}")

    _save_specs(specs, labels, mean, std, out_dir, "test")
    print(f"\nDone. Files written to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio dataset to log-mel .npz files")
    parser.add_argument("--dataset", required=True, choices=["ravdess", "esd"])
    parser.add_argument("--source", default="local", choices=["local", "hf"],
                        help="'local' reads from --raw_dir; 'hf' downloads from HuggingFace")
    parser.add_argument("--raw_dir", default=None, help="Root directory of raw audio files (local source only)")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npz files")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--win_ms", type=float, default=25.0)
    parser.add_argument("--hop_ms", type=float, default=10.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--trim_top_db", type=float, default=None,
                        help="If set, trim leading/trailing silence below this dB threshold (e.g. 30)")
    args = parser.parse_args()

    preprocess_dataset(
        out_dir=args.out_dir,
        dataset=args.dataset,
        source=args.source,
        raw_dir=args.raw_dir,
        sr=args.sr,
        n_mels=args.n_mels,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        trim_top_db=args.trim_top_db,
    )
