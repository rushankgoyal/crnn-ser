# CS231N SER — shared train/val for baseline comparison

This folder contains the **combined RAVDESS + ESD-English** 4-class split that the
AnisotropicCRNN headline model is trained on. Use it (or the speaker partition
below) to train the baseline so the head-to-head comparison is apples-to-apples.

## Files

| file | content |
|---|---|
| `train.npz` | training set, n=11,760 utterances |
| `val.npz`   | validation set, n=1,456 utterances |
| ~~`test.npz`~~ | **NOT in this folder.** Held out for the one-shot final eval. Do not request or evaluate against it until we're ready to lock in numbers. |

## Format

Each `.npz` has three arrays:

| key | shape | dtype | notes |
|---|---|---|---|
| `specs` | object array of `[128, T_i]` | float32 | log-mel spectrogram, variable T per clip |
| `lengths` | `[N]` | int32 | number of time frames per clip (= `T_i`) |
| `labels` | `[N]` | int64 | 0=happy, 1=sad, 2=angry, 3=neutral |

### Preprocessing already applied (do NOT redo)

- 16 kHz mono, 128 log-mel bins, 25 ms window (`n_fft=400`), 10 ms hop.
- Silence trimmed with `librosa.effects.trim(top_db=30)` before feature extraction.
- **Per-bin z-score normalization** using the joint normalizer fit on `train.npz`
  only. Stats are already baked into the arrays — do not refit or re-normalize.
- 4-class taxonomy: emotions are **dropped, not merged** (RAVDESS 8 → 4, ESD
  drops "surprise"). Final per-class counts are roughly balanced (ESD is
  balanced; RAVDESS neutral is slightly under-represented).

### Loading

```python
import numpy as np
d = np.load('train.npz', allow_pickle=True)
specs, lengths, labels = d['specs'], d['lengths'], d['labels']
# specs[i] is a [128, lengths[i]] float32 array — feed to your model directly.
```

## Split protocol (important — use the same one)

**Speaker-independent**, seed=42. Each corpus is split by speaker first, then
the per-corpus train/val/test partitions are concatenated. Within-corpus speaker
overlap is zero, so a model never sees a test/val speaker during training.

Held-out speakers (seed=42, computed deterministically):

| partition | RAVDESS speakers | ESD-English speakers |
|---|---|---|
| **train** | all actors except those listed below | all speakers except those listed below |
| **val**   | Actor_19 (male), Actor_20 (female) | 0017 |
| **test**  | Actor_16 (female), Actor_17 (male) | 0016 |

If your baseline uses log-mels and reads these `.npz` files directly, the split
is enforced for you.

If your baseline consumes raw waveforms (wav2vec2 / HuBERT / etc.) and you
need to build your own pipeline, replicate the partition from the Hugging Face
sources `AbstractTTS/RAVDESS` and `AbstractTTS/ESD_english` (lowercase) — ESD
has no speaker column, so the speaker ID is the filename prefix (e.g.
`0011_…`). Ping me for the exact split script if useful.

## Metric

Primary: **UAR** (unweighted average recall = mean per-class recall = balanced
accuracy). Also report weighted accuracy and a 4×4 confusion matrix on the
test split when we run the one-shot eval.

Utterance-level decision = `argmax` of the final real frame (for per-frame
models). For utterance-level baselines, just argmax the single logit vector.

## Don't

- Don't re-normalize the spectrograms.
- Don't merge emotions (e.g. don't fold "calm" into "neutral" — calm is dropped).
- Don't train on `val.npz`.
- Don't ask me for `test.npz` until we're both ready to commit final numbers.

Ping Rushank with questions.
