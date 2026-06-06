# CS231N — Time-Efficient Speech Emotion Recognition

Unidirectional CRNN with frequency-first anisotropic kernels for sub-utterance SER.

## Architecture

A 4-layer CNN with `(32×1)` kernels (frequency-axis only, no time context) reduces the input from 128 mel bins down to 4 via valid convolution: `128→97→66→35→4`. The resulting `4×64 = 256`-dim feature vector per time step feeds a unidirectional LSTM, which outputs a 4-class prediction at every frame. Training applies cross-entropy loss at every frame, directly optimizing for early commitment.

```
Input [1, 128, T]
  → (opt) FreqPos concat          [1+P, 128, T]
  → (opt) HarmonicDilatedBlock    [harm_ch, 128, T]
  → (opt) FreqPos film            [harm_ch, 128, T]
  → 4× Conv(32×1) + BN + ReLU    [64, 4, T]
  → Flatten freq → permute        [T, 256]
  → Unidirectional LSTM(256→128)  [T, 128]
  → Linear(128→4)                 [T, 4]  ← prediction at every frame
```

### Ablation variants

| Variant | Config | Params |
|---|---|---|
| Baseline | `crnn_ravdess.yaml` | 284,780 |
| +Harmonic | `harmonic_only_ravdess.yaml` | 286,676 |
| +FreqPos | `freqpos_only_ravdess.yaml` | 285,164 |
| +Both | `harmonic_freqpos.yaml` | 286,828 |

All variants are within 1% of baseline parameter count.

## Datasets

- **RAVDESS** — 24 speakers, 1,440 clips, 8 emotions (4-way subset used)
- **ESD-English** — 10 speakers, 17,500 clips, 5 emotions (4-way subset used)

Clips are fed as full variable-length utterances; no fixed-window segmentation.

## Quickstart (local, CPU)

```bash
pip install torch torchaudio
pip install -r requirements.txt

# Run unit tests (CPU only, no data needed)
python tests/test_components.py

# Preprocess RAVDESS
python data/preprocess.py \
  --dataset ravdess \
  --raw_dir /path/to/RAVDESS \
  --out_dir data/processed/ravdess

# Train baseline
python train.py --config configs/crnn_ravdess.yaml

# Evaluate
python evaluate.py \
  --config configs/crnn_ravdess.yaml \
  --checkpoint runs/ravdess_baseline/best.pt
```

## Google Colab

Open `notebooks/run_colab.ipynb`. It handles everything:
clones the repo, installs deps, checks GPU, mounts Drive, runs CPU tests,
trains all ablation variants, evaluates, and generates comparison plots.

### Data on Colab

1. **Preprocess locally** (one-time):
   ```bash
   python data/preprocess.py --dataset ravdess --raw_dir ... --out_dir data/processed/ravdess
   ```
2. **Upload to Drive**: put `train.npz`, `val.npz`, `test.npz` into
   `MyDrive/cs231n-ser/data/ravdess/`
3. **Each Colab session**: run notebook Cell 3 (mount Drive), Cell 4 (verify files).
   The notebook reads from Drive directly — no re-uploading needed.

### Paths

- Data: configured by `data_root` in YAML config (or overridden in runtime config cell)
- Checkpoints: `runs/{run_name}/best.pt` — synced to Drive by Cell 9
- Results/plots: `results/{run_name}/` — synced to Drive by Cell 11 + 14

### Collaborating (friend runs baseline, you run the rest)

1. Both save your `results/` to Drive.
2. Before plotting: copy friend's `results/ravdess_baseline/` into your local `results/`.
3. Run Cell 12 (comparison plot) — it overlays all four variants.

## Dilation table (empirical mode)

```bash
python -m models.harmonic_block --print
```

Prints the harmonic ladder for F0=154.9 Hz (geometric mean of 80–300 Hz range)
mapped to 128 mel bins. Derived dilations: `[1, 2, 4, 8]` (matches octave defaults).

## Evaluation

The primary output is a **latency-accuracy curve**: per-frame accuracy over the test set.
Additional metrics: UAR, weighted accuracy, confusion matrix, per-emotion first-correct-frame.

## Repository Layout

```
crnn-ser/
├── configs/
│   ├── crnn_ravdess.yaml            ← baseline
│   ├── harmonic_only_ravdess.yaml   ← +harmonic only
│   ├── freqpos_only_ravdess.yaml    ← +freqpos only
│   └── harmonic_freqpos.yaml        ← +both
├── data/
│   ├── preprocess.py
│   └── dataset.py
├── models/
│   ├── crnn.py                      ← AnisotropicCRNN (wires everything)
│   ├── harmonic_block.py            ← HarmonicDilatedBlock + compute_harmonic_dilations
│   └── freq_pos.py                  ← FrequencyPositionalConditioning
├── tests/
│   └── test_components.py           ← 22 CPU unit tests
├── utils/
│   └── metrics.py
├── notebooks/
│   └── run_colab.ipynb              ← full Colab runner + comparison plots
├── train.py
└── evaluate.py
```
