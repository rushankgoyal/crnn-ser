# CS231N — Time-Efficient Speech Emotion Recognition

Unidirectional CRNN with frequency-first anisotropic kernels for sub-utterance SER.

## Architecture

A 4-layer CNN with `(32×1)` kernels (frequency-axis only, no time context) reduces the input from 128 mel bins down to 4 via valid convolution: `128→97→66→35→4`. The resulting `4×64 = 256`-dim feature vector per time step feeds a unidirectional LSTM, which outputs a 4-class prediction at every frame. Training applies cross-entropy loss at every frame, directly optimizing for early commitment.

```
Input [1, 128, T]
  → 4× Conv(32×1) + BN + ReLU       [64, 4, T]
  → Flatten freq → permute           [T, 256]
  → Unidirectional LSTM(256→128)     [T, 128]
  → Linear(128→4)                    [T, 4]  ← prediction at every frame
```

## Datasets

- **RAVDESS** — 24 speakers, 1,440 clips, 8 emotions (4-way subset used)
- **ESD-English** — 10 speakers, 17,500 clips, 5 emotions (4-way subset used)

Clips are fed as full variable-length utterances; no fixed-window segmentation.

## Quickstart (local)

```bash
pip install -r requirements.txt

# Preprocess RAVDESS
python data/preprocess.py \
  --dataset ravdess \
  --raw_dir /path/to/RAVDESS \
  --out_dir data/processed/ravdess

# Train
python train.py --config configs/crnn_ravdess.yaml

# Evaluate
python evaluate.py \
  --config configs/crnn_ravdess.yaml \
  --checkpoint runs/ravdess/best.pt
```

## Google Colab

Open `notebooks/train_colab.ipynb`. It mounts Google Drive, copies preprocessed `.npz` files to `/content/`, and runs training.

**Workflow:** preprocess locally → upload `.npz` files to `MyDrive/cs231n-ser/data/` → open notebook each session.

## Evaluation

The primary output is a **latency-accuracy curve**: per-frame accuracy over the test set, broken down by emotion class. This shows how quickly the model commits to a correct prediction.

Additional metrics: UAR (Unweighted Average Recall), weighted accuracy, confusion matrix, per-emotion first-correct-frame distribution.

## Repository Layout

```
cs231n-ser/
├── configs/          # YAML configs per dataset
├── data/
│   ├── preprocess.py # raw audio → log-mel .npz
│   └── dataset.py    # PyTorch Dataset
├── models/
│   └── crnn.py       # AnisotropicCRNN
├── utils/
│   └── metrics.py    # UAR, per-frame accuracy, etc.
├── train.py
├── evaluate.py
└── notebooks/
    └── train_colab.ipynb
```
