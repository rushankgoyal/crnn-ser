"""
SharanCRNN: representative published CNN-BiLSTM baseline for SER.

This is the architecture we benchmark against on slide 4 of the milestone deck:
a "prominent published CNN-RNN approach by Sharan et al." The exact paper is one
of several Sharan-authored SER works; we implement the architecture pattern
common across them rather than a single paper byte-for-byte. That pattern is:

  Conv2d(3×3, same-pad) → BN → ReLU → MaxPool(2×2)   ← stack of 3–4 blocks
  Reshape over (channels × freq) per time step
  BiLSTM
  Last-frame classification

Differences from our proposed model that this baseline exposes:
  • Square 3×3 kernels (vs. our 32×1 freq-only) → tests "kernel shape" hypothesis
  • MaxPool downsampling on both axes (vs. our valid-conv freq-only reduction)
  • Bidirectional LSTM (vs. our unidirectional) → tests "causality" hypothesis
  • Last-frame loss (vs. our per-frame) → tests "early-commitment" hypothesis

So this single baseline differs from the proposed model along multiple axes at
once. Use it for an overall apples-to-apples comparison against the published
SOTA pattern, and use the single-knob baselines (baseline_square_kernel,
baseline_bilstm, baseline_last_frame_loss) for causal isolation of each effect.

Output:
  Forward returns per-frame logits [B, T_pooled, num_classes] for consistency
  with the evaluate.py latency-curve pipeline. The training script applies
  last-frame loss when loss_mode='last_frame' is set in the config.
  T_pooled = T // (2 ** n_time_pools); frame stride in ms scales accordingly.
"""

import torch
import torch.nn as nn


class SharanCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        n_mels: int = 128,
        conv_channels: list = None,        # in + 4 out, e.g. [1, 32, 64, 128, 128]
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        n_time_pools: int = 3,             # how many MaxPool layers pool the time axis
        verbose: bool = False,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [1, 32, 64, 128, 128]
        assert len(conv_channels) == 5, "conv_channels must have 5 entries (in + 4 out)"
        assert 0 <= n_time_pools <= 4, "n_time_pools must be 0..4"

        blocks = []
        cur_freq = n_mels
        for i in range(4):
            in_ch = conv_channels[i]
            out_ch = conv_channels[i + 1]
            # First n_time_pools blocks pool both axes; remaining blocks pool only freq.
            # This keeps T from collapsing too aggressively (the latency-curve eval
            # still needs enough time steps to be meaningful).
            time_pool = 2 if i < n_time_pools else 1
            pool_kernel = (2, time_pool)
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pool_kernel),
                nn.Dropout2d(p=0.1),
            ]
            cur_freq //= 2
        self.cnn = nn.Sequential(*blocks)
        self.n_time_pools = n_time_pools

        # After 4 freq-pools of 2x, freq = n_mels // 16 (= 8 for 128 mels)
        freq_out = cur_freq
        lstm_input_size = freq_out * conv_channels[-1]

        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        head_in = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Linear(head_in, num_classes)

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            print(f"[SharanCRNN] total params={total:,}  freq_out={freq_out}  "
                  f"lstm_input={lstm_input_size}  bidirectional={bidirectional}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n_mels, T]
        out = self.cnn(x)                         # [B, C, F', T']
        B, C, F, T = out.shape
        out = out.permute(0, 3, 1, 2).reshape(B, T, C * F)  # [B, T', C*F]
        out, _ = self.lstm(out)                   # [B, T', H or 2H]
        out = self.dropout(out)
        logits = self.classifier(out)             # [B, T', num_classes]
        return logits
