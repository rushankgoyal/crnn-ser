"""
FrequencyPositionalConditioning: breaks the CNN's vertical translation-equivariance
by injecting learned per-mel-bin positional information.

A standard conv is equivariant along both axes, but on a spectrogram the vertical
(frequency) position is absolute and meaningful (200 Hz != 4 kHz). Injecting each
mel row's position as data makes fixed conv weights produce position-dependent
responses, without untying weights per row (which would explode parameters).

Two modes, selected by freq_pos_mode:

  concat: prepend a learned (n_mels × pos_dim) embedding as extra input channels.
          Input [B, C, F, T] → [B, C+pos_dim, F, T].

  film:   apply per-bin affine transform  feat = gamma[f] * feat + beta[f]
          broadcast over batch, channels, time.
          Input [B, C, F, T] → [B, C, F, T]  (same shape, different values).
"""

import math

import torch
import torch.nn as nn


class FrequencyPositionalConditioning(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        pos_dim: int = 1,
        mode: str = "concat",
        pos_init: str = "learned",
    ):
        super().__init__()
        assert mode in ("concat", "film"), f"Unknown freq_pos_mode: {mode!r}"
        assert pos_init in ("learned", "sinusoidal", "linear_ramp"), (
            f"Unknown pos_init: {pos_init!r}"
        )
        self.mode = mode
        self.n_mels = n_mels
        self.pos_dim = pos_dim

        if mode == "concat":
            self.pos_emb = nn.Parameter(torch.zeros(n_mels, pos_dim))
            self._init_emb(pos_init)
        elif mode == "film":
            self.gamma = nn.Parameter(torch.ones(n_mels))
            self.beta = nn.Parameter(torch.zeros(n_mels))

    def _init_emb(self, pos_init: str) -> None:
        with torch.no_grad():
            if pos_init == "linear_ramp":
                ramp = torch.linspace(0.0, 1.0, self.n_mels).unsqueeze(1)
                self.pos_emb.copy_(ramp.expand(-1, self.pos_dim))
            elif pos_init == "sinusoidal":
                pe = torch.zeros(self.n_mels, self.pos_dim)
                pos = torch.arange(self.n_mels).float().unsqueeze(1)
                half = self.pos_dim // 2 or 1
                div = torch.exp(
                    torch.arange(0, 2 * half, 2).float()
                    * (-math.log(10000.0) / max(self.pos_dim, 2))
                )
                pe[:, 0::2] = torch.sin(pos * div[:pe[:, 0::2].shape[1]])
                if self.pos_dim > 1:
                    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
                self.pos_emb.copy_(pe)
            # 'learned': start from zeros; optimizer moves from there

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        if self.mode == "concat":
            B, _, _, T = x.shape
            # pos_emb [F, P] → [1, P, F, 1] → [B, P, F, T]
            emb = self.pos_emb.T.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, T)
            return torch.cat([x, emb], dim=1)
        else:  # film
            g = self.gamma.view(1, 1, -1, 1)
            b = self.beta.view(1, 1, -1, 1)
            return g * x + b
