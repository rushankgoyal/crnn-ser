import torch
import torch.nn as nn


class FrameLayerNorm(nn.Module):
    """Per-frame LayerNorm for a conv feature map [B, C, F, T].

    Normalizes each time frame independently over (channels x frequency), with a
    per-channel affine transform. Because each frame is normalized using only its
    own content, this is:
      * batch-size independent (stable from batch size 1 upward), and
      * padding invariant — a valid frame's output does not depend on how the clip
        is zero-padded in a batch, which keeps batched training consistent with the
        per-clip evaluation path, and
      * strictly causal — no statistics leak across time.

    This matches the paper's "LayerNorm in the convolutional stack" (PDF §3.4) for a
    frame-synchronous model, where the time axis is the sequence (token) dimension.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T] — normalize over (C, F) for each (B, T)
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def make_norm(norm: str, num_features: int) -> nn.Module:
    """2-D normalization layer, selectable and batch-size independent by default.

    The paper (PDF §3.4) uses LayerNorm in the convolutional stack. With the
    batch-size-32 padded training described there, BatchNorm is a poor fit because
    padded frames pollute the per-channel statistics. We therefore default to a
    per-frame LayerNorm:

      'layernorm' -> FrameLayerNorm: normalizes each time frame over (channels x
                     frequency). Batch-independent, padding-invariant, causal.
      'batchnorm' -> BatchNorm2d: original behavior (kept for ablation/back-compat).
    """
    key = (norm or "layernorm").lower()
    if key in ("layernorm", "ln", "framelayernorm"):
        return FrameLayerNorm(num_features)
    if key in ("batchnorm", "bn"):
        return nn.BatchNorm2d(num_features)
    raise ValueError(f"unknown norm {norm!r} (expected 'layernorm' or 'batchnorm')")
