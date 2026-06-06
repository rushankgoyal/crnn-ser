"""
HarmonicDilatedBlock: parallel dilated convolutions along the frequency axis.

Voiced speech has harmonics at F0, 2·F0, 3·F0, ... . On the log-like mel axis
this spacing is uneven (large gaps at low freq, small at high freq). A bank of
dilated (frequency-axis only) convs at different rates can span multiple harmonic
rungs with a single filter. No single dilation fits the whole axis, so we use
parallel branches and fuse.

HONESTY: F0 varies per utterance and speaker, so any fixed dilation bank is a
compromise; the multi-dilation bank hedges across that variation. Taps do not
land exactly on harmonics for every speaker or phoneme.
"""

import math

import torch
import torch.nn as nn


def compute_harmonic_dilations(
    n_mels: int = 128,
    sample_rate: int = 16000,
    n_fft: int = 400,
    fmin: float = 0.0,
    fmax: float = None,
    f0_range: tuple = (80, 300),
    n_dilations: int = 4,
    print_table: bool = False,
) -> list:
    """
    Derive integer frequency-axis dilations from the mel filterbank + a target F0 range.

    Builds HTK-mel bin centre frequencies, maps harmonics of a representative F0
    (geometric mean of f0_range) to nearest bin indices, measures bin-gaps between
    consecutive harmonics, and returns a sorted unique list of n_dilations integers
    approximating the median gaps across low / mid / high thirds of the series.

    Returns ints clamped to [1, n_mels//4].
    """
    import numpy as np

    if fmax is None:
        fmax = sample_rate / 2.0

    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_lo = _hz_to_mel(max(fmin, 1e-6))
    mel_hi = _hz_to_mel(fmax)
    # n_mels+2 linearly-spaced mel points; interior n_mels are bin centres
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    bin_hz = np.array([_mel_to_hz(m) for m in mel_pts[1:-1]])  # (n_mels,)

    # Representative F0: geometric mean of range
    f0 = math.sqrt(f0_range[0] * f0_range[1])

    harmonics_hz, harmonic_bins = [], []
    h = 1
    while h * f0 <= fmax:
        hz = h * f0
        bin_idx = int(np.argmin(np.abs(bin_hz - hz)))
        harmonics_hz.append(hz)
        harmonic_bins.append(bin_idx)
        h += 1

    if len(harmonic_bins) < 2:
        return [1, 2, 4, 8][:n_dilations]

    gaps = [max(1, harmonic_bins[i + 1] - harmonic_bins[i])
            for i in range(len(harmonic_bins) - 1)]

    if print_table:
        print(f"\nF0 = {f0:.1f} Hz  (geometric mean of f0_range={f0_range})")
        print(f"n_mels={n_mels}  fmin={fmin}  fmax={fmax}  sr={sample_rate}")
        print(f"\n  {'Harmonic':<14} {'Hz':>8}  {'Mel bin':>8}  {'Gap→next':>10}")
        print("  " + "-" * 46)
        for i, (hz, b) in enumerate(zip(harmonics_hz, harmonic_bins)):
            gap_s = f"{gaps[i]:>10d}" if i < len(gaps) else f"{'(last)':>10}"
            print(f"  H{i + 1:<3d} ({i + 1:>2d}×F0)   {hz:>8.1f}  {b:>8d}  {gap_s}")
        print()

    # Sample representative gaps from low / mid / high thirds of the series
    n = len(gaps)
    thirds = [
        gaps[: max(1, n // 3)],
        gaps[max(1, n // 3): max(2, 2 * n // 3)],
        gaps[max(2, 2 * n // 3):],
    ]
    rep = [int(np.median(t)) for t in thirds if t]

    max_dil = max(1, n_mels // 4)
    dilations = sorted(set(max(1, min(g, max_dil)) for g in rep))

    # Extend to n_dilations with geometric doubling if needed
    while len(dilations) < n_dilations:
        nxt = min(dilations[-1] * 2, max_dil)
        if nxt in dilations:
            break
        dilations.append(nxt)

    return dilations[:n_dilations]


class HarmonicDilatedBlock(nn.Module):
    """
    Bank of parallel frequency-axis dilated convolutions fused by a 1×1 conv.

    Each branch: Conv2d with kernel (kernel_h × 1), dilation (d, 1), and
    same-padding along the frequency axis so the freq dimension is preserved.
    The time dimension is untouched (kernel width = 1 along time).

    Branches are concatenated along channels, then fused: Conv1×1 + BN + ReLU.

    dilation_mode is resolved BEFORE constructing this block (by the caller);
    pass the resolved list of ints as `dilations`.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilations=(1, 2, 4, 8),
        kernel_h: int = 3,
        verbose: bool = False,
    ):
        super().__init__()
        assert kernel_h % 2 == 1, "kernel_h must be odd for exact same-padding"
        dilations = list(dilations)
        n = len(dilations)

        base_ch = out_ch // n
        branch_chs = [base_ch] * n
        branch_chs[-1] += out_ch - sum(branch_chs)  # remainder → last branch

        self.branches = nn.ModuleList()
        for d, b_ch in zip(dilations, branch_chs):
            pad = (d * (kernel_h - 1) // 2, 0)
            self.branches.append(
                nn.Conv2d(
                    in_ch, b_ch,
                    kernel_size=(kernel_h, 1),
                    dilation=(d, 1),
                    padding=pad,
                    bias=False,
                )
            )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            print(
                f"[HarmonicDilatedBlock] dilations={dilations}  kernel_h={kernel_h}"
                f"  in={in_ch}  out={out_ch}  params={n_params:,}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        outs = [branch(x) for branch in self.branches]
        assert all(o.shape[2] == x.shape[2] for o in outs), (
            f"Freq dim must be preserved: input {x.shape[2]}, "
            f"branch outputs {[o.shape[2] for o in outs]}"
        )
        return self.fuse(torch.cat(outs, dim=1))


# ---------------------------------------------------------------------------
# CLI: python -m models.harmonic_block --print
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print empirical harmonic dilations for the configured mel setup"
    )
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--f0_min", type=float, default=80.0)
    parser.add_argument("--f0_max", type=float, default=300.0)
    parser.add_argument("--n_dilations", type=int, default=4)
    parser.add_argument("--print", dest="print_table", action="store_true", default=True)
    args = parser.parse_args()

    dilations = compute_harmonic_dilations(
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        fmin=args.fmin,
        fmax=args.fmax,
        f0_range=(args.f0_min, args.f0_max),
        n_dilations=args.n_dilations,
        print_table=True,
    )
    print(f"Derived dilations: {dilations}")
