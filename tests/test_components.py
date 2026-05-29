"""
CPU unit tests for HarmonicDilatedBlock, FrequencyPositionalConditioning,
and the wired AnisotropicCRNN.  All tests run on CPU — no GPU required.

Run:
    python -m pytest tests/test_components.py -v
or:
    python tests/test_components.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
try:
    import pytest
except ImportError:
    pytest = None

from models.harmonic_block import HarmonicDilatedBlock, compute_harmonic_dilations
from models.freq_pos import FrequencyPositionalConditioning
from models.crnn import AnisotropicCRNN
from models.sharan import SharanCRNN


# ---------------------------------------------------------------------------
# Component A — HarmonicDilatedBlock
# ---------------------------------------------------------------------------

def test_harmonic_block_shape_preserved():
    """Output freq and time dims must equal input."""
    B, C_in, F, T = 2, 1, 128, 40
    block = HarmonicDilatedBlock(in_ch=C_in, out_ch=8, dilations=[1, 2, 4, 8], kernel_h=3)
    x = torch.randn(B, C_in, F, T)
    y = block(x)
    assert y.shape == (B, 8, F, T), f"Expected ({B}, 8, {F}, {T}), got {y.shape}"


def test_harmonic_block_various_out_ch():
    """out_ch not divisible by len(dilations) should still work (remainder to last branch)."""
    block = HarmonicDilatedBlock(in_ch=1, out_ch=10, dilations=[1, 2, 4], kernel_h=3)
    x = torch.randn(1, 1, 128, 20)
    y = block(x)
    assert y.shape == (1, 10, 128, 20)


def test_harmonic_block_backward():
    block = HarmonicDilatedBlock(in_ch=1, out_ch=8, dilations=[1, 2, 4, 8])
    x = torch.randn(1, 1, 128, 30, requires_grad=True)
    loss = block(x).sum()
    loss.backward()
    assert x.grad is not None


def test_harmonic_block_single_dilation():
    block = HarmonicDilatedBlock(in_ch=3, out_ch=6, dilations=[4], kernel_h=3)
    x = torch.randn(2, 3, 128, 15)
    y = block(x)
    assert y.shape == (2, 6, 128, 15)


# ---------------------------------------------------------------------------
# Dilation computation
# ---------------------------------------------------------------------------

def test_compute_octave_dilations_defaults():
    """octave mode just uses the passed list; empirical mode returns a list."""
    d = compute_harmonic_dilations(n_mels=128, sample_rate=16000, n_dilations=4)
    assert isinstance(d, list)
    assert len(d) <= 4
    assert all(isinstance(v, int) and v >= 1 for v in d)


def test_compute_empirical_dilations_clamped():
    d = compute_harmonic_dilations(n_mels=128, sample_rate=16000, n_dilations=4)
    max_allowed = 128 // 4
    assert all(v <= max_allowed for v in d), f"Dilation exceeds n_mels//4: {d}"


def test_compute_dilations_print_table(capsys):
    compute_harmonic_dilations(n_mels=128, print_table=True)
    out = capsys.readouterr().out
    assert "F0" in out
    assert "Hz" in out


# ---------------------------------------------------------------------------
# Component B — FrequencyPositionalConditioning
# ---------------------------------------------------------------------------

def test_freqpos_concat_shape():
    """Concat mode must bump channels by pos_dim."""
    mod = FrequencyPositionalConditioning(n_mels=128, pos_dim=1, mode="concat")
    x = torch.randn(2, 1, 128, 40)
    y = mod(x)
    assert y.shape == (2, 2, 128, 40), f"Got {y.shape}"


def test_freqpos_concat_multichannel():
    mod = FrequencyPositionalConditioning(n_mels=128, pos_dim=4, mode="concat")
    x = torch.randn(2, 3, 128, 40)
    y = mod(x)
    assert y.shape == (2, 7, 128, 40)


def test_freqpos_film_shape():
    """FiLM mode must preserve shape."""
    mod = FrequencyPositionalConditioning(n_mels=128, mode="film")
    x = torch.randn(2, 8, 128, 40)
    y = mod(x)
    assert y.shape == x.shape


def test_freqpos_film_changes_output():
    """FiLM with non-trivial gamma/beta must produce different values than identity."""
    torch.manual_seed(0)
    mod = FrequencyPositionalConditioning(n_mels=128, mode="film")
    with torch.no_grad():
        mod.gamma.fill_(2.0)
        mod.beta.fill_(1.0)
    x = torch.randn(1, 8, 128, 10)
    y = mod(x)
    assert not torch.allclose(y, x), "FiLM with gamma=2, beta=1 should change output"


def test_freqpos_determinism():
    """Same seed → same pos_emb initialization."""
    torch.manual_seed(42)
    m1 = FrequencyPositionalConditioning(n_mels=128, pos_dim=2, mode="concat", pos_init="learned")
    torch.manual_seed(42)
    m2 = FrequencyPositionalConditioning(n_mels=128, pos_dim=2, mode="concat", pos_init="learned")
    assert torch.allclose(m1.pos_emb, m2.pos_emb)


def test_freqpos_sinusoidal_init():
    mod = FrequencyPositionalConditioning(n_mels=128, pos_dim=4, mode="concat", pos_init="sinusoidal")
    x = torch.randn(1, 1, 128, 10)
    y = mod(x)
    assert y.shape == (1, 5, 128, 10)


def test_freqpos_linear_ramp_init():
    mod = FrequencyPositionalConditioning(n_mels=128, pos_dim=1, mode="concat", pos_init="linear_ramp")
    emb = mod.pos_emb.detach().squeeze()
    assert emb[0].item() < emb[-1].item(), "linear_ramp should increase bin 0 → 127"


# ---------------------------------------------------------------------------
# Wired AnisotropicCRNN — all four ablation combinations
# ---------------------------------------------------------------------------

def _make_input(B=1, F=128, T=50):
    return torch.randn(B, 1, F, T)


def test_crnn_baseline():
    model = AnisotropicCRNN()
    out = model(_make_input())
    assert out.shape == (1, 50, 4)


def test_crnn_harmonic_only():
    model = AnisotropicCRNN(use_harmonic_block=True, harmonic_out_ch=8)
    out = model(_make_input())
    assert out.shape == (1, 50, 4)


def test_crnn_freqpos_concat():
    model = AnisotropicCRNN(use_freq_pos=True, freq_pos_mode="concat", pos_dim=1)
    out = model(_make_input())
    assert out.shape == (1, 50, 4)


def test_crnn_freqpos_film():
    model = AnisotropicCRNN(use_freq_pos=True, freq_pos_mode="film")
    out = model(_make_input())
    assert out.shape == (1, 50, 4)


def test_crnn_both():
    model = AnisotropicCRNN(
        use_harmonic_block=True, harmonic_out_ch=8,
        use_freq_pos=True, freq_pos_mode="concat", pos_dim=2,
    )
    out = model(_make_input())
    assert out.shape == (1, 50, 4)


def test_crnn_empirical_dilation_mode():
    model = AnisotropicCRNN(
        use_harmonic_block=True,
        harmonic_out_ch=8,
        dilation_mode="empirical",
        dilations=[1, 2, 4, 8],
    )
    out = model(_make_input())
    assert out.shape == (1, 50, 4)


def test_crnn_first_conv_in_ch_bumped():
    """With concat freq pos, first conv in_ch must be 1+pos_dim."""
    model = AnisotropicCRNN(use_freq_pos=True, freq_pos_mode="concat", pos_dim=3)
    first_conv = model.cnn[0]  # first Conv2d in Sequential
    assert first_conv.in_channels == 1 + 3, (
        f"Expected 4 in_channels, got {first_conv.in_channels}"
    )


def test_crnn_all_off_equals_baseline():
    """
    Param count and forward output must be identical to vanilla baseline
    when all new flags are explicitly off.
    """
    torch.manual_seed(7)
    baseline = AnisotropicCRNN()

    torch.manual_seed(7)
    explicit_off = AnisotropicCRNN(
        use_harmonic_block=False,
        use_freq_pos=False,
        freq_pos_mode="concat",
    )

    n_baseline = sum(p.numel() for p in baseline.parameters())
    n_explicit = sum(p.numel() for p in explicit_off.parameters())
    assert n_baseline == n_explicit, (
        f"Param count mismatch: baseline={n_baseline}, all-off={n_explicit}"
    )

    x = torch.randn(1, 1, 128, 30)
    baseline.eval()
    explicit_off.eval()
    with torch.no_grad():
        out_base = baseline(x)
        out_off = explicit_off(x)
    assert torch.allclose(out_base, out_off), (
        "All-off model output must match baseline"
    )


# ---------------------------------------------------------------------------
# Baseline variants — 3×3 square kernel, BiLSTM, Sharan
# ---------------------------------------------------------------------------

def test_crnn_square_kernel_3x3_shape():
    """3×3 kernel with freq_stride=2 should halve freq per layer (128→64→32→16→8)
       and preserve T thanks to same-padding on time."""
    model = AnisotropicCRNN(kernel_freq=3, kernel_time=3, freq_stride=2)
    out = model(_make_input(T=50))
    assert out.shape == (1, 50, 4), f"got {out.shape}"


def test_crnn_square_kernel_backward():
    model = AnisotropicCRNN(kernel_freq=3, kernel_time=3, freq_stride=2)
    x = torch.randn(1, 1, 128, 40)
    out = model(x)
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"


def test_crnn_bidirectional_shape():
    """BiLSTM variant must still produce (B, T, C) and have a 2H -> C head."""
    model = AnisotropicCRNN(bidirectional=True, lstm_hidden=64)
    out = model(_make_input(T=50))
    assert out.shape == (1, 50, 4)
    # classifier in_features should be 2 * lstm_hidden
    assert model.classifier.in_features == 128, model.classifier.in_features


def test_crnn_bidirectional_param_count_grows():
    uni = AnisotropicCRNN(bidirectional=False, lstm_hidden=64)
    bi  = AnisotropicCRNN(bidirectional=True,  lstm_hidden=64)
    n_uni = sum(p.numel() for p in uni.parameters())
    n_bi  = sum(p.numel() for p in bi.parameters())
    assert n_bi > n_uni, f"BiLSTM should have more params: {n_uni} vs {n_bi}"


def test_sharan_shape():
    """Sharan baseline returns per-frame logits at a pooled time resolution."""
    model = SharanCRNN(num_classes=4, n_time_pools=3)
    x = torch.randn(1, 1, 128, 80)  # T=80 → after 3 time-pools → T'=10
    out = model(x)
    assert out.shape[0] == 1 and out.shape[2] == 4
    assert out.shape[1] == 80 // 8, f"expected T'=10, got {out.shape[1]}"


def test_sharan_backward():
    model = SharanCRNN()
    x = torch.randn(1, 1, 128, 80)
    out = model(x)
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"


def test_crnn_backward_both():
    model = AnisotropicCRNN(
        use_harmonic_block=True, harmonic_out_ch=8,
        use_freq_pos=True, freq_pos_mode="concat",
    )
    x = torch.randn(1, 1, 128, 40)
    out = model(x)
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# Standalone runner (no pytest needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_harmonic_block_shape_preserved,
        test_harmonic_block_various_out_ch,
        test_harmonic_block_backward,
        test_harmonic_block_single_dilation,
        test_compute_octave_dilations_defaults,
        test_compute_empirical_dilations_clamped,
        test_freqpos_concat_shape,
        test_freqpos_concat_multichannel,
        test_freqpos_film_shape,
        test_freqpos_film_changes_output,
        test_freqpos_determinism,
        test_freqpos_sinusoidal_init,
        test_freqpos_linear_ramp_init,
        test_crnn_baseline,
        test_crnn_harmonic_only,
        test_crnn_freqpos_concat,
        test_crnn_freqpos_film,
        test_crnn_both,
        test_crnn_empirical_dilation_mode,
        test_crnn_first_conv_in_ch_bumped,
        test_crnn_all_off_equals_baseline,
        test_crnn_square_kernel_3x3_shape,
        test_crnn_square_kernel_backward,
        test_crnn_bidirectional_shape,
        test_crnn_bidirectional_param_count_grows,
        test_sharan_shape,
        test_sharan_backward,
        test_crnn_backward_both,
    ]

    class _FakeCapsys:
        def readouterr(self):
            class R:
                out = ""
            return R()

    passed = failed = 0
    for t in tests:
        try:
            import inspect
            sig = inspect.signature(t)
            if "capsys" in sig.parameters:
                t(_FakeCapsys())
            else:
                t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
