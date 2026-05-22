import torch
import torch.nn as nn

from models.harmonic_block import HarmonicDilatedBlock, compute_harmonic_dilations
from models.freq_pos import FrequencyPositionalConditioning


class AnisotropicCRNN(nn.Module):
    """
    Unidirectional CRNN with frequency-first anisotropic kernels.

    CNN: 4 layers of Conv2d with (kernel_freq × 1) kernels and no padding on the
    frequency axis. Valid convolution reduces the frequency dimension:
        128 → 97 → 66 → 35 → 4  (for default kernel_freq=32)
    giving a 4 × 64 = 256-dim feature vector per time step.

    LSTM: unidirectional, processes one 256-dim frame at a time; hidden state
    carries all temporal context from previous frames.

    Head: linear layer applied at every time step → logits [T, num_classes].

    Optional components (all off by default → byte-for-byte baseline):
      use_freq_pos / freq_pos_mode='concat': prepend learned per-bin embedding.
      use_freq_pos / freq_pos_mode='film':   per-bin gamma/beta affine modulation.
      use_harmonic_block: parallel dilated freq-axis convs before the main CNN.

    forward(x) returns logits of shape [B, T, num_classes].
    """

    def __init__(
        self,
        num_classes: int = 4,
        conv_channels: list = None,
        kernel_freq: int = 32,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        n_mels: int = 128,
        # Component B — FrequencyPositionalConditioning
        use_freq_pos: bool = False,
        freq_pos_mode: str = "concat",   # "concat" | "film"
        pos_dim: int = 1,
        pos_init: str = "learned",       # "learned" | "sinusoidal" | "linear_ramp"
        # Component A — HarmonicDilatedBlock
        use_harmonic_block: bool = False,
        harmonic_out_ch: int = 8,
        dilation_mode: str = "octave",   # "octave" | "empirical"
        dilations: list = None,
        kernel_h: int = 3,
        # empirical dilation params (passed through from config)
        sample_rate: int = 16000,
        n_fft: int = 400,
        fmin: float = 0.0,
        fmax: float = None,
        f0_range: tuple = (80, 300),
        verbose: bool = False,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [1, 8, 16, 32, 64]
        if dilations is None:
            dilations = [1, 2, 4, 8]

        assert len(conv_channels) == 5, "conv_channels must have 5 entries (in + 4 out)"

        # --- resolve effective first-conv input channels ---
        first_in_ch = conv_channels[0]  # baseline: 1

        # Component B (concat prepends channels before anything else)
        self.freq_pos = None
        if use_freq_pos and freq_pos_mode != "none":
            self.freq_pos = FrequencyPositionalConditioning(
                n_mels=n_mels,
                pos_dim=pos_dim,
                mode=freq_pos_mode,
                pos_init=pos_init,
            )
            if freq_pos_mode == "concat":
                first_in_ch += pos_dim

        # Component A (consumes current first_in_ch, outputs harmonic_out_ch)
        self.harmonic_block = None
        if use_harmonic_block:
            if dilation_mode == "empirical":
                dilations = compute_harmonic_dilations(
                    n_mels=n_mels,
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    fmin=fmin,
                    fmax=fmax,
                    f0_range=f0_range,
                    n_dilations=len(dilations),
                    print_table=verbose,
                )
            self.harmonic_block = HarmonicDilatedBlock(
                in_ch=first_in_ch,
                out_ch=harmonic_out_ch,
                dilations=dilations,
                kernel_h=kernel_h,
                verbose=verbose,
            )
            first_in_ch = harmonic_out_ch

        # --- build main CNN (only first conv's in_ch may differ from baseline) ---
        conv_layers = []
        for i in range(4):
            in_ch = first_in_ch if i == 0 else conv_channels[i]
            out_ch = conv_channels[i + 1]
            conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_freq, 1), padding=0),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        self.cnn = nn.Sequential(*conv_layers)

        freq_out = n_mels - 4 * (kernel_freq - 1)
        lstm_input_size = freq_out * conv_channels[-1]

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            print(f"[AnisotropicCRNN] total params={total:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n_mels, T]

        # concat-mode pos embedding prepended before harmonic block
        if self.freq_pos is not None and self.freq_pos.mode == "concat":
            x = self.freq_pos(x)

        # harmonic dilated block (freq dim preserved)
        if self.harmonic_block is not None:
            x = self.harmonic_block(x)

        # film-mode modulation applied to the feature map entering the main CNN
        if self.freq_pos is not None and self.freq_pos.mode == "film":
            x = self.freq_pos(x)

        out = self.cnn(x)                        # [B, 64, freq_out, T]
        B, C, F, T = out.shape
        out = out.permute(0, 3, 1, 2)            # [B, T, 64, freq_out]
        out = out.reshape(B, T, C * F)           # [B, T, lstm_input_size]
        out, _ = self.lstm(out)                  # [B, T, lstm_hidden]
        out = self.dropout(out)
        logits = self.classifier(out)            # [B, T, num_classes]
        return logits
