import torch
import torch.nn as nn


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

    forward(x) returns logits of shape [1, T, num_classes].
    Training applies cross-entropy at every frame; inference inspects all T
    outputs to build a latency-accuracy curve.
    """

    def __init__(
        self,
        num_classes: int = 4,
        conv_channels: list = None,
        kernel_freq: int = 32,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [1, 8, 16, 32, 64]

        assert len(conv_channels) == 5, "conv_channels must have 5 entries (in + 4 out)"

        conv_layers = []
        for i in range(4):
            in_ch = conv_channels[i]
            out_ch = conv_channels[i + 1]
            conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_freq, 1), padding=0),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        self.cnn = nn.Sequential(*conv_layers)

        # After 4 conv layers with kernel_freq=32 on 128 mel bins:
        # freq_out = 128 - 4*(kernel_freq-1) = 128 - 4*31 = 4
        freq_out = 128 - 4 * (kernel_freq - 1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 1, 128, T]
        out = self.cnn(x)                        # [1, 64, 4, T]
        B, C, F, T = out.shape
        out = out.permute(0, 3, 1, 2)            # [1, T, 64, 4]
        out = out.reshape(B, T, C * F)           # [1, T, 256]
        out, _ = self.lstm(out)                  # [1, T, lstm_hidden]
        out = self.dropout(out)
        logits = self.classifier(out)            # [1, T, num_classes]
        return logits
