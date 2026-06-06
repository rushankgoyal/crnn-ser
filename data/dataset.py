import numpy as np
import torch
from torch.utils.data import Dataset


def collate_pad(batch):
    """Pad a batch of (spec [1, F, T_i], label[, vad]) to the batch's max length.

    Pad value is 0.0, which equals the per-bin normalized mean. Padded frames are
    excluded from the LSTM (via pack_padded_sequence) and from the loss (via the
    length mask), so padding never affects the model's outputs on real frames.

    Returns:
        specs   FloatTensor [B, 1, F, T_max]
        labels  LongTensor  [B]
        lengths LongTensor  [B]   real frame counts per clip
        vads    FloatTensor [B, 3]  ONLY when every item in the batch carries a
                                    VAD tuple (3-element items); omitted otherwise
                                    to preserve the existing 3-return contract.
    """
    has_vad = len(batch[0]) == 3
    if has_vad:
        specs, labels, vads = zip(*batch)
    else:
        specs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[-1] for s in specs], dtype=torch.long)
    t_max = int(lengths.max())
    b = len(specs)
    f = specs[0].shape[1]
    padded = torch.zeros(b, 1, f, t_max, dtype=torch.float32)
    for i, s in enumerate(specs):
        padded[i, :, :, : s.shape[-1]] = s
    if has_vad:
        return padded, torch.stack(labels), lengths, torch.stack(vads)
    return padded, torch.stack(labels), lengths


class SERDataset(Dataset):
    """
    Loads variable-length log-mel spectrograms from a .npz file produced by preprocess.py.

    Each item is (spec_tensor, label) where:
        spec_tensor : FloatTensor of shape [1, 128, T_i]
        label       : LongTensor scalar
    """

    def __init__(
        self,
        npz_path: str,
        augment: bool = False,
        n_freq_masks: int = 2,
        freq_mask_param: int = 15,
        n_time_masks: int = 2,
        time_mask_param: int = 25,
        time_mask_p: float = 0.2,
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.specs = data["X"]    # object array, each element is (128, T_i)
        self.labels = data["y"]   # int64 array of shape (N,)
        self.vads = data["vad"] if "vad" in data.files else None  # (N, 3) or None
        self.augment = augment
        self.n_freq_masks = n_freq_masks
        self.freq_mask_param = freq_mask_param
        self.n_time_masks = n_time_masks
        self.time_mask_param = time_mask_param
        self.time_mask_p = time_mask_p

    def __len__(self) -> int:
        return len(self.labels)

    def _spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment-style masking to one spectrogram (training only).

        Implements the frequency- and time-masking operations of SpecAugment
        (Park et al., "SpecAugment: A Simple Data Augmentation Method for
        Automatic Speech Recognition," Interspeech 2019, arXiv:1904.08779). The
        third operation of the original policy, time warping, is omitted: it is
        the most computationally expensive component and yields the smallest
        accuracy gain, so we follow common practice and apply masking only.

        Augmentation is applied on the fly at load time, so a fresh random mask
        is drawn each epoch; the validation and test sets are never augmented.
        The input is a per-frequency-bin z-score-normalized log-mel spectrogram
        of shape (1, F, T) with F = 128 mel bins and variable frame length T.

        Frequency masking (repeated n_freq_masks times):
            f  ~ Uniform{0, ..., freq_mask_param}
            f0 ~ Uniform{0, ..., F - f}
            spec[:, f0 : f0 + f, :] <- 0

        Time masking (repeated n_time_masks times):
            w_max = min(time_mask_param, floor(time_mask_p * T))
            t  ~ Uniform{0, ..., w_max}
            t0 ~ Uniform{0, ..., T - t}
            spec[:, :, t0 : t0 + t] <- 0

        Masked regions are set to 0.0. Because the spectrograms are z-score
        normalized per mel bin upstream, 0.0 equals the per-bin mean, so masking
        injects no spurious energy (equivalent to the mean-masking of the
        original paper). The time-mask width is upper-bounded by a fraction
        time_mask_p of the clip length (the paper's p*tau bound): clips here are
        short and variable after silence trimming (~190-250 frames), so a fixed
        absolute width could erase a large fraction of a short utterance. As the
        model computes cross-entropy at every frame, masked frames keep the
        utterance label, requiring the recurrent layer to bridge the gap from
        temporal context.

        Default hyperparameters: n_freq_masks=2, freq_mask_param=15 mel bins,
        n_time_masks=2, time_mask_param=25 frames, time_mask_p=0.2.
        """
        # spec: (1, F, T)
        F, T = spec.shape[1], spec.shape[2]
        for _ in range(self.n_freq_masks):
            w = int(torch.randint(0, self.freq_mask_param + 1, (1,)))
            if 0 < w < F:
                s = int(torch.randint(0, F - w + 1, (1,)))
                spec[:, s:s + w, :] = 0.0
        # cap time-mask width at a fraction of this clip's length (clips are short/variable)
        max_t = max(1, min(self.time_mask_param, int(self.time_mask_p * T)))
        for _ in range(self.n_time_masks):
            w = int(torch.randint(0, max_t + 1, (1,)))
            if 0 < w < T:
                s = int(torch.randint(0, T - w + 1, (1,)))
                spec[:, :, s:s + w] = 0.0
        return spec

    def __getitem__(self, idx: int):
        spec = self.specs[idx].astype(np.float32)          # (128, T)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, 128, T)
        if self.augment:
            spec_tensor = self._spec_augment(spec_tensor)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.vads is not None:
            vad = torch.tensor(self.vads[idx], dtype=torch.float32)  # (3,)
            return spec_tensor, label, vad
        return spec_tensor, label
