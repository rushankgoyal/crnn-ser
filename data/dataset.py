import numpy as np
import torch
from torch.utils.data import Dataset


class SERDataset(Dataset):
    """
    Loads variable-length log-mel spectrograms from a .npz file produced by preprocess.py.

    Each item is (spec_tensor, label) where:
        spec_tensor : FloatTensor of shape [1, 128, T_i]
        label       : LongTensor scalar
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.specs = data["X"]    # object array, each element is (128, T_i)
        self.labels = data["y"]   # int64 array of shape (N,)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        spec = self.specs[idx].astype(np.float32)          # (128, T)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, 128, T)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec_tensor, label
