import numpy as np
import torch
from torch.utils.data import Dataset


class LabeledChunkDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[index]
        label = self.labels[index]
        return torch.from_numpy(sample).permute(1, 0), torch.tensor(label, dtype=torch.long)


class PhaseLabeledChunkDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, head_indices: np.ndarray):
        if labels.shape[0] != head_indices.shape[0]:
            raise ValueError("labels and head_indices must have the same length")
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.head_indices = head_indices.astype(np.int64)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.data[index]
        label = self.labels[index]
        head_index = self.head_indices[index]
        return (
            torch.from_numpy(sample).permute(1, 0),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(head_index, dtype=torch.long),
        )
