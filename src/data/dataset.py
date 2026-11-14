"""
PyTorch Dataset for tabular data.

Provides a simple Dataset wrapper for use with DL models (MambaTab,
FT-Transformer, DCN). Sklearn/XGBoost models use numpy arrays directly.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    """Simple PyTorch Dataset for tabular data.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader from numpy arrays.

    Args:
        X: Feature array.
        y: Target array.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for CUDA transfer.

    Returns:
        DataLoader instance.
    """
    dataset = TabularDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
