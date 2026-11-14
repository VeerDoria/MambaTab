"""
Data splitting utilities.

Creates reproducible train/val/test splits following the paper's
70/10/20 protocol with seed control.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_split(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    target_column: str = "class",
    stratify: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Create a single train/val/test split.

    Args:
        df: Full DataFrame.
        seed: Random seed for reproducibility.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        target_column: Name of the target column for stratification.
        stratify: Whether to stratify the split by target.

    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    strat_col = df[target_column] if stratify else None

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=strat_col,
    )

    # Second split: train vs val
    val_relative = val_ratio / (train_ratio + val_ratio)
    strat_col_tv = train_val_df[target_column] if stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=seed,
        stratify=strat_col_tv,
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_split_indices(
    splits: Dict[str, pd.DataFrame],
    save_dir: str,
    seed: int,
) -> None:
    """Save split indices to disk for reproducibility.

    Args:
        splits: Dictionary of DataFrames from create_split.
        save_dir: Directory to save index files.
        seed: Seed used for the split.
    """
    os.makedirs(save_dir, exist_ok=True)
    indices = {
        key: df.index.tolist() for key, df in splits.items()
    }
    path = os.path.join(save_dir, f"split_indices_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(indices, f)


def load_split_indices(
    save_dir: str, seed: int
) -> Optional[Dict[str, List[int]]]:
    """Load previously saved split indices.

    Returns:
        Dictionary of index lists, or None if not found.
    """
    path = os.path.join(save_dir, f"split_indices_seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)
