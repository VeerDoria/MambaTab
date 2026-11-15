"""
Feature-Incremental Learning (FIL) Data Partitioning Utilities.

Follows the protocol from MambaTab (Section IV-C) and TransTab (Wang & Sun, 2022):
- Feature set F is partitioned into 3 non-overlapping subsets: s1, s2, s3.
- Stage 1: Uses features in s1.
- Stage 2: Uses features in s1 ∪ s2.
- Stage 3: Uses all features s1 ∪ s2 ∪ s3.
"""

from typing import List, Tuple
import numpy as np


def get_feature_subsets(num_features: int, n_stages: int = 3) -> List[int]:
    """Calculate the cumulative feature boundary for each incremental stage.

    Following official MambaTab implementation:
    subset_size = num_features // 3
    Stage 1: subset_size * 1
    Stage 2: subset_size * 2
    Stage 3: num_features (all features)

    Args:
        num_features: Total number of features in the dataset.
        n_stages: Number of incremental stages (default: 3).

    Returns:
        List of integer boundaries [end_1, end_2, ..., num_features].
    """
    if n_stages <= 1:
        return [num_features]

    subset_size = num_features // n_stages
    boundaries = []
    for stage in range(1, n_stages):
        boundaries.append(subset_size * stage)
    boundaries.append(num_features)
    return boundaries


def slice_stage_features(
    X: np.ndarray,
    end_col: int,
    zero_pad: bool = False,
    total_features: int = 0,
) -> np.ndarray:
    """Slice features for a specific incremental stage.

    Args:
        X: Feature matrix of shape (N, total_features).
        end_col: Number of active features up to this stage.
        zero_pad: If True, returns full-width matrix with columns [end_col:] zeroed out
                  (for baselines with static input architectures).
                  If False, returns sliced matrix of shape (N, end_col).
        total_features: Total feature dimension (used if zero_pad=True).

    Returns:
        Processed feature matrix for this stage.
    """
    if zero_pad:
        tot = total_features if total_features > 0 else X.shape[1]
        out = np.zeros((X.shape[0], tot), dtype=X.dtype)
        out[:, :end_col] = X[:, :end_col]
        return out
    else:
        return X[:, :end_col].copy()
