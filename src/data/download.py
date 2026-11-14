"""
Dataset download utilities.

Downloads datasets from OpenML and saves them as processed CSVs
in the expected directory structure.
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("data.download")


def download_from_openml(dataset_id: int) -> pd.DataFrame:
    """Download a dataset from OpenML by its ID.

    Tries openml package first, then falls back to scikit-learn's fetch_openml.

    Args:
        dataset_id: OpenML dataset identifier.

    Returns:
        DataFrame with features and target.
    """
    try:
        import openml

        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        data, *_ = dataset.get_data(dataset_format="dataframe")
        if isinstance(data, pd.DataFrame):
            return data
    except Exception as e:
        logger.info(f"openml client fetch failed ({e}); trying sklearn.datasets.fetch_openml...")

    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    df = bunch.frame
    return df


def _download_adult(save_dir: str) -> pd.DataFrame:
    """Download and prepare the Adult (AD) dataset."""
    logger.info("Downloading Adult dataset from OpenML (ID: 1590)...")
    df = download_from_openml(1590)

    # Ensure binary target
    target_col = "class"
    if target_col in df.columns:
        # Map to binary: '>50K' -> 1, '<=50K' -> 0
        mapping = {">50K": 1, "<=50K": 0, ">50K.": 1, "<=50K.": 0}
        if df[target_col].dtype == "category":
            df[target_col] = df[target_col].astype(str)
        df[target_col] = df[target_col].map(mapping).fillna(0).astype(int)

    return df


def _download_income(save_dir: str) -> pd.DataFrame:
    """Download and prepare the Income-1995 (IC) dataset.

    Income-1995 is closely related to Adult. Following TransTab's protocol,
    we use the first 32,561 rows (original Census extraction).
    """
    logger.info("Downloading Income-1995 dataset (Adult subset, first 32561 rows)...")
    df = download_from_openml(1590)

    target_col = "class"
    if target_col in df.columns:
        mapping = {">50K": 1, "<=50K": 0, ">50K.": 1, "<=50K.": 0}
        if df[target_col].dtype == "category":
            df[target_col] = df[target_col].astype(str)
        df[target_col] = df[target_col].map(mapping).fillna(0).astype(int)

    # Take first 32,561 samples for Income-1995
    df = df.iloc[:32561].reset_index(drop=True)
    return df


def _download_blastchar(save_dir: str) -> pd.DataFrame:
    """Download and prepare the Blastchar (BL) telecom churn dataset."""
    logger.info("Downloading Blastchar dataset from OpenML (ID: 42178)...")
    df = download_from_openml(42178)

    # Ensure binary target
    target_col = "Churn"
    if target_col in df.columns:
        mapping = {"Yes": 1, "No": 0, "True": 1, "False": 0, "1": 1, "0": 0}
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .map(mapping)
            .fillna(0)
            .astype(int)
        )

    return df


def _download_credit_g(save_dir: str) -> pd.DataFrame:
    """Download and prepare the Credit-g (CG) German credit dataset."""
    logger.info("Downloading Credit-g dataset from OpenML (ID: 31)...")
    df = download_from_openml(31)

    # Ensure binary target
    target_col = "class"
    if target_col in df.columns:
        mapping = {"good": 1, "bad": 0, "1": 1, "2": 0}
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .map(mapping)
            .fillna(0)
            .astype(int)
        )

    return df


def _download_cylinder_bands(save_dir: str) -> pd.DataFrame:
    """Download and prepare the Cylinder-bands (CB) dataset."""
    logger.info("Downloading Cylinder-bands dataset from OpenML (ID: 6332)...")
    df = download_from_openml(6332)

    # Ensure binary target: paper Table I positive rate = 0.58 -> 'noband' is 1
    target_col = "band_type"
    if target_col in df.columns:
        mapping = {"noband": 1, "band": 0, "1": 1, "0": 0}
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .map(mapping)
            .fillna(0)
            .astype(int)
        )

    return df


# Registry of dataset download functions
_DOWNLOAD_REGISTRY = {
    "adult": _download_adult,
    "income": _download_income,
    "blastchar": _download_blastchar,
    "credit_g": _download_credit_g,
    "cylinder_bands": _download_cylinder_bands,
}


def download_dataset(
    dataset_name: str,
    processed_dir: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download and save a dataset.

    Args:
        dataset_name: One of 'adult', 'income', 'blastchar', 'credit_g',
                      'cylinder_bands'.
        processed_dir: Directory to save the processed CSV. Uses config default
                       if None.
        force: If True, re-download even if file exists.

    Returns:
        DataFrame with the downloaded data.
    """
    if processed_dir is None:
        config = load_config()
        processed_dir = config["data"]["processed_dir"]

    save_dir = os.path.join(processed_dir, dataset_name)
    save_path = os.path.join(save_dir, "data_processed.csv")

    if os.path.exists(save_path) and not force:
        logger.info(f"Dataset '{dataset_name}' already exists at {save_path}. Loading...")
        return pd.read_csv(save_path)

    if dataset_name not in _DOWNLOAD_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(_DOWNLOAD_REGISTRY.keys())}"
        )

    download_fn = _DOWNLOAD_REGISTRY[dataset_name]
    df = download_fn(save_dir)

    # Save processed CSV
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved {dataset_name} ({len(df)} rows) to {save_path}")

    return df


def download_all_datasets(
    processed_dir: Optional[str] = None, force: bool = False
) -> dict:
    """Download all 5 datasets.

    Returns:
        Dictionary mapping dataset names to DataFrames.
    """
    results = {}
    for name in _DOWNLOAD_REGISTRY:
        try:
            results[name] = download_dataset(name, processed_dir, force)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
    return results
