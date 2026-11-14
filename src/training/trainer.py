"""
Common training utilities for the MambaTab replication experiments.

Provides ``run_single_experiment``, a high-level function that orchestrates
a complete single-model / single-dataset / single-seed experiment run:
seed → load → split → preprocess → train → predict → evaluate.
"""

import os
import time
import traceback
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.preprocess import TabularPreprocessor
from src.data.splits import create_split
from src.evaluation.metrics import compute_metrics
from src.models.factory import create_model
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("training.trainer")


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    seed: int,
    config: dict,
) -> Dict[str, Any]:
    """Run a single model on a single dataset with a single seed.

    This function handles the complete pipeline:
      1. Set random seed
      2. Load dataset (from processed CSV)
      3. Create train/val/test split
      4. Preprocess (fit on train, transform val/test)
      5. Create model
      6. Train model
      7. Generate predictions on test set
      8. Compute all metrics
      9. Return results dict

    Args:
        model_name: One of ``'mambatab_d'``, ``'xgboost'``,
            ``'logistic_regression'``, ``'ft_transformer'``, ``'dcn'``.
        dataset_name: One of ``'adult'``, ``'income'``, ``'blastchar'``,
            ``'credit_g'``, ``'cylinder_bands'``.
        seed: Random seed.
        config: Configuration dictionary (as returned by ``load_config``).

    Returns:
        Dictionary containing:
            - ``'metrics'``: Dict of metric_name → value
            - ``'model_name'``: str
            - ``'dataset_name'``: str
            - ``'seed'``: int
            - ``'num_parameters'``: int
            - ``'training_time'``: float (seconds)

    Raises:
        FileNotFoundError: If the processed dataset CSV does not exist.
    """
    # ------------------------------------------------------------------
    # 1. Set random seed
    # ------------------------------------------------------------------
    set_seed(seed)

    # ------------------------------------------------------------------
    # 2. Load dataset from processed CSV
    # ------------------------------------------------------------------
    processed_dir = config["data"]["processed_dir"]
    csv_path = os.path.join(processed_dir, dataset_name, "data_processed.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {csv_path}. "
            f"Run download first (e.g. python experiments/run_vanilla.py --download)."
        )
    df = pd.read_csv(csv_path)
    logger.info(
        f"Loaded {dataset_name} ({len(df)} rows, {df.shape[1]} cols) from {csv_path}"
    )

    # ------------------------------------------------------------------
    # 3. Create train / val / test split
    # ------------------------------------------------------------------
    target_column = config["datasets"][dataset_name]["target_column"]
    data_cfg = config.get("data", {})
    splits = create_split(
        df,
        seed=seed,
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        test_ratio=data_cfg.get("test_ratio", 0.2),
        target_column=target_column,
    )
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]
    logger.info(
        f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    # ------------------------------------------------------------------
    # 4. Preprocess (fit on train, transform val & test)
    # ------------------------------------------------------------------
    preprocessor = TabularPreprocessor(target_column=target_column)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)
    num_features = preprocessor.num_features
    logger.info(f"Preprocessed — num_features: {num_features}")

    # ------------------------------------------------------------------
    # 5. Create model
    # ------------------------------------------------------------------
    model = create_model(model_name, num_features, config, dataset_name=dataset_name)

    # ------------------------------------------------------------------
    # 6. Train model (timed)
    # ------------------------------------------------------------------
    logger.info(f"Training {model_name} on {dataset_name} (seed={seed})...")
    t_start = time.time()
    model.fit(X_train, y_train, X_val, y_val, config)
    training_time = time.time() - t_start
    logger.info(f"Training completed in {training_time:.2f}s")

    # ------------------------------------------------------------------
    # 7. Generate predictions on test set
    # ------------------------------------------------------------------
    y_prob = model.predict_proba(X_test)

    # ------------------------------------------------------------------
    # 8. Compute all metrics
    # ------------------------------------------------------------------
    threshold = config.get("evaluation", {}).get("threshold", 0.5)
    metrics = compute_metrics(y_test, y_prob, threshold=threshold)
    logger.info(
        f"Results — AUROC: {metrics.get('auroc', 'N/A'):.4f}, "
        f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}"
    )

    # ------------------------------------------------------------------
    # 9. Return results dict
    # ------------------------------------------------------------------
    return {
        "metrics": metrics,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seed": seed,
        "num_parameters": getattr(model, "num_parameters", 0),
        "training_time": training_time,
    }
