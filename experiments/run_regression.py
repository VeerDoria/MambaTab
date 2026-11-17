#!/usr/bin/env python3
"""
Tabular Regression Benchmark Runner
===================================
Evaluates MambaTab (MambaTab-D, MambaTab-T) and tabular baselines
(XGBoost, XGBoost-Paper, Ridge Regression, FT-Transformer, DCN)
on standard regression benchmarks:
- California Housing (20.6k samples, 8 features)
- Abalone (4.2k samples, 8 features)
- Concrete Compressive Strength (1.0k samples, 8 features)

Usage:
    python3 experiments/run_regression.py                             # Run all 7 models on all 3 datasets
    python3 experiments/run_regression.py --download                  # Download datasets first
    python3 experiments/run_regression.py --models mambatab_d xgboost # Specific models
    python3 experiments/run_regression.py --datasets concrete         # Specific dataset
    python3 experiments/run_regression.py --seeds 1                   # Fast 1-seed test
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.download import download_dataset
from src.training.regression import (
    REGRESSION_DATASETS,
    REGRESSION_MODELS,
    run_regression_experiments,
)
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("experiments.run_regression")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the regression runner."""
    parser = argparse.ArgumentParser(
        description="Run tabular regression benchmarks across MambaTab and baseline models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=f"Model(s) to evaluate. Choices: {', '.join(REGRESSION_MODELS)}. Default: all.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help=f"Dataset(s) to use. Choices: {', '.join(REGRESSION_DATASETS)}. Default: all.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        metavar="N",
        help="Number of random seeds (uses first N from config). Default: 5.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download regression datasets before running experiments.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save results. Default: auto-generated incremental folder (results/regression/run_1, etc.).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to custom config YAML file. Default: configs/default.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the regression experiment runner."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config) if args.config else load_config()

    # Handle seeds override
    if args.seeds is not None:
        config.setdefault("training", {})["seeds"] = list(range(args.seeds))

    # Resolve models & datasets
    model_names = args.models if args.models else REGRESSION_MODELS
    dataset_names = args.datasets if args.datasets else REGRESSION_DATASETS

    # Validate names
    for m in model_names:
        if m not in REGRESSION_MODELS:
            logger.error(f"Unknown model '{m}'. Choices: {REGRESSION_MODELS}")
            sys.exit(1)

    for d in dataset_names:
        if d not in REGRESSION_DATASETS:
            logger.error(f"Unknown dataset '{d}'. Choices: {REGRESSION_DATASETS}")
            sys.exit(1)

    # Pre-download datasets if requested or missing
    processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
    for ds in dataset_names:
        csv_path = os.path.join(processed_dir, ds, "data_processed.csv")
        if args.download or not os.path.exists(csv_path):
            logger.info(f"Downloading/preparing regression dataset: {ds}")
            download_dataset(ds, processed_dir=processed_dir, force=args.download)

    # Launch regression benchmark
    run_regression_experiments(
        model_names=model_names,
        dataset_names=dataset_names,
        config=config,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
