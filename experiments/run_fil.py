#!/usr/bin/env python3
"""
Feature-Incremental Learning (FIL) Experiments Runner
=====================================================
Replication of Table III from the MambaTab paper (Section IV-C).

Usage:
    python3 experiments/run_fil.py                          # Run all models on all datasets (5 seeds)
    python3 experiments/run_fil.py --models mambatab_d      # Specific models
    python3 experiments/run_fil.py --datasets credit_g      # Specific datasets
    python3 experiments/run_fil.py --seeds 1                # Fast verification run (1 seed)
    python3 experiments/run_fil.py --stages 3               # Number of incremental stages
"""

import argparse
import os
import sys

# Ensure the project root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.download import download_all_datasets, download_dataset
from src.training.incremental import (
    ALL_DATASETS,
    ALL_MODELS,
    run_incremental_experiments,
)
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("experiments.run_fil")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Feature-Incremental Learning (FIL) experiments (MambaTab Table III replication).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "Model(s) to evaluate. "
            f"Choices: {', '.join(ALL_MODELS)}. Default: all."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help=(
            "Dataset(s) to use. "
            f"Choices: {', '.join(ALL_DATASETS)}. Default: all."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        metavar="N",
        help="Number of random seeds (uses first N from config). Default: 5.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=3,
        metavar="K",
        help="Number of incremental feature stages (default: 3).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/fil",
        metavar="DIR",
        help="Directory to save FIL results. Default: results/fil.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        metavar="PATH",
        help="Path to config YAML file. Default: configs/default.yaml.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets if missing before running.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for the FIL experiment runner."""
    args = parse_args()

    # 1. Load configuration
    config = load_config(config_path=args.config)

    # Override seeds if requested
    if args.seeds is not None:
        all_seeds = config.get("training", {}).get("seeds", [0, 1, 2, 3, 4])
        seeds = all_seeds[: args.seeds]
    else:
        seeds = config.get("training", {}).get("seeds", [0, 1, 2, 3, 4])

    model_names = args.models if args.models else ALL_MODELS
    dataset_names = args.datasets if args.datasets else ALL_DATASETS

    # 2. Optionally download datasets
    if args.download:
        print("\nVerifying datasets ...")
        for ds_name in dataset_names:
            try:
                download_dataset(ds_name)
            except Exception as exc:
                logger.error(f"Failed to download {ds_name}: {exc}")
        print("Dataset check complete.\n")

    # 3. Run FIL benchmark
    run_incremental_experiments(
        models=model_names,
        datasets=dataset_names,
        seeds=seeds,
        config_path=args.config,
        save_dir=args.results_dir,
        n_stages=args.stages,
    )

    print("\nFeature-Incremental Learning experiments completed successfully. ✓")


if __name__ == "__main__":
    main()
