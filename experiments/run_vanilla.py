#!/usr/bin/env python3
"""
Vanilla Supervised Learning Experiments
=======================================
Replication of Table II from the MambaTab paper.

Usage:
    python experiments/run_vanilla.py                    # Run all
    python experiments/run_vanilla.py --models mambatab_d xgboost  # Specific models
    python experiments/run_vanilla.py --datasets adult credit_g    # Specific datasets
    python experiments/run_vanilla.py --seeds 3                    # Fewer seeds
    python experiments/run_vanilla.py --download                   # Download datasets first
"""

import argparse
import os
import sys

# Ensure the project root is on sys.path so that ``from src.…`` imports work
# regardless of the working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.download import download_all_datasets, download_dataset
from src.training.vanilla import (
    ALL_DATASETS,
    ALL_MODELS,
    _print_summary_table,
    run_vanilla_experiments,
)
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("experiments.run_vanilla")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run vanilla supervised learning experiments (MambaTab Table II replication).",
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
        help="Number of random seeds (uses the first N from the config). Default: 5.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets before running experiments.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save results. Default: results/vanilla/.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to config YAML file. Default: configs/default.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for the vanilla experiment runner."""
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    config = load_config(config_path=args.config)

    # Override seeds if requested
    if args.seeds is not None:
        all_seeds = config["training"]["seeds"]
        config["training"]["seeds"] = all_seeds[: args.seeds]
        config["training"]["num_seeds"] = args.seeds

    model_names = args.models if args.models else None
    dataset_names = args.datasets if args.datasets else None

    # ------------------------------------------------------------------
    # 2. Optionally download datasets
    # ------------------------------------------------------------------
    if args.download:
        print("\nDownloading datasets …")
        targets = dataset_names if dataset_names else ALL_DATASETS
        for ds_name in targets:
            try:
                download_dataset(ds_name)
            except Exception as exc:
                logger.error(f"Failed to download {ds_name}: {exc}")
        print("Download complete.\n")

    # ------------------------------------------------------------------
    # 3. Run experiments
    # ------------------------------------------------------------------
    aggregated = run_vanilla_experiments(
        model_names=model_names,
        dataset_names=dataset_names,
        config=config,
        results_dir=args.results_dir,
    )

    # Results already printed by run_vanilla_experiments

    print("Done. ✓")


if __name__ == "__main__":
    main()
