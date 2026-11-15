"""
Vanilla supervised learning experiment orchestrator.

Replicates Table II from the MambaTab paper by running all models on all
datasets with multiple seeds, aggregating results, and generating
comparison tables and visualisations.
"""

import json
import os
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from src.evaluation.analysis import (
    PAPER_VANILLA_RESULTS,
    aggregate_seed_results,
    create_comparison_table,
    save_results,
)
from src.evaluation.visualization import generate_all_vanilla_plots
from src.training.trainer import run_single_experiment
from src.utils.config import load_config
from src.utils.logging_utils import save_experiment_meta, setup_logger

logger = setup_logger("training.vanilla")

# Default lists when the caller does not specify
ALL_MODELS = [
    "mambatab_d",
    "mambatab_t",
    "xgboost",
    "xgboost_2022",
    "logistic_regression",
    "ft_transformer",
    "dcn",
]
ALL_DATASETS = ["adult", "income", "blastchar", "credit_g", "cylinder_bands"]


def _print_experiment_plan(
    model_names: List[str],
    dataset_names: List[str],
    seeds: List[int],
) -> None:
    """Print a human-readable experiment plan."""
    total = len(model_names) * len(dataset_names) * len(seeds)
    print("\n" + "=" * 70)
    print("  VANILLA SUPERVISED LEARNING EXPERIMENTS")
    print("  Replication of Table II — MambaTab Paper")
    print("=" * 70)
    print(f"  Models   ({len(model_names)}): {', '.join(model_names)}")
    print(f"  Datasets ({len(dataset_names)}): {', '.join(dataset_names)}")
    print(f"  Seeds    ({len(seeds)}): {seeds}")
    print(f"  Total runs: {total}")
    print("=" * 70 + "\n")


def _print_summary_table(
    all_results: Dict,
    dataset_names: List[str],
    metric: str = "auroc",
) -> None:
    """Print a formatted comparison table to the console."""
    # Determine column widths
    model_col_w = max(20, max(len(m) for m in all_results) + 2)
    ds_col_w = 16

    header = f"{'Model':<{model_col_w}}"
    for ds in dataset_names:
        header += f"{ds:>{ds_col_w}}"
    header += f"{'Avg':>{ds_col_w}}"

    unit_label = " (seconds)" if metric == "training_time" else ""
    print("\n" + "=" * len(header))
    print(f"  {metric.upper()}{unit_label} Comparison Table")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model_name, ds_results in all_results.items():
        row = f"{model_name:<{model_col_w}}"
        values = []
        for ds in dataset_names:
            if ds in ds_results:
                ds_data = ds_results[ds]
                metric_data = ds_data.get("metrics", ds_data) if isinstance(ds_data, dict) else {}
                val_stat = metric_data.get(metric, {})
                if isinstance(val_stat, dict):
                    mean = val_stat.get("mean", float("nan"))
                    std = val_stat.get("std", 0.0)
                elif isinstance(val_stat, (int, float)):
                    mean = float(val_stat)
                    std = 0.0
                else:
                    mean = float("nan")
                    std = 0.0

                if not np.isnan(mean):
                    if metric == "training_time":
                        row += f"{mean:>{ds_col_w - 6}.2f}s±{std:<4.2f}"
                    else:
                        row += f"{mean:>{ds_col_w - 7}.4f}±{std:<5.4f}"
                    values.append(mean)
                else:
                    row += f"{'N/A':>{ds_col_w}}"
            else:
                row += f"{'N/A':>{ds_col_w}}"
        # Average across datasets
        if values:
            avg = np.mean(values)
            if metric == "training_time":
                row += f"{avg:>{ds_col_w - 1}.2f}s"
            else:
                row += f"{avg:>{ds_col_w}.4f}"
        else:
            row += f"{'N/A':>{ds_col_w}}"
        print(row)

    print("=" * len(header) + "\n")


def run_vanilla_experiments(
    model_names: Optional[List[str]] = None,
    dataset_names: Optional[List[str]] = None,
    config: Optional[dict] = None,
    results_dir: Optional[str] = None,
) -> Dict:
    """Run the complete vanilla supervised learning experiments.

    Runs all models on all datasets with all seeds, aggregates results,
    generates comparison tables and visualisations.

    Args:
        model_names: List of model names.  Defaults to all 5 models.
        dataset_names: List of dataset names.  Defaults to all 5 datasets.
        config: Configuration dict.  Loaded from default if ``None``.
        results_dir: Where to save results.  Defaults to ``results/vanilla/``.

    Returns:
        Nested dict::

            model_name -> dataset_name -> {
                'metrics': {metric: {'mean': float, 'std': float}},
                'num_parameters': int,
            }
    """
    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    if model_names is None:
        model_names = ALL_MODELS
    if dataset_names is None:
        dataset_names = ALL_DATASETS
    if config is None:
        config = load_config()
    if results_dir is None:
        results_dir = os.path.join(config["results"]["base_dir"], "vanilla")

    seeds = config["training"]["seeds"]

    # ------------------------------------------------------------------
    # 1. Print experiment plan
    # ------------------------------------------------------------------
    _print_experiment_plan(model_names, dataset_names, seeds)

    # ------------------------------------------------------------------
    # Save experiment metadata
    # ------------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)
    save_experiment_meta(
        save_path=os.path.join(results_dir, "experiment_meta.json"),
        config=config,
        setting="vanilla",
        models=model_names,
        datasets=dataset_names,
    )

    # ------------------------------------------------------------------
    # 2–4. Main experiment loop
    # ------------------------------------------------------------------
    # Load existing raw results if present to allow incremental model additions
    raw_results_path = os.path.join(results_dir, "raw_results.json")
    if os.path.exists(raw_results_path):
        try:
            with open(raw_results_path, "r") as f:
                raw_results = json.load(f)
        except Exception:
            raw_results = {}
    else:
        raw_results = {}

    for m in model_names:
        if m not in raw_results:
            raw_results[m] = {}
        for d in dataset_names:
            raw_results[m][d] = []

    failed_runs: List[Dict[str, Any]] = []

    # tqdm is optional but provides clean in-place progress in terminals and Kaggle/Colab notebooks
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

    total_runs = len(model_names) * len(dataset_names) * len(seeds)
    run_list = [
        (model, dataset, seed)
        for model in model_names
        for dataset in dataset_names
        for seed in seeds
    ]

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            run_list,
            total=total_runs,
            desc="Vanilla Benchmark",
            unit="run",
            dynamic_ncols=True,
        )
        iterable = pbar
    else:
        iterable = run_list

    for idx, (model_name, dataset_name, seed) in enumerate(iterable, 1):
        run_label = f"{model_name}/{dataset_name}/seed={seed}"
        if pbar is not None:
            pbar.set_description(f"Running: {model_name[:12]} on {dataset_name} (s={seed})")
        else:
            print(f"[{idx}/{total_runs}] Running {run_label} ...", flush=True)

        try:
            result = run_single_experiment(
                model_name=model_name,
                dataset_name=dataset_name,
                seed=seed,
                config=config,
            )
            raw_results[model_name][dataset_name].append(result)
            t_run = result.get("training_time", 0.0)
            auroc_run = result.get("metrics", {}).get("auroc", 0.0)
            if pbar is not None:
                pbar.set_postfix({
                    "last": f"{model_name[:8]}/{dataset_name[:6]}",
                    "time": f"{t_run:.2f}s",
                    "auroc": f"{auroc_run:.4f}",
                })
            else:
                print(f"    -> Done in {t_run:.2f}s | AUROC: {auroc_run:.4f}")
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"FAILED {run_label}:\n{tb}")
            failed_runs.append(
                {"model": model_name, "dataset": dataset_name, "seed": seed, "error": tb}
            )

    # ------------------------------------------------------------------
    # 5. Save raw (per-seed) results as JSON
    # ------------------------------------------------------------------
    raw_save_path = os.path.join(results_dir, "raw_results.json")
    _save_raw_results(raw_results, raw_save_path)
    logger.info(f"Raw results saved to {raw_save_path}")

    # ------------------------------------------------------------------
    # 6. Aggregate results per (model, dataset) across seeds
    # ------------------------------------------------------------------
    aggregated: Dict[str, Dict[str, Any]] = {}
    for model_name in model_names:
        aggregated[model_name] = {}
        for dataset_name in dataset_names:
            seed_results = raw_results[model_name][dataset_name]
            if seed_results:
                agg = aggregate_seed_results(seed_results)
                aggregated[model_name][dataset_name] = agg
            else:
                logger.warning(
                    f"No successful runs for {model_name}/{dataset_name}. Skipping aggregation."
                )

    # ------------------------------------------------------------------
    # 7. Create and save comparison tables
    # ------------------------------------------------------------------
    try:
        for metric in ["auroc", "accuracy", "f1_score", "auprc", "training_time"]:
            table = create_comparison_table(aggregated, metric=metric)
            table_path = os.path.join(results_dir, f"comparison_{metric}.csv")
            table.to_csv(table_path)
            logger.info(f"Comparison table ({metric}) saved to {table_path}")
    except Exception:
        logger.error(f"Failed to create comparison tables:\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # 8. Create paper comparison table
    # ------------------------------------------------------------------
    try:
        if PAPER_VANILLA_RESULTS:
            paper_table = create_comparison_table(aggregated, metric="auroc")
            paper_table_path = os.path.join(results_dir, "paper_comparison_auroc.csv")
            paper_table.to_csv(paper_table_path)
            logger.info(f"Paper comparison table saved to {paper_table_path}")
    except Exception:
        logger.error(f"Failed to create paper comparison table:\n{traceback.format_exc()}")

    # Save aggregated results via the analysis module
    try:
        save_results(aggregated, results_dir, setting="vanilla")
    except Exception:
        logger.error(f"Failed to save aggregated results:\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # 9. Generate visualisations
    # ------------------------------------------------------------------
    try:
        figures_dir = os.path.join(results_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        generate_all_vanilla_plots(
            all_results=aggregated,
            save_dir=figures_dir,
            paper_results=PAPER_VANILLA_RESULTS,
        )
        logger.info(f"Figures saved to {figures_dir}")
    except Exception:
        logger.error(f"Failed to generate visualisations:\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # 10. Print summary to console
    # ------------------------------------------------------------------
    _print_summary_table(aggregated, dataset_names, metric="auroc")
    _print_summary_table(aggregated, dataset_names, metric="training_time")

    if failed_runs:
        print(f"\n⚠  {len(failed_runs)} run(s) failed. See logs for details.")
        for fr in failed_runs:
            print(f"   - {fr['model']}/{fr['dataset']}/seed={fr['seed']}")

    return aggregated


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _save_raw_results(
    raw_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    path: str,
) -> None:
    """Serialise raw per-seed results to JSON."""

    def _make_serialisable(obj: Any) -> Any:
        """Convert numpy types to Python builtins for JSON."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serialisable(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_make_serialisable(raw_results), f, indent=2)
