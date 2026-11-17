"""
Regression experiment runner and orchestrator.

Provides the training loop, seed aggregation, and persistence for tabular regression
benchmarks across standard models (MambaTab-D, MambaTab-T, XGBoost, XGBoost-Paper,
Ridge/Linear Regression, FT-Transformer, DCN) on regression datasets:
- California Housing (large, continuous)
- Abalone (medium, mixed)
- Concrete Compressive Strength (small, dense)
"""

import copy
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.preprocess import TabularPreprocessor
from src.data.splits import create_split
from src.evaluation.metrics import compute_regression_metrics, get_regression_metric_names
from src.models.base import TorchBaseModel
from src.models.factory import create_model
from src.utils.config import load_config
from src.utils.experiment_utils import generate_run_summary_md, get_next_run_dir
from src.utils.logging_utils import save_experiment_meta, setup_logger
from src.utils.reproducibility import get_device, set_seed

logger = setup_logger("training.regression")

REGRESSION_MODELS = [
    "mambatab_d",
    "mambatab_t",
    "xgboost",
    "xgboost_paper",
    "logistic_regression",
    "ft_transformer",
    "dcn",
]

REGRESSION_DATASETS = [
    "california_housing",
    "abalone",
    "concrete",
]


def run_single_regression_experiment(
    model_name: str,
    dataset_name: str,
    seed: int,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Train and evaluate a single model on a regression dataset with one seed.

    Args:
        model_name: Identifier for the model.
        dataset_name: Identifier for the regression dataset.
        seed: Random seed for data splitting and initialization.
        config: Full configuration dictionary.

    Returns:
        Dictionary containing regression metrics, training time, and parameter count.
    """
    if config is None:
        config = load_config()

    set_seed(seed)
    logger.debug(f"Starting regression: {model_name} on {dataset_name} (seed {seed})")

    # 1. Load data
    processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
    data_path = os.path.join(processed_dir, dataset_name, "data_processed.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed data not found at {data_path}. Run download first."
        )

    df = pd.read_csv(data_path)
    target_column = config["datasets"][dataset_name]["target_column"]

    # 2. Split (70/10/20 non-stratified random split for continuous target)
    splits = create_split(
        df,
        seed=seed,
        train_ratio=config.get("data", {}).get("train_ratio", 0.7),
        val_ratio=config.get("data", {}).get("val_ratio", 0.1),
        test_ratio=config.get("data", {}).get("test_ratio", 0.2),
        target_column=target_column,
        stratify=False,
    )
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    # 3. Preprocess features
    preprocessor = TabularPreprocessor(target_column=target_column)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)
    num_features = preprocessor.num_features

    # 4. Target standardization for neural networks
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-6:
        y_std = 1.0

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    # 5. Model instantiation
    model = create_model(
        model_name=model_name,
        num_features=num_features,
        config=config,
        dataset_name=dataset_name,
        task_type="regression",
    )

    # 6. Fit and evaluate
    t_start = time.time()
    if isinstance(model, TorchBaseModel):
        model.fit(X_train, y_train_norm, X_val, y_val_norm, config)
        y_pred_norm = model.predict(X_test)
        y_pred = (y_pred_norm * y_std) + y_mean
    else:
        model.fit(X_train, y_train, X_val, y_val, config)
        y_pred = model.predict(X_test)

    train_time = time.time() - t_start

    # 7. Compute regression metrics
    metrics = compute_regression_metrics(y_test, y_pred)
    metrics["training_time"] = float(train_time)

    return {
        "metrics": metrics,
        "num_parameters": getattr(model, "num_parameters", 0),
        "training_time": float(train_time),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seed": seed,
    }


def _print_regression_summary_table(
    all_results: Dict[str, Dict[str, Any]],
    dataset_names: List[str],
    metric: str = "rmse",
) -> None:
    """Print a formatted comparison table for regression results to console."""
    model_col_w = max(22, max(len(m) for m in all_results) + 2)
    ds_col_w = 18

    header = f"{'Model':<{model_col_w}}"
    for ds in dataset_names:
        header += f"{ds:>{ds_col_w}}"
    header += f"{'Avg':>{ds_col_w}}"

    unit_label = " (seconds)" if metric == "training_time" else ""
    print("\n" + "=" * len(header))
    print(f"  REGRESSION: {metric.upper()}{unit_label} Comparison Table")
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


def run_regression_experiments(
    model_names: Optional[List[str]] = None,
    dataset_names: Optional[List[str]] = None,
    config: Optional[dict] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run regression benchmark across models, datasets, and seeds.

    Args:
        model_names: List of models to evaluate. Defaults to REGRESSION_MODELS.
        dataset_names: List of regression datasets. Defaults to REGRESSION_DATASETS.
        config: Configuration dictionary.
        results_dir: Base directory where regression results are stored.

    Returns:
        Aggregated results dictionary.
    """
    if model_names is None:
        model_names = REGRESSION_MODELS
    if dataset_names is None:
        dataset_names = REGRESSION_DATASETS
    if config is None:
        config = load_config()

    seeds = config.get("training", {}).get("seeds", [0, 1, 2, 3, 4])

    # Route to isolated run directory (e.g. results/regression/run_1/)
    if results_dir is None or results_dir.rstrip("/\\").endswith("regression"):
        base_dir = (
            results_dir
            if results_dir
            else os.path.join(config.get("results", {}).get("base_dir", "results"), "regression")
        )
        results_dir = get_next_run_dir(base_dir)

    os.makedirs(results_dir, exist_ok=True)
    save_experiment_meta(
        save_path=os.path.join(results_dir, "experiment_meta.json"),
        config=config,
        setting="regression",
        models=model_names,
        datasets=dataset_names,
    )

    total_runs = len(model_names) * len(dataset_names) * len(seeds)
    print("\n" + "=" * 70)
    print("  TABULAR REGRESSION BENCHMARK")
    print("=" * 70)
    print(f"  Models   ({len(model_names)}): {', '.join(model_names)}")
    print(f"  Datasets ({len(dataset_names)}): {', '.join(dataset_names)}")
    print(f"  Seeds    ({len(seeds)}): {seeds}")
    print(f"  Total planned runs: {total_runs}")
    print(f"  Output Directory  : {results_dir}")
    print("=" * 70 + "\n")

    raw_results = {m: {d: [] for d in dataset_names} for m in model_names}
    failed_runs = []

    pbar = tqdm(total=total_runs, desc="Regression Progress", unit="run", dynamic_ncols=True)

    for ds in dataset_names:
        for m in model_names:
            for s in seeds:
                pbar.set_description(f"Running {m} on {ds} (seed={s})")
                try:
                    result = run_single_regression_experiment(
                        model_name=m,
                        dataset_name=ds,
                        seed=s,
                        config=config,
                    )
                    raw_results[m][ds].append(result)
                    rmse = result["metrics"].get("rmse", float("nan"))
                    r2 = result["metrics"].get("r2", float("nan"))
                    pbar.set_postfix(RMSE=f"{rmse:.4f}", R2=f"{r2:.3f}")
                except Exception as e:
                    logger.error(
                        f"Run failed for {m}/{ds}/seed={s}:\n{traceback.format_exc()}"
                    )
                    failed_runs.append(f"{m}/{ds}/seed={s}: {e}")
                finally:
                    pbar.update(1)

    pbar.close()

    # Aggregate across seeds
    aggregated: Dict[str, Dict[str, Any]] = {}
    metric_keys = get_regression_metric_names() + ["training_time"]

    for m in model_names:
        aggregated[m] = {}
        for ds in dataset_names:
            seed_runs = raw_results[m][ds]
            if not seed_runs:
                continue

            num_params = seed_runs[0].get("num_parameters", 0)
            ds_metrics: Dict[str, Any] = {}

            for met in metric_keys:
                vals = [r["metrics"][met] for r in seed_runs if met in r["metrics"] and np.isfinite(r["metrics"][met])]
                if vals:
                    ds_metrics[met] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "values": [float(v) for v in vals],
                    }
                else:
                    ds_metrics[met] = {"mean": float("nan"), "std": 0.0, "values": []}

            aggregated[m][ds] = {
                "metrics": ds_metrics,
                "num_parameters": num_params,
            }

    # Save raw results JSON
    raw_json_path = os.path.join(results_dir, "regression_results_raw.json")
    try:
        with open(raw_json_path, "w") as f:
            json.dump(raw_results, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not save raw results JSON: {e}")

    # Build and export CSV comparison tables
    for met in ["rmse", "r2", "mae", "training_time"]:
        rows = []
        for m in model_names:
            row = {"Model": m}
            for ds in dataset_names:
                stat = aggregated.get(m, {}).get(ds, {}).get("metrics", {}).get(met)
                if stat and np.isfinite(stat["mean"]):
                    row[ds] = f"{stat['mean']:.4f} ± {stat['std']:.4f}"
                else:
                    row[ds] = "N/A"
            rows.append(row)
        df_table = pd.DataFrame(rows)
        csv_path = os.path.join(results_dir, f"comparison_{met}.csv")
        df_table.to_csv(csv_path, index=False)

    # Generate SUMMARY.md
    try:
        summary_path = generate_run_summary_md(
            aggregated_results=aggregated,
            dataset_names=dataset_names,
            setting="regression",
            save_dir=results_dir,
            extra_meta={"seeds": seeds},
        )
        logger.info(f"Summary report written to {summary_path}")
    except Exception as e:
        logger.warning(f"Could not generate SUMMARY.md: {e}")

    # Print summary to console
    _print_regression_summary_table(aggregated, dataset_names, metric="rmse")
    _print_regression_summary_table(aggregated, dataset_names, metric="r2")
    _print_regression_summary_table(aggregated, dataset_names, metric="training_time")

    print(f"📁 Regression results saved to: {results_dir}")
    print(f"📄 Summary report:             {os.path.join(results_dir, 'SUMMARY.md')}\n")

    if failed_runs:
        print(f"\n⚠  {len(failed_runs)} run(s) failed:")
        for fr in failed_runs:
            print(f"   - {fr}")
    else:
        print("All regression runs completed successfully! ✓")

    return aggregated
