"""
Feature-Incremental Learning (FIL) experiment runner and orchestrator.

Replicates Table III from the MambaTab paper (Section IV-C):
- Divides features into 3 stages: s1, s1 ∪ s2, s1 ∪ s2 ∪ s3.
- MambaTab dynamically expands its input representation layer while
  preserving learned Mamba blocks and classification parameters.
- Baselines are evaluated under progressive feature unmasking.
- Tracks metrics across stages and evaluates on the final test set.
"""

import copy
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.incremental import get_feature_subsets, slice_stage_features
from src.data.preprocess import TabularPreprocessor
from src.data.splits import create_split
from src.evaluation.analysis import (
    DATASET_ABBREV,
    PAPER_FIL_RESULTS,
    aggregate_seed_results,
    create_comparison_table,
    create_paper_comparison_table,
    save_results,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualization import generate_all_vanilla_plots, plot_stage_progression
from src.models.factory import create_model
from src.utils.config import load_config
from src.utils.logging_utils import save_experiment_meta, setup_logger
from src.utils.reproducibility import get_device, set_seed

logger = setup_logger("training.incremental")

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


def run_single_incremental_experiment(
    model_name: str,
    dataset_name: str,
    seed: int,
    config: dict,
    n_stages: int = 3,
    preserve_weights: bool = True,
) -> Dict[str, Any]:
    """Run a single model on a dataset across incremental feature stages.

    Args:
        model_name: Model identifier ('mambatab_d', 'mambatab_t', 'xgboost', etc.).
        dataset_name: Dataset identifier ('adult', 'income', etc.).
        seed: Random seed.
        config: Loaded experiment configuration dictionary.
        n_stages: Number of incremental stages (default: 3).
        preserve_weights: Whether MambaTab preserves weights when expanding input dim.

    Returns:
        Dict with final metrics, stage progression, parameters, and timing.
    """
    set_seed(seed)
    device = get_device()

    # 1. Load data
    processed_dir = config["data"]["processed_dir"]
    csv_path = os.path.join(processed_dir, dataset_name, "data_processed.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Processed dataset not found: {csv_path}. Please download/preprocess first."
        )
    df = pd.read_csv(csv_path)

    # 2. Split (70/10/20 stratified)
    target_column = config["datasets"][dataset_name]["target_column"]
    splits = create_split(
        df,
        seed=seed,
        train_ratio=config.get("data", {}).get("train_ratio", 0.7),
        val_ratio=config.get("data", {}).get("val_ratio", 0.1),
        test_ratio=config.get("data", {}).get("test_ratio", 0.2),
        target_column=target_column,
        stratify=True,
    )
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    # 3. Preprocess
    preprocessor = TabularPreprocessor(target_column=target_column)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)
    total_features = preprocessor.num_features

    # 4. Partition features into stages
    boundaries = get_feature_subsets(total_features, n_stages=n_stages)
    logger.info(
        f"[{model_name}/{dataset_name}/seed={seed}] Total features: {total_features}, "
        f"Stage boundaries: {boundaries}"
    )

    is_dynamic_model = model_name in ("mambatab_d", "mambatab_t")
    is_paper_baseline = model_name in ("xgboost_2022", "xgboost_paper")
    model = None
    stage_progressions = []
    total_training_time = 0.0

    # 5. Incremental training through each stage
    for stage_idx, end_col in enumerate(boundaries):
        stage_num = stage_idx + 1
        logger.info(
            f"--- Stage {stage_num}/{n_stages}: using {end_col}/{total_features} features ---"
        )

        t_stage_start = time.time()
        if is_dynamic_model:
            # Sliced feature representation
            X_tr = slice_stage_features(X_train, end_col, zero_pad=False)
            X_va = slice_stage_features(X_val, end_col, zero_pad=False)
            X_te = slice_stage_features(X_test, end_col, zero_pad=False)

            if model is None:
                # Initialize model for stage 1
                model = create_model(model_name, end_col, config, dataset_name=dataset_name)
            else:
                # Dynamically expand input representation layer
                model.expand_input_features(end_col, preserve_weights=preserve_weights)
                logger.info(
                    f"Expanded {model_name} input layer to {end_col} features "
                    f"(preserve_weights={preserve_weights})"
                )

            model.fit(X_tr, y_train, X_va, y_val, config)

        elif is_paper_baseline:
            # 2022 Paper Baseline Setting: Model only trains on initial common feature set s1
            # Dropping subsequent incrementally added features as specified in Section IV-C
            s1_col = boundaries[0]
            X_tr = slice_stage_features(X_train, s1_col, zero_pad=False)
            X_va = slice_stage_features(X_val, s1_col, zero_pad=False)
            X_te = slice_stage_features(X_test, s1_col, zero_pad=False)

            if model is None:
                model = create_model(model_name, s1_col, config, dataset_name=dataset_name)
                model.fit(X_tr, y_train, X_va, y_val, config)

        else:
            # Baseline models: progressive feature unmasking (zero-padded)
            X_tr = slice_stage_features(X_train, end_col, zero_pad=True, total_features=total_features)
            X_va = slice_stage_features(X_val, end_col, zero_pad=True, total_features=total_features)
            X_te = slice_stage_features(X_test, end_col, zero_pad=True, total_features=total_features)

            if model is None:
                model = create_model(model_name, total_features, config, dataset_name=dataset_name)

            model.fit(X_tr, y_train, X_va, y_val, config)

        stage_time = time.time() - t_stage_start
        total_training_time += stage_time

        # Evaluate stage performance on current test representation
        stage_probs = model.predict_proba(X_te)
        stage_metrics = compute_metrics(y_test, stage_probs)
        logger.info(
            f"Stage {stage_num} test AUROC: {stage_metrics.get('auroc', 0.0):.4f} "
            f"(trained in {stage_time:.2f}s)"
        )
        stage_progressions.append({
            "stage": stage_num,
            "active_features": end_col,
            "total_features": total_features,
            "auroc": float(stage_metrics.get("auroc", 0.0)),
            "accuracy": float(stage_metrics.get("accuracy", 0.0)),
            "f1_score": float(stage_metrics.get("f1_score", 0.0)),
            "auprc": float(stage_metrics.get("auprc", 0.0)),
        })

    # 6. Final test set evaluation on 100% full features
    if is_dynamic_model:
        y_prob = model.predict_proba(X_test)
    elif is_paper_baseline:
        y_prob = model.predict_proba(slice_stage_features(X_test, boundaries[0], zero_pad=False))
    else:
        y_prob = model.predict_proba(X_test)

    final_metrics = compute_metrics(y_test, y_prob, threshold=0.5)

    return {
        "metrics": final_metrics,
        "stage_progression": stage_progressions,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seed": seed,
        "num_parameters": getattr(model, "num_parameters", 0),
        "training_time": total_training_time,
    }


def run_incremental_experiments(
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    config_path: str = "configs/default.yaml",
    save_dir: str = "results/fil",
    n_stages: int = 3,
) -> Dict[str, Any]:
    """Run full Feature-Incremental Learning (FIL) benchmark.

    Args:
        models: List of model names to run (defaults to ALL_MODELS).
        datasets: List of dataset names to run (defaults to ALL_DATASETS).
        seeds: List of seeds (defaults to config seeds).
        config_path: Path to configuration YAML.
        save_dir: Directory where FIL tables, charts, and raw JSONs are stored.
        n_stages: Number of incremental stages.

    Returns:
        Aggregated results dictionary.
    """
    config = load_config(config_path)
    models = models or ALL_MODELS
    datasets = datasets or ALL_DATASETS
    if seeds is None:
        seeds = config.get("training", {}).get("seeds", [0, 1, 2, 3, 4])

    preserve_weights = config.get("incremental", {}).get("preserve_weights", True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    total_runs = len(models) * len(datasets) * len(seeds)
    print("\n" + "=" * 70)
    print("  FEATURE-INCREMENTAL LEARNING (FIL) BENCHMARK")
    print("  Replication of Table III — MambaTab Paper (Section IV-C)")
    print("=" * 70)
    print(f"  Models   ({len(models)}): {', '.join(models)}")
    print(f"  Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"  Seeds    ({len(seeds)}): {seeds}")
    print(f"  Stages   : {n_stages} (s1 -> s1∪s2 -> s1∪s2∪s3)")
    print(f"  Total planned runs: {total_runs}")
    print("=" * 70 + "\n")

    raw_results_path = os.path.join(save_dir, "raw_results.json")
    if os.path.exists(raw_results_path):
        try:
            with open(raw_results_path, "r") as f:
                raw_results = json.load(f)
        except Exception:
            raw_results = {}
    else:
        raw_results = {}

    for m in models:
        if m not in raw_results:
            raw_results[m] = {}
        for d in datasets:
            raw_results[m][d] = []

    stage_prog_records = []
    run_count = 0
    failed_runs = []

    try:
        from tqdm.auto import tqdm
    except ImportError:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

    run_list = [
        (dataset_name, model_name, seed)
        for dataset_name in datasets
        for model_name in models
        for seed in seeds
    ]

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            run_list,
            total=total_runs,
            desc="FIL Benchmark",
            unit="run",
            dynamic_ncols=True,
        )
        iterable = pbar
    else:
        iterable = run_list

    for d_idx, (dataset_name, model_name, seed) in enumerate(iterable, 1):
        run_count += 1
        prefix = f"[{run_count}/{total_runs}] {model_name} on {dataset_name} (seed={seed})"
        if pbar is not None:
            pbar.set_description(f"FIL: {model_name[:12]} on {dataset_name} (s={seed})")
        else:
            print(f"{prefix} ...", end=" ", flush=True)

        try:
            res = run_single_incremental_experiment(
                model_name=model_name,
                dataset_name=dataset_name,
                seed=seed,
                config=config,
                n_stages=n_stages,
                preserve_weights=preserve_weights,
            )
            raw_results[model_name][dataset_name].append(res)
            auroc_val = res["metrics"].get("auroc", 0.0)
            acc_val = res["metrics"].get("accuracy", 0.0)
            t_val = res.get("training_time", 0.0)
            if pbar is not None:
                pbar.set_postfix({
                    "last": f"{model_name[:8]}/{dataset_name[:6]}",
                    "time": f"{t_val:.2f}s",
                    "auroc": f"{auroc_val:.4f}",
                })
            else:
                print(f"AUROC: {auroc_val:.4f} | Acc: {acc_val:.4f} ({t_val:.2f}s) ✓")

            # Record stage progression for plotting
            for st in res.get("stage_progression", []):
                stage_prog_records.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "seed": seed,
                    "stage": st["stage"],
                    "active_features": st["active_features"],
                    "auroc": st["auroc"],
                    "accuracy": st["accuracy"],
                })

        except Exception as e:
            if pbar is None:
                print(f"FAILED ✗: {e}")
            failed_runs.append(f"{model_name}/{dataset_name}/seed={seed}: {e}")
            logger.error(f"Error in {prefix}:\n{traceback.format_exc()}")

    # Aggregate across seeds
    aggregated_results: Dict[str, Dict[str, Any]] = {
        m: {} for m in models
    }
    for m in models:
        for d in datasets:
            seed_runs = raw_results[m][d]
            if seed_runs:
                aggregated_results[m][d] = aggregate_seed_results(seed_runs)

    # 1. Save Raw Results JSON
    with open(os.path.join(save_dir, "raw_results.json"), "w") as f:
        json.dump(raw_results, f, indent=2, default=str)
    with open(os.path.join(save_dir, "fil_results_raw.json"), "w") as f:
        json.dump(aggregated_results, f, indent=2, default=str)

    # 2. Save Stage Progression CSV & Plots
    if stage_prog_records:
        df_prog = pd.DataFrame(stage_prog_records)
        df_prog.to_csv(os.path.join(save_dir, "tables", "stage_progression.csv"), index=False)
        try:
            plot_stage_progression(df_prog, os.path.join(save_dir, "figures"))
        except Exception as e:
            logger.warning(f"Could not plot stage progressions: {e}")

    # 3. Generate Comparison Tables
    for metric in ("auroc", "accuracy", "f1_score", "auprc", "training_time"):
        try:
            df_tab = create_comparison_table(aggregated_results, metric=metric, format="dataframe")
            df_tab.to_csv(os.path.join(save_dir, "tables", f"fil_comparison_{metric}.csv"))
        except Exception as e:
            logger.warning(f"Could not build {metric} comparison table: {e}")

    # 4. Generate Paper Table III Comparison Table
    try:
        model_display_map = {
            "mambatab_d": "MambaTab-D",
            "mambatab_t": "MambaTab-T",
            "xgboost": "XGBoost",
            "xgboost_2022": "XGBoost-2022",
            "logistic_regression": "LR",
            "ft_transformer": "FT-Trans",
            "dcn": "DCN",
        }
        mapped_results = {
            model_display_map.get(m, m): aggregated_results[m] for m in aggregated_results
        }
        df_paper_iii = create_paper_comparison_table(
            our_results=mapped_results,
            paper_results=PAPER_FIL_RESULTS,
            metric="auroc",
        )
        df_paper_iii.to_csv(os.path.join(save_dir, "tables", "paper_comparison_table_iii.csv"))
    except Exception as e:
        logger.warning(f"Could not build Paper Table III comparison: {e}")

    # 5. Generate Visualizations
    try:
        generate_all_vanilla_plots(
            all_results=aggregated_results,
            save_dir=os.path.join(save_dir, "figures"),
            paper_results=PAPER_FIL_RESULTS,
        )
    except Exception as e:
        logger.warning(f"Could not generate visual plots: {e}")

    # Summary Output
    print("\n" + "=" * 70)
    print("  FEATURE-INCREMENTAL LEARNING (FIL) SUMMARY (AUROC)")
    print("=" * 70)
    header = f"{'Model':<22} | " + " | ".join([f"{DATASET_ABBREV.get(d, d):^7}" for d in datasets])
    print(header)
    print("-" * len(header))

    for m in models:
        row_str = f"{m:<22} | "
        vals = []
        for d in datasets:
            m_data = aggregated_results[m].get(d, {}).get("auroc")
            if m_data:
                vals.append(f"{m_data['mean']:.3f}±{m_data['std']:.2f}")
            else:
                vals.append("  N/A  ")
        row_str += " | ".join(vals)
        print(row_str)
    print("=" * 70)

    if failed_runs:
        print(f"\n⚠  {len(failed_runs)} run(s) failed:")
        for fr in failed_runs:
            print(f"   - {fr}")
    else:
        print("\nAll runs completed successfully! ✓")

    return aggregated_results
