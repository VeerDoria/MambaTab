"""
Results aggregation, comparison tables, and persistence for MambaTab replication.

Provides utilities to aggregate metrics across seeds, build comparison tables
(including against the original paper's Table II), and save/load results.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ============================================================
# Paper results — hardcoded from Table II (vanilla setting)
# ============================================================
# Dataset abbreviations:
#   adult -> AD, income -> IC, blastchar -> BL,
#   credit_g -> CG, cylinder_bands -> CB,
#   california -> CA, diabetes_support -> DS, ionosphere -> IO
PAPER_VANILLA_RESULTS = {
    "MambaTab-D": {
        "CG": 0.771,
        "CA": 0.954,
        "DS": 0.643,
        "AD": 0.906,
        "CB": 0.862,
        "BL": 0.852,
        "IO": 0.785,
        "IC": 0.906,
    },
    "MambaTab-T": {
        "CG": 0.801,
        "CA": 0.963,
        "DS": 0.681,
        "AD": 0.914,
        "CB": 0.896,
        "BL": 0.854,
        "IO": 0.812,
        "IC": 0.920,
    },
    "LR": {
        "CG": 0.720,
        "CA": 0.836,
        "DS": 0.557,
        "AD": 0.851,
        "CB": 0.748,
        "BL": 0.801,
        "IO": 0.769,
        "IC": 0.860,
    },
    "XGBoost": {
        "CG": 0.726,
        "CA": 0.895,
        "DS": 0.587,
        "AD": 0.912,
        "CB": 0.892,
        "BL": 0.821,
        "IO": 0.758,
        "IC": 0.925,
    },
    "FT-Trans": {
        "CG": 0.739,
        "CA": 0.859,
        "DS": 0.657,
        "AD": 0.913,
        "CB": 0.862,
        "BL": 0.841,
        "IO": 0.793,
        "IC": 0.915,
    },
    "DCN": {
        "CG": 0.739,
        "CA": 0.870,
        "DS": 0.674,
        "AD": 0.913,
        "CB": 0.848,
        "BL": 0.840,
        "IO": 0.768,
        "IC": 0.915,
    },
}

# ============================================================
# Paper results — hardcoded from Table III (Feature Incremental Learning)
# ============================================================
PAPER_FIL_RESULTS = {
    "MambaTab-D": {
        "CG": 0.787,
        "CA": 0.961,
        "DS": 0.669,
        "AD": 0.904,
        "CB": 0.860,
        "BL": 0.853,
        "IO": 0.783,
        "IC": 0.908,
    },
    "TransTab": {
        "CG": 0.741,
        "CA": 0.879,
        "DS": 0.665,
        "AD": 0.894,
        "CB": 0.791,
        "BL": 0.841,
        "IO": 0.739,
        "IC": 0.897,
    },
    "LR": {
        "CG": 0.670,
        "CA": 0.773,
        "DS": 0.475,
        "AD": 0.832,
        "CB": 0.727,
        "BL": 0.806,
        "IO": 0.655,
        "IC": 0.825,
    },
    "XGBoost": {
        "CG": 0.608,
        "CA": 0.817,
        "DS": 0.527,
        "AD": 0.891,
        "CB": 0.778,
        "BL": 0.816,
        "IO": 0.692,
        "IC": 0.898,
    },
    "FT-Trans": {
        "CG": 0.662,
        "CA": 0.824,
        "DS": 0.626,
        "AD": 0.892,
        "CB": 0.768,
        "BL": 0.840,
        "IO": 0.645,
        "IC": 0.889,
    },
    "DCN": {
        "CG": 0.674,
        "CA": 0.835,
        "DS": 0.578,
        "AD": 0.893,
        "CB": 0.778,
        "BL": 0.840,
        "IO": 0.660,
        "IC": 0.891,
    },
}

# Mapping from our full dataset names to paper abbreviations.
DATASET_ABBREV = {
    "adult": "AD",
    "income": "IC",
    "blastchar": "BL",
    "credit_g": "CG",
    "cylinder_bands": "CB",
    "credit_approval": "CA",
    "dresses_sales": "DS",
    "insurance_co": "IO",
}

# Reverse mapping: abbreviation -> full name.
ABBREV_DATASET = {v: k for k, v in DATASET_ABBREV.items()}


# ============================================================
# Aggregation
# ============================================================


def aggregate_seed_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple seeds.

    Args:
        results: List of experiment result dictionaries or metric dictionaries (one per seed).

    Returns:
        Dict mapping metric_name -> {'mean': float, 'std': float}.
    """
    if not results:
        return {}

    # Unpack nested 'metrics' dict if present
    unpacked_list = []
    for r in results:
        if isinstance(r, dict) and "metrics" in r and isinstance(r["metrics"], dict):
            m = dict(r["metrics"])
            if "training_time" in r:
                m["training_time"] = r["training_time"]
            if "num_parameters" in r:
                m["num_parameters"] = r["num_parameters"]
            unpacked_list.append(m)
        elif isinstance(r, dict):
            unpacked_list.append(r)

    if not unpacked_list:
        return {}

    # Collect scalar metrics only (skip nested dicts like confusion_matrix).
    first_item = unpacked_list[0]
    metric_names = [k for k in first_item if isinstance(first_item[k], (int, float))]

    aggregated: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        values = [r[name] for r in unpacked_list if name in r and isinstance(r[name], (int, float))]
        if values:
            aggregated[name] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }
    return aggregated


# ============================================================
# Comparison tables
# ============================================================


def create_comparison_table(
    all_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metric: str = "auroc",
    format: str = "dataframe",
) -> pd.DataFrame:
    """Create a comparison table across models and datasets.

    Args:
        all_results: Nested dict:
            model_name -> dataset_name -> metric_name -> {'mean': ..., 'std': ...}
        metric: Which metric to display.
        format: 'dataframe' returns raw floats;
                'latex' or 'markdown' formats values as ``mean ± std``.

    Returns:
        DataFrame with models as rows and datasets as columns.
    """
    rows = {}
    for model, datasets in all_results.items():
        row = {}
        for dataset, metrics in datasets.items():
            abbrev = DATASET_ABBREV.get(dataset, dataset)
            if metric in metrics:
                mean = metrics[metric]["mean"]
                std = metrics[metric]["std"]
                if format in ("latex", "markdown"):
                    row[abbrev] = f"{mean:.3f} ± {std:.3f}"
                else:
                    row[abbrev] = mean
            else:
                row[abbrev] = np.nan
        rows[model] = row

    df = pd.DataFrame(rows).T
    df.index.name = "Model"

    # Sort columns in a consistent order if possible.
    ordered_cols = [c for c in DATASET_ABBREV.values() if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining]

    if format == "latex":
        print(df.to_latex())
    elif format == "markdown":
        print(df.to_markdown())

    return df


def create_paper_comparison_table(
    our_results: Dict,
    paper_results: Optional[Dict] = None,
    metric: str = "auroc",
) -> pd.DataFrame:
    """Compare our results against the original paper's Table II.

    Args:
        our_results: Nested dict —
            model_name -> dataset_name -> metric_name -> {'mean': ..., 'std': ...}
        paper_results: Paper reference values
            (model_name -> dataset_abbrev -> float).
            Defaults to :data:`PAPER_VANILLA_RESULTS`.
        metric: Metric to compare.

    Returns:
        DataFrame with multi-level rows (model, source) and dataset columns.
    """
    if paper_results is None:
        paper_results = PAPER_VANILLA_RESULTS

    records = []
    processed_models = set()
    for model in paper_results:
        processed_models.add(model)
        # Paper row
        for ds_abbrev, value in paper_results[model].items():
            records.append(
                {
                    "Model": model,
                    "Source": "Paper",
                    "Dataset": ds_abbrev,
                    "Value": f"{value:.3f}",
                }
            )
        # Our row (if available)
        if model in our_results:
            for dataset, metrics in our_results[model].items():
                abbrev = DATASET_ABBREV.get(dataset, dataset)
                if metric in metrics:
                    mean = metrics[metric]["mean"]
                    std = metrics[metric]["std"]
                    records.append(
                        {
                            "Model": model,
                            "Source": "Ours",
                            "Dataset": abbrev,
                            "Value": f"{mean:.3f} ± {std:.3f}",
                        }
                    )

    # Also include any models in our_results that were not in the original paper (e.g. XGBoost-2022, MambaTab-T in FIL)
    for model in our_results:
        if model not in processed_models:
            for dataset, metrics in our_results[model].items():
                abbrev = DATASET_ABBREV.get(dataset, dataset)
                if metric in metrics:
                    mean = metrics[metric]["mean"]
                    std = metrics[metric]["std"]
                    records.append(
                        {
                            "Model": model,
                            "Source": "Ours",
                            "Dataset": abbrev,
                            "Value": f"{mean:.3f} ± {std:.3f}",
                        }
                    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.pivot_table(
        index=["Model", "Source"],
        columns="Dataset",
        values="Value",
        aggfunc="first",
    )

    # Order columns consistently.
    ordered = [c for c in DATASET_ABBREV.values() if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    return df


def create_parameter_comparison_table(
    model_params: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """Create Table V equivalent: parameter counts per model per dataset.

    Args:
        model_params: model_name -> dataset_name -> number_of_learnable_params.

    Returns:
        DataFrame with models as rows and datasets as columns.
    """
    rows = {}
    for model, datasets in model_params.items():
        row = {}
        for dataset, count in datasets.items():
            abbrev = DATASET_ABBREV.get(dataset, dataset)
            row[abbrev] = count
        rows[model] = row

    df = pd.DataFrame(rows).T
    df.index.name = "Model"

    ordered = [c for c in DATASET_ABBREV.values() if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    return df


# ============================================================
# Persistence
# ============================================================


def save_results(
    all_results: Dict,
    save_dir: str,
    setting: str = "vanilla",
) -> None:
    """Save all raw results and aggregated tables to disk.

    Saves two files under ``save_dir``:
    - ``{setting}_results_raw.json``  — the full nested results dict.
    - ``{setting}_results_table.csv`` — the AUROC comparison table.

    Args:
        all_results: model -> dataset -> metric -> {'mean', 'std'}.
        save_dir: Directory to write into (created if needed).
        setting: Experiment setting label (e.g. 'vanilla', 'incremental').
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Raw JSON ---
    raw_path = os.path.join(save_dir, f"{setting}_results_raw.json")
    # Convert numpy types for JSON serialisation.
    serialisable = _make_serialisable(all_results)
    with open(raw_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    # --- AUROC comparison table ---
    try:
        df = create_comparison_table(all_results, metric="auroc", format="dataframe")
        csv_path = os.path.join(save_dir, f"{setting}_results_table.csv")
        df.to_csv(csv_path)
    except Exception:
        pass  # Table generation is best-effort.


def load_results(save_dir: str, setting: str = "vanilla") -> Dict:
    """Load previously saved results.

    Args:
        save_dir: Directory containing saved results.
        setting: Experiment setting label.

    Returns:
        The nested results dictionary.
    """
    raw_path = os.path.join(save_dir, f"{setting}_results_raw.json")
    with open(raw_path, "r") as f:
        return json.load(f)


# ============================================================
# Helpers
# ============================================================


def _make_serialisable(obj):
    """Recursively convert numpy scalars/arrays to Python-native types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
