"""
Experiment utilities for incremental run directories and clean markdown summaries.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def get_next_run_dir(base_dir: str, prefix: str = "run_") -> str:
    """Find the next incremental run directory (e.g. run_1, run_2, etc.).

    Args:
        base_dir: Base directory (e.g. results/vanilla or results/fil).
        prefix: Prefix for run folders (default: 'run_').

    Returns:
        Path to the newly created run directory.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_nums = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.startswith(prefix):
            suffix = item[len(prefix):]
            if suffix.isdigit():
                existing_nums.append(int(suffix))

    next_num = max(existing_nums) + 1 if existing_nums else 1
    run_dir = os.path.join(base_dir, f"{prefix}{next_num}")
    os.makedirs(run_dir, exist_ok=True)

    # Maintain a 'latest' pointer for convenient inspection
    latest_path = os.path.join(base_dir, "latest")
    try:
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(f"{prefix}{next_num}", latest_path)
    except Exception:
        try:
            with open(latest_path + ".txt", "w") as f:
                f.write(run_dir + "\n")
        except Exception:
            pass

    return run_dir


def generate_run_summary_md(
    aggregated_results: Dict[str, Dict[str, Any]],
    dataset_names: List[str],
    setting: str,
    save_dir: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a clean, publication-quality SUMMARY.md report with formatted metric tables.

    Args:
        aggregated_results: Nested dict of model -> dataset -> {metric -> {mean, std}}.
        dataset_names: List of evaluated dataset names.
        setting: 'vanilla' or 'fil' (Feature-Incremental Learning).
        save_dir: Path to the run folder where SUMMARY.md will be saved.
        extra_meta: Optional extra metadata dictionary.

    Returns:
        Full path to the generated SUMMARY.md file.
    """
    os.makedirs(save_dir, exist_ok=True)
    run_name = os.path.basename(save_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = []
    doc.append(f"# 📊 Experiment Results — {run_name.upper()}\n")
    doc.append(f"- **Setting**: {setting.upper()}")
    doc.append(f"- **Execution Time**: {timestamp}")
    doc.append(f"- **Output Folder**: `{save_dir}`")
    if extra_meta:
        if "seeds" in extra_meta:
            doc.append(f"- **Random Seeds**: `{extra_meta['seeds']}`")
        if "stages" in extra_meta:
            doc.append(f"- **Incremental Stages**: `{extra_meta['stages']}`")
    doc.append("\n---\n")

    models = list(aggregated_results.keys())

    # Core metrics to generate summary tables for
    core_metrics = [
        ("auroc", "AUROC (Primary Metric)", False),
        ("accuracy", "Accuracy", False),
        ("f1_score", "F1-Score", False),
        ("balanced_accuracy", "Balanced Accuracy", False),
        ("auprc", "AUPRC (Precision-Recall Area)", False),
        ("training_time", "Average Training Time (seconds)", True),
    ]

    for metric_key, metric_title, is_time in core_metrics:
        doc.append(f"## {metric_title}\n")

        # Table Header
        header = f"| Model | " + " | ".join([f"{d}" for d in dataset_names]) + " | Mean |"
        sep = f"| :--- | " + " | ".join([":---:" for _ in dataset_names]) + " | :---: |"
        doc.append(header)
        doc.append(sep)

        table_rows = []
        for m in models:
            cells = []
            means = []
            row_dict = {"Model": m}
            for d in dataset_names:
                m_data = aggregated_results.get(m, {}).get(d, {})
                # unpack metrics subdict if present
                if isinstance(m_data, dict) and "metrics" in m_data:
                    m_data = m_data["metrics"]

                val_stat = m_data.get(metric_key) if isinstance(m_data, dict) else None
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
                    if is_time:
                        formatted = f"{mean:.2f}s ± {std:.2f}"
                    else:
                        formatted = f"{mean:.4f} ± {std:.4f}"
                    cells.append(formatted)
                    row_dict[d] = formatted
                    means.append(mean)
                else:
                    cells.append("-")
                    row_dict[d] = "N/A"

            if means:
                avg = sum(means) / len(means)
                avg_str = f"**{avg:.2f}s**" if is_time else f"**{avg:.4f}**"
                row_dict["Average"] = f"{avg:.2f}s" if is_time else f"{avg:.4f}"
            else:
                avg_str = "-"
                row_dict["Average"] = "N/A"

            doc.append(f"| **{m}** | " + " | ".join(cells) + f" | {avg_str} |")
            table_rows.append(row_dict)

        doc.append("\n---\n")

        # Also export clean CSV for this metric
        try:
            df_metric = pd.DataFrame(table_rows)
            csv_path = os.path.join(save_dir, f"comparison_{metric_key}.csv")
            df_metric.to_csv(csv_path, index=False)
        except Exception:
            pass

    summary_file = os.path.join(save_dir, "SUMMARY.md")
    with open(summary_file, "w") as f:
        f.write("\n".join(doc))

    return summary_file
