"""
Publication-quality visualizations for MambaTab replication.

All plotting functions follow a consistent style:
- seaborn 'whitegrid' theme
- Shared colour palette keyed by model name
- 150 DPI for saved figures
- Tight layout
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .analysis import DATASET_ABBREV, PAPER_VANILLA_RESULTS

# ============================================================
# Global style defaults
# ============================================================

# One colour per model — deterministic order so colours stay stable.
MODEL_PALETTE = {
    "MambaTab-D": "#1f77b4",
    "MambaTab-T": "#17becf",
    "LR": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "XGBoost-2022": "#8c564b",
    "FT-Trans": "#d62728",
    "DCN": "#9467bd",
}

# Fallback colour cycle for models not in the palette above.
_EXTRA_COLORS = [
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _model_color(model: str) -> str:
    """Return a deterministic colour for *model*."""
    if model in MODEL_PALETTE:
        return MODEL_PALETTE[model]
    # Assign from the extra pool based on hash so it's stable.
    idx = hash(model) % len(_EXTRA_COLORS)
    return _EXTRA_COLORS[idx]


def _apply_style():
    """Set the shared matplotlib/seaborn style."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
        }
    )


def _save_fig(fig, save_path: str):
    """Save figure and close."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _get_dataset_order(datasets):
    """Return datasets sorted by the canonical paper order."""
    paper_order = list(DATASET_ABBREV.values())
    abbrevs = [DATASET_ABBREV.get(d, d) for d in datasets]
    # Sort: those in the paper order first (by their index), then the rest.
    return sorted(
        abbrevs, key=lambda d: paper_order.index(d) if d in paper_order else 999
    )


# ============================================================
# Plotting functions
# ============================================================


def plot_auroc_comparison(
    all_results: Dict,
    save_path: str,
    paper_results: Optional[Dict] = None,
) -> None:
    """Grouped bar chart comparing AUROC across models and datasets.

    Args:
        all_results: model -> dataset -> metric -> {'mean', 'std'}.
        save_path: File path to save the figure.
        paper_results: If provided, add hatched bars for paper values.
    """
    _apply_style()

    models = list(all_results.keys())
    # Collect all dataset abbreviations.
    all_datasets = set()
    for m in models:
        for ds in all_results[m]:
            all_datasets.add(DATASET_ABBREV.get(ds, ds))
    datasets = _get_dataset_order(all_datasets)

    n_datasets = len(datasets)
    n_models = len(models)
    bar_width = 0.8 / (n_models * (2 if paper_results else 1))
    x = np.arange(n_datasets)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        means, stds = [], []
        for ds in datasets:
            # Look up by abbreviation or full name.
            found = False
            for full_name, metrics in all_results[model].items():
                abbrev = DATASET_ABBREV.get(full_name, full_name)
                if abbrev == ds and "auroc" in metrics:
                    means.append(metrics["auroc"]["mean"])
                    stds.append(metrics["auroc"]["std"])
                    found = True
                    break
            if not found:
                means.append(0)
                stds.append(0)

        offset = i * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            label=model,
            color=_model_color(model),
            capsize=3,
        )

        # Paper reference bars (hatched, slightly offset).
        if paper_results and model in paper_results:
            paper_vals = [paper_results[model].get(ds, 0) for ds in datasets]
            ax.bar(
                x + offset + bar_width,
                paper_vals,
                bar_width,
                label=f"{model} (paper)" if i == 0 or True else None,
                color=_model_color(model),
                alpha=0.35,
                hatch="//",
                edgecolor="black",
            )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC Comparison Across Models and Datasets")
    ax.set_xticks(x + bar_width * n_models / 2)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_metric_heatmap(
    all_results: Dict,
    metric: str,
    save_path: str,
) -> None:
    """Heatmap of a metric across models × datasets.

    Args:
        all_results: model -> dataset -> metric -> {'mean', 'std'}.
        metric: Metric to display.
        save_path: File path to save the figure.
    """
    _apply_style()

    models = list(all_results.keys())
    all_datasets = set()
    for m in models:
        for ds in all_results[m]:
            all_datasets.add(DATASET_ABBREV.get(ds, ds))
    datasets = _get_dataset_order(all_datasets)

    data = np.full((len(models), len(datasets)), np.nan)
    for i, model in enumerate(models):
        for full_name, metrics in all_results[model].items():
            abbrev = DATASET_ABBREV.get(full_name, full_name)
            if abbrev in datasets and metric in metrics:
                j = datasets.index(abbrev)
                data[i, j] = metrics[metric]["mean"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        xticklabels=datasets,
        yticklabels=models,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": metric.upper()},
    )
    ax.set_title(f"{metric.upper()} — Models × Datasets")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Model")
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_multi_metric_radar(
    results_for_dataset: Dict,
    dataset_name: str,
    save_path: str,
) -> None:
    """Radar / spider chart comparing multiple metrics for all models on one dataset.

    Args:
        results_for_dataset: model_name -> metric_name -> {'mean', 'std'}.
        dataset_name: Human-readable dataset name (for the title).
        save_path: File path to save the figure.
    """
    _apply_style()

    # Metrics to include on the radar (skip log_loss — inverted scale).
    radar_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "specificity",
    ]

    models = list(results_for_dataset.keys())
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in models:
        values = []
        for m in radar_metrics:
            if m in results_for_dataset[model]:
                val = results_for_dataset[model][m]
                values.append(val["mean"] if isinstance(val, dict) else float(val))
            else:
                values.append(0)
        values += values[:1]  # close
        ax.plot(angles, values, "o-", label=model, color=_model_color(model), linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=_model_color(model))

    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        [m.replace("_", "\n") for m in radar_metrics],
        fontsize=9,
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"Multi-Metric Comparison — {dataset_name}", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_parameter_comparison(
    model_params: Dict[str, int],
    save_path: str,
) -> None:
    """Bar chart comparing total learnable parameters across models.

    Args:
        model_params: model_name -> total_param_count.
        save_path: File path to save the figure.
    """
    _apply_style()

    models = list(model_params.keys())
    counts = [model_params[m] for m in models]
    colors = [_model_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, counts, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate each bar.
    for bar, count in zip(bars, counts):
        label = f"{count:,}" if count < 1_000_000 else f"{count / 1_000_000:.2f}M"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Number of Parameters")
    ax.set_title("Learnable Parameter Count per Model")
    ax.set_xlabel("Model")
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_training_curves(
    training_history: Dict,
    save_path: str,
) -> None:
    """Plot training loss and validation AUROC curves over epochs.

    Args:
        training_history: Dict with keys 'train_loss' (list of floats)
            and 'val_auroc' (list of floats).
        save_path: File path to save the figure.
    """
    _apply_style()

    train_loss = training_history.get("train_loss", [])
    val_auroc = training_history.get("val_auroc", [])
    epochs = range(1, max(len(train_loss), len(val_auroc)) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_loss = "#d62728"
    color_auroc = "#1f77b4"

    if train_loss:
        ax1.plot(
            epochs[: len(train_loss)],
            train_loss,
            color=color_loss,
            linewidth=2,
            label="Train Loss",
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    if val_auroc:
        ax2.plot(
            epochs[: len(val_auroc)],
            val_auroc,
            color=color_auroc,
            linewidth=2,
            linestyle="--",
            label="Val AUROC",
        )
    ax2.set_ylabel("Validation AUROC", color=color_auroc)
    ax2.tick_params(axis="y", labelcolor=color_auroc)

    # Unified legend.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Training Curves")
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_confusion_matrices(
    confusion_matrices: Dict[str, Dict],
    dataset_name: str,
    save_path: str,
) -> None:
    """Grid of confusion matrices for all models on one dataset.

    Args:
        confusion_matrices: model_name -> {'tn': int, 'fp': int, 'fn': int, 'tp': int}.
        dataset_name: Dataset label for the title.
        save_path: File path to save the figure.
    """
    _apply_style()

    models = list(confusion_matrices.keys())
    n = len(models)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, model in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        cm = confusion_matrices[model]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            cbar=False,
        )
        ax.set_title(model, fontsize=11)

    # Hide unused axes.
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Confusion Matrices — {dataset_name}", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ============================================================
# Convenience: generate all standard plots
# ============================================================


def generate_all_vanilla_plots(
    all_results: Dict,
    save_dir: str,
    paper_results: Optional[Dict] = None,
) -> None:
    """Generate all standard plots for the vanilla-setting results.

    Creates the following figures under *save_dir*:
    - ``auroc_comparison.png``
    - ``auroc_heatmap.png``
    - ``auprc_heatmap.png``
    - ``radar_<dataset>.png`` for each dataset
    - ``confusion_<dataset>.png`` for each dataset (if confusion matrices present)

    Args:
        all_results: model -> dataset -> metric -> {'mean', 'std'}.
        save_dir: Directory to save all figures.
        paper_results: If provided, overlay paper reference on the bar chart.
    """
    os.makedirs(save_dir, exist_ok=True)
    if paper_results is None:
        paper_results = PAPER_VANILLA_RESULTS

    # 1. AUROC grouped bar chart.
    plot_auroc_comparison(
        all_results,
        os.path.join(save_dir, "auroc_comparison.png"),
        paper_results=paper_results,
    )

    # 2. Heatmaps.
    for metric in ("auroc", "auprc"):
        plot_metric_heatmap(
            all_results,
            metric,
            os.path.join(save_dir, f"{metric}_heatmap.png"),
        )

    # 3. Radar charts — one per dataset.
    # Collect all datasets.
    all_datasets = set()
    for model_data in all_results.values():
        all_datasets.update(model_data.keys())

    for dataset in sorted(all_datasets):
        abbrev = DATASET_ABBREV.get(dataset, dataset)
        per_model = {}
        for model, model_data in all_results.items():
            if dataset in model_data:
                per_model[model] = model_data[dataset]
        if per_model:
            plot_multi_metric_radar(
                per_model,
                abbrev,
                os.path.join(save_dir, f"radar_{abbrev}.png"),
            )

    # 4. Confusion matrices — if the raw results contain them.
    for dataset in sorted(all_datasets):
        abbrev = DATASET_ABBREV.get(dataset, dataset)
        cms = {}
        for model, model_data in all_results.items():
            if dataset in model_data:
                m_data = model_data[dataset]
                if "confusion_matrix" in m_data:
                    cms[model] = m_data["confusion_matrix"]
        if cms:
            plot_confusion_matrices(
                cms,
                abbrev,
                os.path.join(save_dir, f"confusion_{abbrev}.png"),
            )


def plot_stage_progression(
    df_progression: Any,
    save_dir: str,
) -> None:
    """Plot AUROC progression across incremental feature stages (Stage 1 -> 2 -> 3).

    Args:
        df_progression: pandas DataFrame with columns ['model', 'dataset', 'stage', 'auroc'].
        save_dir: Directory where progression figures will be saved.
    """
    import pandas as pd
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    if not isinstance(df_progression, pd.DataFrame) or df_progression.empty:
        return

    datasets = df_progression["dataset"].unique()
    for ds in datasets:
        abbrev = DATASET_ABBREV.get(ds, ds)
        ds_df = df_progression[df_progression["dataset"] == ds]
        agg = ds_df.groupby(["model", "stage"])["auroc"].mean().reset_index()
        models = agg["model"].unique()

        fig, ax = plt.subplots(figsize=(8, 5))
        for m in models:
            m_sub = agg[agg["model"] == m].sort_values("stage")
            stages = [f"Stage {st}" for st in m_sub["stage"]]
            ax.plot(
                stages,
                m_sub["auroc"],
                marker="o",
                linewidth=2.5,
                label=m,
                color=_model_color(m),
            )

        ax.set_xlabel("Incremental Feature Stage")
        ax.set_ylabel("Test AUROC")
        ax.set_title(f"FIL Progression (s1 -> s1∪s2 -> All) — {ds.capitalize()} ({abbrev})")
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save_fig(fig, os.path.join(save_dir, f"progression_{abbrev}.png"))

