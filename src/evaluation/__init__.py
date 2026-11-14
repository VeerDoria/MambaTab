"""Evaluation sub-package: metrics, analysis, and visualization."""

from .metrics import compute_confusion_matrix, compute_metrics, get_metric_names
from .analysis import (
    DATASET_ABBREV,
    PAPER_VANILLA_RESULTS,
    aggregate_seed_results,
    create_comparison_table,
    create_paper_comparison_table,
    create_parameter_comparison_table,
    load_results,
    save_results,
)
from .visualization import (
    generate_all_vanilla_plots,
    plot_auroc_comparison,
    plot_confusion_matrices,
    plot_metric_heatmap,
    plot_multi_metric_radar,
    plot_parameter_comparison,
    plot_training_curves,
)

__all__ = [
    # metrics
    "compute_metrics",
    "compute_confusion_matrix",
    "get_metric_names",
    # analysis
    "aggregate_seed_results",
    "create_comparison_table",
    "create_paper_comparison_table",
    "create_parameter_comparison_table",
    "save_results",
    "load_results",
    "PAPER_VANILLA_RESULTS",
    "DATASET_ABBREV",
    # visualization
    "plot_auroc_comparison",
    "plot_metric_heatmap",
    "plot_multi_metric_radar",
    "plot_parameter_comparison",
    "plot_training_curves",
    "plot_confusion_matrices",
    "generate_all_vanilla_plots",
]
