"""
Comprehensive metrics computation for MambaTab replication.

Computes all classification metrics used in the paper evaluation,
including AUROC (primary metric), AUPRC, and standard binary classification metrics.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """Return confusion matrix components.

    Args:
        y_true: Ground truth binary labels (0/1), shape (n_samples,).
        y_prob: Predicted probabilities for positive class, shape (n_samples,).
        threshold: Decision threshold for converting probabilities to predictions.

    Returns:
        Dictionary with keys 'tn', 'fp', 'fn', 'tp'.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute all classification metrics.

    Args:
        y_true: Ground truth binary labels (0/1), shape (n_samples,).
        y_prob: Predicted probabilities for positive class, shape (n_samples,).
        threshold: Decision threshold for converting probabilities to predictions.

    Returns:
        Dictionary with metric names as keys and values as floats.
        Includes: accuracy, balanced_accuracy, precision, recall, f1_score,
        auroc, auprc, log_loss, specificity, and confusion_matrix (as sub-dict).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # --- Standard classification metrics ---
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))

    # --- Probability-based metrics ---
    # Handle edge case: if y_prob has no variance (all same value), AUROC/AUPRC
    # are undefined — return NaN.
    if np.std(y_prob) == 0 or len(np.unique(y_true)) < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))

    # Clip probabilities to avoid log(0)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    metrics["log_loss"] = float(log_loss(y_true, y_prob_clipped))

    # --- Specificity (true negative rate) ---
    cm = compute_confusion_matrix(y_true, y_prob, threshold)
    tn, fp = cm["tn"], cm["fp"]
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # --- Confusion matrix ---
    metrics["confusion_matrix"] = cm

    return metrics


def get_metric_names() -> List[str]:
    """Return list of all metric names (excluding confusion_matrix).

    These correspond to the scalar metrics computed by :func:`compute_metrics`.
    """
    return [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "log_loss",
        "specificity",
    ]


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute all standard regression evaluation metrics.

    Args:
        y_true: Ground truth continuous targets, shape (n_samples,).
        y_pred: Predicted continuous values, shape (n_samples,).

    Returns:
        Dictionary with: rmse, mae, r2, mse, mape, pearson_r.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # Filter non-finite values if any
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(mask):
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "mse": float("nan"),
            "mape": float("nan"),
            "pearson_r": float("nan"),
        }

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R^2 score
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    if ss_tot > 1e-12:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = 0.0

    # MAPE (avoiding division by zero)
    non_zero = np.abs(y_true) > 1e-8
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
    else:
        mape = float("nan")

    # Pearson correlation r
    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        corr_matrix = np.corrcoef(y_true, y_pred)
        pearson_r = float(corr_matrix[0, 1])
    else:
        pearson_r = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mse": mse,
        "mape": mape,
        "pearson_r": pearson_r,
    }


def get_regression_metric_names() -> List[str]:
    """Return list of regression metric names."""
    return ["rmse", "mae", "r2", "mse", "mape", "pearson_r"]
