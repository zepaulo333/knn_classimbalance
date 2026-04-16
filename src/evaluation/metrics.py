"""Performance metrics for imbalanced binary classification.

All functions accept plain numpy arrays so they work regardless of which
algorithm produced the predictions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def geometric_mean(y_true: NDArray, y_pred: NDArray) -> float:
    """G-mean = sqrt(sensitivity * specificity)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(sensitivity * specificity))


def compute_all_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: NDArray | None = None,
) -> dict[str, float]:
    """Return a dictionary of all relevant metrics for one fold."""
    results: dict[str, float] = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "geometric_mean": geometric_mean(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            results["roc_auc"] = float("nan")
    return results
