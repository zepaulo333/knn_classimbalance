"""Performance metrics for imbalanced binary classification.

All functions accept plain numpy arrays so they work regardless of which
algorithm produced the predictions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def geometric_mean(y_true: NDArray, y_pred: NDArray) -> float:
    """G-mean = sqrt(sensitivity * specificity). Used internally by KNNFairRankCV."""
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
    """Return raw confusion matrix counts plus probability-based scores.

    All threshold-based metrics (MCC, F1, G-mean, etc.) are derived from
    these counts in the analysis layer — adding a new metric never requires
    re-running the benchmark.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results: dict[str, float] = {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    if y_proba is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            results["roc_auc"] = float("nan")
        try:
            results["pr_auc"] = average_precision_score(y_true, y_proba)
        except ValueError:
            results["pr_auc"] = float("nan")
    else:
        results["roc_auc"] = float("nan")
        results["pr_auc"] = float("nan")
    return results
