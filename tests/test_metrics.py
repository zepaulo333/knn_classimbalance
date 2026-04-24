"""Tests for evaluation metrics."""

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

from src.evaluation.metrics import compute_all_metrics, geometric_mean


# ── geometric_mean (standalone — used by KNNFairRankCV) ──────────────────────

def test_geometric_mean_perfect():
    y = np.array([0, 0, 1, 1])
    assert geometric_mean(y, y) == pytest.approx(1.0)


def test_geometric_mean_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    assert geometric_mean(y_true, y_pred) == pytest.approx(0.0)


# ── compute_all_metrics: output schema ───────────────────────────────────────

def test_compute_all_metrics_schema_no_proba():
    y = np.array([0, 0, 1, 1, 0, 1])
    metrics = compute_all_metrics(y, y)
    assert set(metrics.keys()) == {"tp", "tn", "fp", "fn", "roc_auc", "pr_auc"}


def test_compute_all_metrics_schema_with_proba():
    y = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = compute_all_metrics(y, y, proba)
    assert set(metrics.keys()) == {"tp", "tn", "fp", "fn", "roc_auc", "pr_auc"}


# ── compute_all_metrics: confusion matrix counts ──────────────────────────────

def test_counts_perfect_predictor():
    y = np.array([0, 0, 0, 1, 1, 1])
    m = compute_all_metrics(y, y)
    assert m["tp"] == 3
    assert m["tn"] == 3
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_counts_mixed_predictions():
    # y_pred [0, 1, 0, 1] vs y_true [0, 0, 1, 1] → TP=1 TN=1 FP=1 FN=1
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    m = compute_all_metrics(y_true, y_pred)
    assert m["tp"] == 1
    assert m["tn"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1


def test_counts_all_predicted_majority():
    # Predicts all 0 → no TP, all positives become FN
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    m = compute_all_metrics(y_true, y_pred)
    assert m["tp"] == 0
    assert m["tn"] == 3
    assert m["fp"] == 0
    assert m["fn"] == 2


def test_counts_all_predicted_minority():
    # Predicts all 1 → no TN, all negatives become FP
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1])
    m = compute_all_metrics(y_true, y_pred)
    assert m["tp"] == 2
    assert m["tn"] == 0
    assert m["fp"] == 3
    assert m["fn"] == 0


def test_counts_are_integer_typed():
    y = np.array([0, 0, 1, 1])
    m = compute_all_metrics(y, y)
    for key in ("tp", "tn", "fp", "fn"):
        assert isinstance(m[key], (int, np.integer)), f"{key} should be int"


# ── compute_all_metrics: probability scores ───────────────────────────────────

def test_no_proba_gives_nan_for_auc_metrics():
    y = np.array([0, 0, 1, 1])
    m = compute_all_metrics(y, y)
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["pr_auc"])


def test_with_proba_roc_auc_in_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    m = compute_all_metrics(y, y, proba)
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_with_proba_pr_auc_in_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    m = compute_all_metrics(y, y, proba)
    assert 0.0 <= m["pr_auc"] <= 1.0


def test_with_proba_perfect_scores():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_all_metrics(y, y, proba)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)


def test_with_proba_matches_sklearn():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    proba  = np.array([0.1, 0.6, 0.9, 0.8, 0.3, 0.4, 0.2, 0.7])
    m = compute_all_metrics(y_true, y_pred, proba)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y_true, proba))
    assert m["pr_auc"]  == pytest.approx(average_precision_score(y_true, proba))


def test_single_class_roc_auc_gives_nan():
    # roc_auc_score raises ValueError when only one class is present → caught → nan
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    proba  = np.array([0.1, 0.2, 0.3, 0.4])
    m = compute_all_metrics(y_true, y_pred, proba)
    assert np.isnan(m["roc_auc"])


def test_no_positive_class_pr_auc_is_zero():
    # average_precision_score warns but returns 0.0 (not nan) when no positives exist
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    proba  = np.array([0.1, 0.2, 0.3, 0.4])
    m = compute_all_metrics(y_true, y_pred, proba)
    assert m["pr_auc"] == pytest.approx(0.0)


# ── MCC formula validation (used in notebook _derive_metrics) ─────────────────
# These tests verify that the pandas/numpy formula in the notebook produces the
# same result as sklearn's reference implementation for a range of cases.

def _notebook_mcc(tp, tn, fp, fn):
    """The exact formula from the notebook's _derive_metrics cell."""
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if mcc_den == 0 else mcc_num / mcc_den


@pytest.mark.parametrize("y_true, y_pred", [
    (np.array([0, 0, 1, 1]),         np.array([0, 0, 1, 1])),   # perfect
    (np.array([0, 0, 1, 1]),         np.array([0, 1, 0, 1])),   # 50/50
    (np.array([0, 0, 1, 1]),         np.array([1, 1, 0, 0])),   # all wrong
    (np.array([0, 0, 0, 1, 1, 1]),   np.array([0, 0, 1, 1, 1, 0])),  # mixed
    (np.array([0, 0, 0, 0, 1]),      np.array([0, 0, 0, 0, 0])),     # always majority
])
def test_notebook_mcc_matches_sklearn(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    expected = matthews_corrcoef(y_true, y_pred)
    assert _notebook_mcc(tp, tn, fp, fn) == pytest.approx(expected, abs=1e-9)


def test_notebook_mcc_zero_denominator_returns_zero():
    # All samples predicted as majority → denominator = 0 → MCC defined as 0
    assert _notebook_mcc(tp=0, tn=3, fp=0, fn=2) == pytest.approx(0.0)
