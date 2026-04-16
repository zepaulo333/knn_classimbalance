"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_all_metrics, geometric_mean


def test_geometric_mean_perfect():
    y = np.array([0, 0, 1, 1])
    assert geometric_mean(y, y) == pytest.approx(1.0)


def test_geometric_mean_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    assert geometric_mean(y_true, y_pred) == pytest.approx(0.0)


def test_compute_all_metrics_keys():
    y = np.array([0, 0, 1, 1, 0, 1])
    metrics = compute_all_metrics(y, y)
    assert "f1" in metrics
    assert "balanced_accuracy" in metrics
    assert "geometric_mean" in metrics


def test_compute_all_metrics_with_proba():
    y = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = compute_all_metrics(y, y, proba)
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
