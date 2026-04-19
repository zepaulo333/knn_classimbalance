"""Tests for KNNAdaptiveTopo."""

import numpy as np
import pytest

from src.algorithms.knn_adaptive_topo import KNNAdaptiveTopo


@pytest.fixture
def imbalanced_data():
    rng = np.random.default_rng(0)
    X_maj = rng.standard_normal((90, 4))
    X_min = rng.standard_normal((10, 4)) + 3
    X = np.vstack([X_maj, X_min])
    y = np.array([0] * 90 + [1] * 10)
    return X, y


def test_fit_predict_shape(imbalanced_data):
    X, y = imbalanced_data
    clf = KNNAdaptiveTopo()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})


def test_proba_sums_to_one(imbalanced_data):
    X, y = imbalanced_data
    clf = KNNAdaptiveTopo()
    clf.fit(X, y)
    proba = clf.predict_proba(X[:10])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_minority_class_detected(imbalanced_data):
    X, y = imbalanced_data
    clf = KNNAdaptiveTopo()
    clf.fit(X, y)
    assert clf._minority_class == 1  # class 1 has 10 samples vs 90


def test_explicit_k_max(imbalanced_data):
    X, y = imbalanced_data
    clf = KNNAdaptiveTopo(k_min=1, k_max=7)
    clf.fit(X, y)
    preds = clf.predict(X[:5])
    assert len(preds) == 5


def test_degenerate_input():
    """All-identical feature vectors should not crash."""
    X = np.ones((20, 3))
    y = np.array([0] * 18 + [1] * 2)
    clf = KNNAdaptiveTopo()
    clf.fit(X, y)
    preds = clf.predict(X[:5])
    assert len(preds) == 5
