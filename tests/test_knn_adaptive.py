"""Tests for adaptive-k KNN variants (entropy and eigenvalue)."""

import numpy as np
import pytest

from src.algorithms.knn_adaptive_entropy import KNNAdaptiveEntropy
from src.algorithms.knn_adaptive_eigen import KNNAdaptiveEigen


@pytest.fixture
def imbalanced_data():
    rng = np.random.default_rng(0)
    X_maj = rng.standard_normal((90, 4))
    X_min = rng.standard_normal((10, 4)) + 3
    X = np.vstack([X_maj, X_min])
    y = np.array([0] * 90 + [1] * 10)
    return X, y


@pytest.mark.parametrize("Clf", [KNNAdaptiveEntropy, KNNAdaptiveEigen])
def test_fit_predict(imbalanced_data, Clf):
    X, y = imbalanced_data
    clf = Clf(k_range=[3, 5, 7])
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape


@pytest.mark.parametrize("Clf", [KNNAdaptiveEntropy, KNNAdaptiveEigen])
def test_proba_sums_to_one(imbalanced_data, Clf):
    X, y = imbalanced_data
    clf = Clf(k_range=[3, 5])
    clf.fit(X, y)
    proba = clf.predict_proba(X[:10])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
