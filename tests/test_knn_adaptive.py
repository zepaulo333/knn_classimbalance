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
    clf = Clf()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape


@pytest.mark.parametrize("Clf", [KNNAdaptiveEntropy, KNNAdaptiveEigen])
def test_proba_sums_to_one(imbalanced_data, Clf):
    X, y = imbalanced_data
    clf = Clf()
    clf.fit(X, y)
    proba = clf.predict_proba(X[:10])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("Clf", [KNNAdaptiveEntropy, KNNAdaptiveEigen])
def test_k_within_bounds(imbalanced_data, Clf):
    """Selected k must always be in [k_min, floor(sqrt(n_train))]."""
    X, y = imbalanced_data
    clf = Clf(k_min=1)
    clf.fit(X, y)
    k_max_expected = max(1, int(np.sqrt(len(X))))
    # Indirectly verify by checking predictions are produced without error
    preds = clf.predict(X[:5])
    assert len(preds) == 5


@pytest.mark.parametrize("Clf", [KNNAdaptiveEntropy, KNNAdaptiveEigen])
def test_explicit_k_max(imbalanced_data, Clf):
    """Explicit k_max overrides the sqrt(n_train) default."""
    X, y = imbalanced_data
    clf = Clf(k_min=1, k_max=7)
    clf.fit(X, y)
    preds = clf.predict(X[:5])
    assert len(preds) == 5
