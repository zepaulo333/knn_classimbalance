"""Tests for DANN and DANNAdaptive."""

import numpy as np
import pytest

from src.algorithms.dann import DANN
from src.algorithms.dann_adaptive import DANNAdaptive


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.mark.parametrize("Clf,kwargs", [
    (DANN, {"k": 5}),
    (DANNAdaptive, {"k_range": [3, 5], "adaptation_strategy": "entropy"}),
    (DANNAdaptive, {"k_range": [3, 5], "adaptation_strategy": "eigen"}),
])
def test_fit_predict_shape(data, Clf, kwargs):
    X, y = data
    clf = Clf(**kwargs)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_dann_proba_sums_to_one(data):
    X, y = data
    clf = DANN(k=3)
    clf.fit(X, y)
    proba = clf.predict_proba(X[:5])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
