"""Tests for the standard KNN baseline."""

import numpy as np
import pytest

from src.algorithms.knn_base import KNNClassifier


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_fit_predict_shape(simple_data):
    X, y = simple_data
    clf = KNNClassifier(k=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape


def test_predict_proba_sums_to_one(simple_data):
    X, y = simple_data
    clf = KNNClassifier(k=5)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classes_attribute(simple_data):
    X, y = simple_data
    clf = KNNClassifier(k=3)
    clf.fit(X, y)
    assert set(clf.classes_) == {0, 1}


def test_perfect_separation():
    X_train = np.array([[0.0], [0.1], [10.0], [10.1]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.05], [10.05]])
    clf = KNNClassifier(k=2)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    np.testing.assert_array_equal(preds, [0, 1])
