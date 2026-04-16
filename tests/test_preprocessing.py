"""Tests for preprocessing utilities."""

import numpy as np

from src.data.preprocessing import binarise_labels, remove_constant_features, standardise


def test_standardise_zero_mean():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_train_scaled, _ = standardise(X_train, X_train.copy())
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-10)


def test_standardise_uses_train_stats():
    X_train = np.ones((10, 3)) * 5.0
    X_train[:, 0] = np.arange(10)
    X_test = np.ones((3, 3)) * 100.0
    _, X_test_scaled = standardise(X_train, X_test)
    # test set scaled by train statistics — should differ from zero
    assert not np.allclose(X_test_scaled, 0.0)


def test_binarise_minority_is_one():
    y = np.array([0, 0, 0, 1])  # 1 is minority
    y_bin = binarise_labels(y)
    assert y_bin.sum() == 1  # only one positive


def test_remove_constant_features():
    X = np.array([[1, 5, 3], [1, 6, 4], [1, 7, 5]])
    X_clean = remove_constant_features(X)
    assert X_clean.shape[1] == 2  # column 0 is constant
