"""Preprocessing utilities: scaling, encoding, and label binarisation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def standardise(X_train: NDArray, X_test: NDArray) -> tuple[NDArray, NDArray]:
    """Z-score normalise using training statistics only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant features
    return (X_train - mean) / std, (X_test - mean) / std


def binarise_labels(y: NDArray) -> NDArray:
    """Map arbitrary binary labels to {0, 1}.

    The minority class (smaller count) is mapped to 1 (positive class).
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Expected exactly 2 classes, got {len(classes)}.")
    minority = classes[np.argmin(counts)]
    return (y == minority).astype(int)


def remove_constant_features(X: NDArray) -> NDArray:
    """Drop features with zero variance."""
    mask = X.std(axis=0) > 0
    return X[:, mask]
