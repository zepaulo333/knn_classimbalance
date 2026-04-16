"""KNN base classes.

Core algorithm adapted from rushter/MLAlgorithms:
  https://github.com/rushter/MLAlgorithms/blob/master/mla/knn.py

Modifications over the original:
- sklearn-compatible fit/predict/predict_proba interface
- classes_ attribute set on fit
- predict_proba support (not in original)
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist, euclidean


class KNNBase:
    """Base class for KNN variants.

    Core neighbor-finding logic adapted from rushter/MLAlgorithms.

    Parameters
    ----------
    k : int
        Number of neighbours. ``0`` means use all training examples.
    distance_func : callable
        A distance function ``f(a, b) -> float``. Any function from
        ``scipy.spatial.distance`` will do. Defaults to Euclidean.
    """

    def __init__(self, k: int = 5, distance_func=euclidean) -> None:
        self.k = None if k == 0 else k
        self.distance_func = distance_func

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNBase":
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.classes_ = np.unique(self.y)
        return self

    def aggregate(self, neighbors_targets):
        raise NotImplementedError()

    def predict(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_x(x) for x in X])

    def predict_proba(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_proba_x(x) for x in X])

    def _predict_x(self, x: NDArray):
        """Predict the label of a single instance.

        Adapted from rushter/MLAlgorithms KNNBase._predict_x.
        """
        distances = (self.distance_func(x, example) for example in self.X)
        neighbors = sorted(
            zip(distances, self.y),
            key=lambda item: item[0],
        )
        neighbors_targets = [target for (_, target) in neighbors[: self.k]]
        return self.aggregate(neighbors_targets)

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        distances = (self.distance_func(x, example) for example in self.X)
        neighbors = sorted(
            zip(distances, self.y),
            key=lambda item: item[0],
        )
        neighbors_targets = np.array([target for (_, target) in neighbors[: self.k]])
        counts = np.array(
            [np.sum(neighbors_targets == c) for c in self.classes_], dtype=float
        )
        total = counts.sum()
        return counts / total if total > 0 else counts


class KNNClassifier(KNNBase):
    """K-Nearest Neighbours classifier.

    Adapted from rushter/MLAlgorithms KNNClassifier.
    Ties are broken arbitrarily.
    """

    def aggregate(self, neighbors_targets):
        """Return the most common target label."""
        return Counter(neighbors_targets).most_common(1)[0][0]


class KNNClassifierFast(KNNClassifier):
    """KNN classifier with vectorised distance computation via scipy.cdist.

    Replaces the per-example ``scipy.euclidean`` loop from
    ``KNNBase._predict_x`` with a single ``scipy.cdist`` call, which
    computes all distances in one C-level operation using the same
    algorithm as ``scipy.euclidean`` — so predictions are bit-for-bit
    identical to ``KNNClassifier``, only execution speed differs.

    This is our first modification of the rushter/MLAlgorithms base code.
    """

    def _predict_x(self, x: NDArray):
        dists = cdist(self.X, x.reshape(1, -1), metric='euclidean').ravel()
        idx = np.argsort(dists)[: self.k]
        return self.aggregate(self.y[idx].tolist())

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        dists = cdist(self.X, x.reshape(1, -1), metric='euclidean').ravel()
        idx = np.argsort(dists)[: self.k]
        neighbors = self.y[idx]
        counts = np.array([np.sum(neighbors == c) for c in self.classes_], dtype=float)
        total = counts.sum()
        return counts / total if total > 0 else counts
