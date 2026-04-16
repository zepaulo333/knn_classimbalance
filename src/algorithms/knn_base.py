"""Standard KNN classifier implemented from scratch.

Follows the sklearn estimator interface (fit / predict / predict_proba)
so it can be swapped interchangeably in the benchmarking pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class KNNClassifier:
    """K-Nearest Neighbours classifier (no scikit-learn in core logic).

    Parameters
    ----------
    k : int
        Number of neighbours to consider.
    metric : str
        Distance metric — ``"euclidean"`` or ``"manhattan"``.
    weights : str
        Voting scheme — ``"uniform"`` or ``"distance"``.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        self.k = k
        self.metric = metric
        self.weights = weights

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        self._X_train = np.asarray(X, dtype=float)
        self._y_train = np.asarray(y)
        self.classes_ = np.unique(self._y_train)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_proba_single(x) for x in X])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _distances(self, x: NDArray) -> NDArray:
        if self.metric == "euclidean":
            diff = self._X_train - x
            return np.sqrt((diff * diff).sum(axis=1))
        if self.metric == "manhattan":
            return np.abs(self._X_train - x).sum(axis=1)
        raise ValueError(f"Unknown metric: {self.metric!r}")

    def _neighbour_indices(self, x: NDArray) -> tuple[NDArray, NDArray]:
        dists = self._distances(x)
        idx = np.argsort(dists)[: self.k]
        return idx, dists[idx]

    def _predict_single(self, x: NDArray):
        idx, dists = self._neighbour_indices(x)
        neighbour_labels = self._y_train[idx]

        if self.weights == "uniform":
            votes = {c: np.sum(neighbour_labels == c) for c in self.classes_}
        else:
            # Distance-weighted: avoid division by zero
            w = 1.0 / (dists + 1e-10)
            votes = {c: w[neighbour_labels == c].sum() for c in self.classes_}

        return max(votes, key=votes.get)

    def _predict_proba_single(self, x: NDArray) -> NDArray:
        idx, dists = self._neighbour_indices(x)
        neighbour_labels = self._y_train[idx]

        if self.weights == "uniform":
            counts = np.array([np.sum(neighbour_labels == c) for c in self.classes_], dtype=float)
            return counts / counts.sum()

        w = 1.0 / (dists + 1e-10)
        counts = np.array([w[neighbour_labels == c].sum() for c in self.classes_], dtype=float)
        return counts / counts.sum()
