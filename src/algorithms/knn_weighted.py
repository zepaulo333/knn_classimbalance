"""KNN with class-frequency weighting (balanced class weights).

Each class's vote count is multiplied by N_total / (n_classes * N_c) before
normalisation — the sklearn "balanced" scheme. For binary imbalance this is
equivalent to weighting minority votes by the imbalance ratio N_maj / N_min.

This is the simplest possible imbalance-aware KNN and serves as a sanity-check
baseline: if FairRank does not outperform this, the rank-correction mechanism
adds nothing beyond the imbalance ratio.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist

from src.algorithms.knn_base import KNNClassifierFast


class KNNWeighted(KNNClassifierFast):
    """KNN with balanced class-frequency weighting."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNWeighted":
        super().fit(X, y)
        counts = np.array([np.sum(self.y == c) for c in self.classes_], dtype=float)
        self._class_weights = len(self.y) / (len(self.classes_) * counts)
        return self

    def _predict_x(self, x: NDArray):
        proba = self._predict_proba_x(x)
        return self.classes_[np.argmax(proba)]

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        dists = cdist(self.X, x.reshape(1, -1), metric="euclidean").ravel()
        idx = np.argsort(dists)[: self.k]
        neighbors = self.y[idx]
        counts = np.array([np.sum(neighbors == c) for c in self.classes_], dtype=float)
        weighted = counts * self._class_weights
        total = weighted.sum()
        return weighted / total if total > 0 else weighted
