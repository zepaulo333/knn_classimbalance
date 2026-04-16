"""KNN with adaptive-k selection driven by local class entropy.

For each query point the neighbourhood size k is chosen from a candidate
set so that the Shannon entropy of the class distribution among neighbours
is maximised (most informative neighbourhood).  This biases the decision
boundary toward the minority class in imbalanced settings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.utils.config import load_config


class KNNAdaptiveEntropy:
    """Adaptive-k KNN using local Shannon entropy as the selection criterion.

    Parameters
    ----------
    k_range : list[int]
        Candidate values of k to search over.
    metric : str
        Distance metric — ``"euclidean"`` or ``"manhattan"``.
    smoothing : float
        Small constant added to probabilities before computing entropy.
    """

    def __init__(
        self,
        k_range: list[int] | None = None,
        metric: str = "euclidean",
        smoothing: float = 1e-9,
    ) -> None:
        cfg = load_config()["knn_adaptive_entropy"]
        self.k_range = k_range if k_range is not None else cfg["k_range"]
        self.metric = metric
        self.smoothing = smoothing

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNAdaptiveEntropy":
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

    def _entropy(self, labels: NDArray) -> float:
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum() + self.smoothing
        p /= p.sum()
        return -float(np.sum(p * np.log2(p)))

    def _best_k(self, sorted_labels: NDArray) -> int:
        best_k, best_h = self.k_range[0], -1.0
        for k in self.k_range:
            if k > len(sorted_labels):
                break
            h = self._entropy(sorted_labels[:k])
            if h > best_h:
                best_h, best_k = h, k
        return best_k

    def _predict_single(self, x: NDArray):
        dists = self._distances(x)
        order = np.argsort(dists)
        sorted_labels = self._y_train[order]
        k = self._best_k(sorted_labels)
        neighbour_labels = sorted_labels[:k]
        votes = {c: np.sum(neighbour_labels == c) for c in self.classes_}
        return max(votes, key=votes.get)

    def _predict_proba_single(self, x: NDArray) -> NDArray:
        dists = self._distances(x)
        order = np.argsort(dists)
        sorted_labels = self._y_train[order]
        k = self._best_k(sorted_labels)
        neighbour_labels = sorted_labels[:k]
        counts = np.array([np.sum(neighbour_labels == c) for c in self.classes_], dtype=float)
        return counts / counts.sum()
