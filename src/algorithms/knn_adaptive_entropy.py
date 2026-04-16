"""KNN with adaptive-k selection driven by local class entropy.

Extends KNNClassifier (adapted from rushter/MLAlgorithms) with adaptive-k:
for each query point, k is chosen from a candidate set to maximise the
Shannon entropy of the class distribution among neighbours — biasing the
decision boundary toward the minority class in imbalanced settings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import euclidean

from src.algorithms.knn_base import KNNClassifier
from src.utils.config import load_config


class KNNAdaptiveEntropy(KNNClassifier):
    """Adaptive-k KNN using local Shannon entropy as the selection criterion.

    Parameters
    ----------
    k_range : list[int]
        Candidate values of k to search over.
    distance_func : callable
        Distance function. Defaults to Euclidean.
    smoothing : float
        Small constant added to probabilities before computing entropy.
    """

    def __init__(
        self,
        k_range: list[int] | None = None,
        distance_func=euclidean,
        smoothing: float = 1e-9,
    ) -> None:
        cfg = load_config()["knn_adaptive_entropy"]
        self.k_range = k_range if k_range is not None else cfg["k_range"]
        self.smoothing = smoothing
        super().__init__(k=max(self.k_range), distance_func=distance_func)

    # ------------------------------------------------------------------
    # Overridden prediction — adaptive k replaces fixed k
    # ------------------------------------------------------------------

    def _predict_x(self, x: NDArray):
        order = self._argsort_distances(x)
        sorted_labels = self.y[order]
        k = self._best_k(sorted_labels)
        return self.aggregate(sorted_labels[:k].tolist())

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        order = self._argsort_distances(x)
        sorted_labels = self.y[order]
        k = self._best_k(sorted_labels)
        neighbors = sorted_labels[:k]
        counts = np.array([np.sum(neighbors == c) for c in self.classes_], dtype=float)
        total = counts.sum()
        return counts / total if total > 0 else counts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _argsort_distances(self, x: NDArray) -> NDArray:
        """Vectorised Euclidean distances to all training points."""
        diff = self.X - x
        return np.argsort(np.sqrt((diff * diff).sum(axis=1)))

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
