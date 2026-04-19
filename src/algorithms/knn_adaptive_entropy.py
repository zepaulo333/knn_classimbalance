"""KNN with adaptive-k selection driven by local class entropy.

Extends KNNClassifier (adapted from rushter/MLAlgorithms) with adaptive-k:
for each query point, k is chosen dynamically by a halve/double search
starting at sqrt(n_train) to maximise the Shannon entropy of the class
distribution among neighbours — biasing the decision boundary toward the
minority class in imbalanced settings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import euclidean

from src.algorithms.knn_base import KNNClassifier
from src.utils.config import load_config


def _to_odd_floor(n: int) -> int:
    """Largest odd integer <= n (minimum 1)."""
    return max(1, n if n % 2 == 1 else n - 1)


class KNNAdaptiveEntropy(KNNClassifier):
    """Adaptive-k KNN using local Shannon entropy as the selection criterion.

    k is selected per query point by a halve/double hill-climb starting at
    floor(sqrt(n_train)), always keeping k odd.  O(log n_train) criterion
    evaluations per prediction instead of scanning a fixed candidate list.

    Parameters
    ----------
    k_min : int
        Smallest k to consider (must be a positive odd integer; 1 if even).
    k_max : int or None
        Largest k to consider.  ``None`` (default) uses floor(sqrt(n_train)).
    distance_func : callable
        Distance function. Defaults to Euclidean.
    smoothing : float
        Small constant added to probabilities before computing entropy.
    """

    def __init__(
        self,
        k_min: int | None = None,
        k_max: int | None = None,
        distance_func=euclidean,
        smoothing: float = 1e-9,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config()["knn_adaptive_entropy"]
        self._k_min = _to_odd_floor(k_min if k_min is not None else cfg.get("k_min", 1))
        self._k_max_cfg = k_max if k_max is not None else cfg.get("k_max", None)
        self.smoothing = smoothing
        self._n_train: int = 0
        super().__init__(k=1, distance_func=distance_func, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # fit — capture training size so k_max can default to sqrt(n_train)
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNAdaptiveEntropy":
        super().fit(X, y)
        self._n_train = len(self.X)
        return self

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
        n_avail = len(sorted_labels)
        k_max_eff = self._k_max_cfg if self._k_max_cfg is not None else max(1, int(np.sqrt(self._n_train)))
        k_max_eff = min(_to_odd_floor(k_max_eff), n_avail)
        k_max_eff = max(self._k_min, k_max_eff)

        # Starting point: sqrt(n_train), odd, clipped to [k_min, k_max]
        k_start = _to_odd_floor(max(1, int(np.sqrt(self._n_train))))
        k = max(self._k_min, min(k_max_eff, k_start))

        best_k = k
        best_h = self._entropy(sorted_labels[:k])

        # Halve downward from k_start while improving
        curr = k
        while curr > self._k_min:
            nxt = max(self._k_min, _to_odd_floor(curr // 2))
            if nxt >= curr:
                break
            h = self._entropy(sorted_labels[:nxt])
            if h > best_h:
                best_h, best_k, curr = h, nxt, nxt
            else:
                break

        # Double upward from k_start while improving
        curr = k
        while curr < k_max_eff:
            nxt = min(k_max_eff, _to_odd_floor(curr * 2))
            if nxt <= curr:
                break
            h = self._entropy(sorted_labels[:nxt])
            if h > best_h:
                best_h, best_k, curr = h, nxt, nxt
            else:
                break

        return best_k
