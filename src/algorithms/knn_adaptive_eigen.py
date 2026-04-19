"""KNN with adaptive-k selection driven by local eigenvalue structure.

Extends KNNClassifier (adapted from rushter/MLAlgorithms) with adaptive-k:
the local covariance matrix of the neighbourhood is computed for each
candidate k; the k that yields the most stable local geometry (as
measured by the eigenvalue spectrum) is selected via a halve/double
hill-climb starting at sqrt(n_train).
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


class KNNAdaptiveEigen(KNNClassifier):
    """Adaptive-k KNN using local eigenvalue analysis for k selection.

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
    eigen_threshold : float
        Cumulative explained variance threshold (0 < threshold <= 1).
    """

    def __init__(
        self,
        k_min: int | None = None,
        k_max: int | None = None,
        distance_func=euclidean,
        eigen_threshold: float = 0.95,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config()["knn_adaptive_eigen"]
        self._k_min = _to_odd_floor(k_min if k_min is not None else cfg.get("k_min", 1))
        self._k_max_cfg = k_max if k_max is not None else cfg.get("k_max", None)
        self.eigen_threshold = eigen_threshold
        self._n_train: int = 0
        super().__init__(k=1, distance_func=distance_func, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # fit — capture training size so k_max can default to sqrt(n_train)
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNAdaptiveEigen":
        super().fit(X, y)
        self._n_train = len(self.X)
        return self

    # ------------------------------------------------------------------
    # Overridden prediction — adaptive k replaces fixed k
    # ------------------------------------------------------------------

    def _predict_x(self, x: NDArray):
        order = self._argsort_distances(x)
        k = self._best_k(order)
        return self.aggregate(self.y[order[:k]].tolist())

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        order = self._argsort_distances(x)
        k = self._best_k(order)
        neighbors = self.y[order[:k]]
        counts = np.array([np.sum(neighbors == c) for c in self.classes_], dtype=float)
        total = counts.sum()
        return counts / total if total > 0 else counts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _argsort_distances(self, x: NDArray) -> NDArray:
        diff = self.X - x
        return np.argsort(np.sqrt((diff * diff).sum(axis=1)))

    def _effective_dim(self, neighbours: NDArray) -> float:
        if neighbours.shape[0] < 2:
            return 1.0
        centred = neighbours - neighbours.mean(axis=0)
        cov = centred.T @ centred / (neighbours.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        eigvals = np.maximum(eigvals, 0.0)
        total = eigvals.sum()
        if total == 0:
            return 1.0
        cumvar = np.cumsum(eigvals) / total
        n_components = int(np.searchsorted(cumvar, self.eigen_threshold)) + 1
        return float(n_components)

    def _best_k(self, order: NDArray) -> int:
        n_avail = len(order)
        k_max_eff = self._k_max_cfg if self._k_max_cfg is not None else max(1, int(np.sqrt(self._n_train)))
        k_max_eff = min(_to_odd_floor(k_max_eff), n_avail)
        k_max_eff = max(self._k_min, k_max_eff)

        # Starting point: sqrt(n_train), odd, clipped to [k_min, k_max]
        k_start = _to_odd_floor(max(1, int(np.sqrt(self._n_train))))
        k = max(self._k_min, min(k_max_eff, k_start))

        best_k = k
        best_score = self._effective_dim(self.X[order[:k]])

        # Halve downward from k_start while improving
        curr = k
        while curr > self._k_min:
            nxt = max(self._k_min, _to_odd_floor(curr // 2))
            if nxt >= curr:
                break
            score = self._effective_dim(self.X[order[:nxt]])
            if score > best_score:
                best_score, best_k, curr = score, nxt, nxt
            else:
                break

        # Double upward from k_start while improving
        curr = k
        while curr < k_max_eff:
            nxt = min(k_max_eff, _to_odd_floor(curr * 2))
            if nxt <= curr:
                break
            score = self._effective_dim(self.X[order[:nxt]])
            if score > best_score:
                best_score, best_k, curr = score, nxt, nxt
            else:
                break

        return best_k
