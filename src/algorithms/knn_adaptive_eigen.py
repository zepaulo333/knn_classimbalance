"""KNN with adaptive-k selection driven by local eigenvalue structure.

Extends KNNClassifier (adapted from rushter/MLAlgorithms) with adaptive-k:
the local covariance matrix of the neighbourhood is computed for each
candidate k; the k that yields the most stable local geometry (as
measured by the eigenvalue spectrum) is selected.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import euclidean

from src.algorithms.knn_base import KNNClassifier
from src.utils.config import load_config


class KNNAdaptiveEigen(KNNClassifier):
    """Adaptive-k KNN using local eigenvalue analysis for k selection.

    Parameters
    ----------
    k_range : list[int]
        Candidate values of k to search over.
    distance_func : callable
        Distance function. Defaults to Euclidean.
    eigen_threshold : float
        Cumulative explained variance threshold (0 < threshold <= 1).
    """

    def __init__(
        self,
        k_range: list[int] | None = None,
        distance_func=euclidean,
        eigen_threshold: float = 0.95,
    ) -> None:
        cfg = load_config()["knn_adaptive_eigen"]
        self.k_range = k_range if k_range is not None else cfg["k_range"]
        self.eigen_threshold = eigen_threshold
        super().__init__(k=max(self.k_range), distance_func=distance_func)

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
        """Vectorised Euclidean distances to all training points."""
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
        scores = {}
        for k in self.k_range:
            if k > len(order):
                break
            scores[k] = self._effective_dim(self.X[order[:k]])
        max_dim = max(scores.values())
        for k in self.k_range:
            if k not in scores:
                break
            if scores[k] >= max_dim:
                return k
        return self.k_range[0]
