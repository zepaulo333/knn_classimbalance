"""DANN with adaptive-k selection.

Extends the baseline DANN by replacing the fixed-k neighbourhood with the
adaptive-k strategy (entropy or eigenvalue based) from the KNN variants.
This is the main proposed contribution of the assignment.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.dann import DANN
from src.utils.config import load_config


class DANNAdaptive(DANN):
    """DANN classifier with adaptive neighbourhood size.

    Parameters
    ----------
    k_range : list[int]
        Candidate values of k to search over.
    sigma : float
        Regularisation parameter for DANN local metric.
    adaptation_strategy : str
        ``"entropy"`` or ``"eigen"`` — the criterion used to select k.
    """

    def __init__(
        self,
        k_range: list[int] | None = None,
        sigma: float = 1.0,
        adaptation_strategy: str = "entropy",
    ) -> None:
        cfg = load_config()["dann_adaptive"]
        self.k_range = k_range if k_range is not None else cfg["k_range"]
        self.adaptation_strategy = adaptation_strategy
        # Initialise DANN with k=max so _local_metric works correctly;
        # actual k used during prediction is chosen adaptively per point.
        super().__init__(k=max(self.k_range), sigma=sigma)

    # ------------------------------------------------------------------
    # k selection strategies
    # ------------------------------------------------------------------

    def _entropy(self, labels: NDArray, smoothing: float = 1e-9) -> float:
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum() + smoothing
        p /= p.sum()
        return -float(np.sum(p * np.log2(p)))

    def _effective_dim(self, neighbours: NDArray, threshold: float = 0.95) -> float:
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
        return float(np.searchsorted(cumvar, threshold)) + 1.0

    def _best_k(self, sorted_labels: NDArray, sorted_X: NDArray) -> int:
        best_k, best_score = self.k_range[0], -np.inf
        for k in self.k_range:
            if k > len(sorted_labels):
                break
            if self.adaptation_strategy == "entropy":
                score = self._entropy(sorted_labels[:k])
            else:
                score = self._effective_dim(sorted_X[:k])
            if score > best_score:
                best_score, best_k = score, k
        return best_k

    # ------------------------------------------------------------------
    # Override prediction helpers
    # ------------------------------------------------------------------

    def _predict_single(self, x: NDArray):
        dists = self._dann_distances(x)
        order = np.argsort(dists)
        sorted_labels = self._y_train[order]
        sorted_X = self._X_train[order]
        k = self._best_k(sorted_labels, sorted_X)
        neighbour_labels = sorted_labels[:k]
        votes = {c: np.sum(neighbour_labels == c) for c in self.classes_}
        return max(votes, key=votes.get)

    def _predict_proba_single(self, x: NDArray) -> NDArray:
        dists = self._dann_distances(x)
        order = np.argsort(dists)
        sorted_labels = self._y_train[order]
        sorted_X = self._X_train[order]
        k = self._best_k(sorted_labels, sorted_X)
        neighbour_labels = sorted_labels[:k]
        counts = np.array([np.sum(neighbour_labels == c) for c in self.classes_], dtype=float)
        return counts / counts.sum()
