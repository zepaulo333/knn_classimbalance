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


def _to_odd_floor(n: int) -> int:
    """Largest odd integer <= n (minimum 1)."""
    return max(1, n if n % 2 == 1 else n - 1)


class DANNAdaptive(DANN):
    """DANN classifier with adaptive neighbourhood size.

    k is selected per query point by a halve/double hill-climb starting at
    floor(sqrt(n_train)), always keeping k odd.  O(log n_train) criterion
    evaluations per prediction instead of scanning a fixed candidate list.

    Parameters
    ----------
    k_min : int
        Smallest k to consider (floored to nearest odd if even).
    k_max : int or None
        Largest k to consider.  ``None`` (default) uses floor(sqrt(n_train)).
    sigma : float
        Regularisation parameter for DANN local metric.
    adaptation_strategy : str
        ``"entropy"`` or ``"eigen"`` — the criterion used to select k.
    """

    def __init__(
        self,
        k_min: int | None = None,
        k_max: int | None = None,
        sigma: float = 1.0,
        adaptation_strategy: str = "entropy",
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config()["dann_adaptive"]
        self._k_min = _to_odd_floor(k_min if k_min is not None else cfg.get("k_min", 1))
        self._k_max_cfg = k_max if k_max is not None else cfg.get("k_max", None)
        self.adaptation_strategy = adaptation_strategy
        self._n_train: int = 0
        super().__init__(k=1, sigma=sigma, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # fit — capture training size so k_max can default to sqrt(n_train)
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DANNAdaptive":
        super().fit(X, y)
        self._n_train = len(self._X_train)
        return self

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

    def _score(self, sorted_labels: NDArray, sorted_X: NDArray, k: int) -> float:
        if self.adaptation_strategy == "entropy":
            return self._entropy(sorted_labels[:k])
        return self._effective_dim(sorted_X[:k])

    def _best_k(self, sorted_labels: NDArray, sorted_X: NDArray) -> int:
        n_avail = len(sorted_labels)
        k_max_eff = self._k_max_cfg if self._k_max_cfg is not None else max(1, int(np.sqrt(self._n_train)))
        k_max_eff = min(_to_odd_floor(k_max_eff), n_avail)
        k_max_eff = max(self._k_min, k_max_eff)

        # Starting point: sqrt(n_train), odd, clipped to [k_min, k_max]
        k_start = _to_odd_floor(max(1, int(np.sqrt(self._n_train))))
        k = max(self._k_min, min(k_max_eff, k_start))

        best_k = k
        best_score = self._score(sorted_labels, sorted_X, k)

        # Halve downward from k_start while improving
        curr = k
        while curr > self._k_min:
            nxt = max(self._k_min, _to_odd_floor(curr // 2))
            if nxt >= curr:
                break
            score = self._score(sorted_labels, sorted_X, nxt)
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
            score = self._score(sorted_labels, sorted_X, nxt)
            if score > best_score:
                best_score, best_k, curr = score, nxt, nxt
            else:
                break

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
