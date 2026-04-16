"""KNN with adaptive-k selection driven by local eigenvalue structure.

The local covariance matrix of the k-neighbourhood is computed for each
candidate k. The *effective dimensionality* of the neighbourhood is
estimated from the eigenvalue spectrum (fraction of variance explained).
The k that yields the most stable / informative local geometry is chosen.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.utils.config import load_config


class KNNAdaptiveEigen:
    """Adaptive-k KNN using local eigenvalue analysis for k selection.

    Parameters
    ----------
    k_range : list[int]
        Candidate values of k to search over.
    metric : str
        Distance metric — ``"euclidean"`` or ``"manhattan"``.
    eigen_threshold : float
        Cumulative explained variance threshold used to estimate effective
        dimensionality (0 < threshold <= 1).
    """

    def __init__(
        self,
        k_range: list[int] | None = None,
        metric: str = "euclidean",
        eigen_threshold: float = 0.95,
    ) -> None:
        cfg = load_config()["knn_adaptive_eigen"]
        self.k_range = k_range if k_range is not None else cfg["k_range"]
        self.metric = metric
        self.eigen_threshold = eigen_threshold

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNAdaptiveEigen":
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

    def _effective_dim(self, neighbours: NDArray) -> float:
        """Estimate effective dimensionality via eigenvalue spectrum."""
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
        """Choose k that maximises local effective dimensionality stability."""
        scores = {}
        for k in self.k_range:
            if k > len(order):
                break
            neighbours = self._X_train[order[:k]]
            scores[k] = self._effective_dim(neighbours)
        # Pick k where the effective dim is closest to its maximum (stable)
        max_dim = max(scores.values())
        for k in self.k_range:
            if k not in scores:
                break
            if scores[k] >= max_dim:
                return k
        return self.k_range[0]

    def _predict_single(self, x: NDArray):
        dists = self._distances(x)
        order = np.argsort(dists)
        k = self._best_k(order)
        neighbour_labels = self._y_train[order[:k]]
        votes = {c: np.sum(neighbour_labels == c) for c in self.classes_}
        return max(votes, key=votes.get)

    def _predict_proba_single(self, x: NDArray) -> NDArray:
        dists = self._distances(x)
        order = np.argsort(dists)
        k = self._best_k(order)
        neighbour_labels = self._y_train[order[:k]]
        counts = np.array([np.sum(neighbour_labels == c) for c in self.classes_], dtype=float)
        return counts / counts.sum()
