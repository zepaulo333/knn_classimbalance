"""KNN with adaptive-k via dual class-anchor Mahalanobis soft ratio.

For each query point x, two reference anchors are identified:
  - m : nearest minority-class training sample to x
  - M : nearest majority-class training sample to x

Each anchor defines a local same-class cloud: the anchor plus its
k nearest same-class neighbours. The cloud's covariance captures the
local orientation and spread of that class. x's fit to each class is
measured by the Mahalanobis distance from x to the cloud centroid,
using the cloud's own covariance:

  d_M(x, cloud) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))

This combines distance AND vector structure: distance is weighed by how
the class actually extends in space, so directions aligned with the
class's principal axes are "cheap" and off-manifold directions are
"expensive." It is scale-invariant per class, so a dispersed minority
cluster does not spuriously look closer than a tight majority cluster.

k is set proportionally via a soft ratio of the two distances:

  ratio = d_min / (d_min + d_maj)

  ratio ≈ 0  →  x fits minority structure  →  k near k_min
  ratio ≈ 1  →  x fits majority structure  →  k near k_max
  ratio = 0.5 →  ambiguous boundary         →  k at midpoint

  k = _to_odd_floor(round(k_min + ratio * (k_max - k_min)))

A separate k_min_proba floor is applied to predict_proba to avoid the
degenerate 0/1 probability output that small k produces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import euclidean

from src.algorithms.knn_base import KNNClassifier
from src.utils.config import load_config


def _to_odd_floor(n: int) -> int:
    return max(1, n if n % 2 == 1 else n - 1)


class KNNAdaptiveDualAnchor(KNNClassifier):
    """Adaptive-k KNN using dual class-anchor perturbation analysis.

    Parameters
    ----------
    k_min : int
        k used when x fits minority structure.
    k_max : int or None
        k used when x fits majority structure. None → floor(sqrt(n_train)).
    k_min_proba : int
        Minimum k used in predict_proba to avoid binary 0/1 probabilities.
    distance_func : callable
        Distance function. Defaults to Euclidean.
    """

    def __init__(
        self,
        k_min: int | None = None,
        k_max: int | None = None,
        k_min_proba: int = 3,
        distance_func=euclidean,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_adaptive_dual_anchor", {})
        self._k_min = _to_odd_floor(k_min if k_min is not None else cfg.get("k_min", 1))
        self._k_max_cfg = k_max if k_max is not None else cfg.get("k_max", None)
        self._k_min_proba = max(self._k_min, k_min_proba if k_min_proba is not None else cfg.get("k_min_proba", 3))
        self._n_train: int = 0
        self._minority_class = None
        self._majority_class = None
        self._minority_idx: NDArray | None = None
        self._majority_idx: NDArray | None = None
        super().__init__(k=1, distance_func=distance_func, n_jobs=n_jobs)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNAdaptiveDualAnchor":
        super().fit(X, y)
        self._n_train = len(self.X)
        counts = {c: int(np.sum(self.y == c)) for c in self.classes_}
        self._minority_class = min(counts, key=counts.get)
        self._majority_class = max(counts, key=counts.get)
        self._minority_idx = np.where(self.y == self._minority_class)[0]
        self._majority_idx = np.where(self.y == self._majority_class)[0]
        return self

    def _predict_x(self, x: NDArray):
        order = self._argsort_distances(x)
        k = self._best_k(x)
        return self.aggregate(self.y[order[:k]].tolist())

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        order = self._argsort_distances(x)
        k = max(self._best_k(x), self._k_min_proba)
        neighbors = self.y[order[:k]]
        counts = np.array([np.sum(neighbors == c) for c in self.classes_], dtype=float)
        total = counts.sum()
        return counts / total if total > 0 else counts

    def _best_k(self, x: NDArray) -> int:
        k_max_eff = self._k_max_cfg if self._k_max_cfg is not None else max(1, int(np.sqrt(self._n_train)))
        k_max_eff = _to_odd_floor(k_max_eff)
        k_max_eff = max(self._k_min, k_max_eff)

        d_min = self._class_mahalanobis(x, self._minority_idx, k_max_eff)
        d_maj = self._class_mahalanobis(x, self._majority_idx, k_max_eff)

        # Edge cases: missing class anchors
        if np.isinf(d_min) and np.isinf(d_maj):
            return max(self._k_min, _to_odd_floor(int(np.sqrt(k_max_eff))))
        if np.isinf(d_min):
            return k_max_eff
        if np.isinf(d_maj):
            return self._k_min

        total = d_min + d_maj
        if total < 1e-12:
            return max(self._k_min, _to_odd_floor(int(np.sqrt(k_max_eff))))

        ratio = d_min / total
        k_continuous = self._k_min + ratio * (k_max_eff - self._k_min)
        return _to_odd_floor(round(k_continuous))

    def _class_mahalanobis(self, x: NDArray, class_idx: NDArray, cloud_k: int) -> float:
        """Mahalanobis distance from x to the local same-class cloud.

        The cloud is the nearest same-class point to x (the anchor) plus
        its cloud_k nearest same-class neighbours. The cloud's covariance
        whitens the space, so the distance is measured in units of the
        class's local standard deviations along each principal axis.
        """
        n_class = len(class_idx)
        if n_class == 0:
            return np.inf

        diff_x = self.X[class_idx] - x
        dists_x = np.sqrt((diff_x * diff_x).sum(axis=1))

        # Anchor = nearest same-class point to x
        nearest_local = int(np.argmin(dists_x))

        cloud_size = min(cloud_k, n_class)
        if cloud_size < 2:
            # Single-point cloud — no covariance, fall back to raw Euclidean
            return float(dists_x[nearest_local])

        anchor = self.X[class_idx[nearest_local]]
        diff_a = self.X[class_idx] - anchor
        dists_a = np.sqrt((diff_a * diff_a).sum(axis=1))
        cloud_local = np.argpartition(dists_a, cloud_size - 1)[:cloud_size]
        cloud = self.X[class_idx[cloud_local]]

        mu = cloud.mean(axis=0)
        centred = cloud - mu
        cov = centred.T @ centred / max(1, cloud_size - 1)

        # Ridge-regularise to handle singular / rank-deficient covariance
        # (common when cloud_k < n_features). Epsilon scales with trace so
        # the regularisation stays proportional to the cloud's spread.
        d = cov.shape[0]
        trace = float(np.trace(cov))
        eps = 1e-6 * (trace / d if trace > 0 else 1.0)
        cov_reg = cov + eps * np.eye(d)

        # Mahalanobis via eigendecomposition: more numerically stable than inv.
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        eigvals = np.maximum(eigvals, 1e-12)

        delta = x - mu
        proj = eigvecs.T @ delta
        return float(np.sqrt(np.sum((proj * proj) / eigvals)))

    def _argsort_distances(self, x: NDArray) -> NDArray:
        diff = self.X - x
        return np.argsort(np.sqrt((diff * diff).sum(axis=1)))
