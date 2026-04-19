"""KNN with adaptive-k via dual class-anchor perturbation analysis.

For each query point x, two reference anchors are identified:
  - m : nearest minority-class training sample to x
  - M : nearest majority-class training sample to x

Each anchor's local class-pure neighbourhood is studied independently:
  - m's neighbourhood contains only minority samples
  - M's neighbourhood contains only majority samples

The disruption score measures how well x fits into each anchor's structure,
defined as the normalised distance from x to the anchor relative to the
anchor's neighbourhood radius (distance to its k-th same-class neighbour):

  disruption_min = d(x, m) / r_m
  disruption_maj = d(x, M) / r_M

  disruption_min < disruption_maj  →  x fits minority structure  →  k_min
  disruption_maj < disruption_min  →  x fits majority structure  →  k_max
  tie / ambiguous                  →  sqrt(k_max)  (boundary)

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
        k_boundary = max(self._k_min, _to_odd_floor(int(np.sqrt(k_max_eff))))

        disruption_min = self._disruption(x, self._minority_idx, k_max_eff)
        disruption_maj = self._disruption(x, self._majority_idx, k_max_eff)

        if disruption_min < disruption_maj:
            return self._k_min
        if disruption_maj < disruption_min:
            return k_max_eff
        return k_boundary

    def _disruption(self, x: NDArray, class_idx: NDArray, anchor_k: int) -> float:
        """Normalised distance from x to its nearest same-class anchor,
        relative to that anchor's k-th same-class neighbour distance."""
        if len(class_idx) == 0:
            return np.inf

        # Distance from x to every sample in this class
        diff_x = self.X[class_idx] - x
        dists_x = np.sqrt((diff_x * diff_x).sum(axis=1))

        nearest_local = int(np.argmin(dists_x))
        d_x_anchor = float(dists_x[nearest_local])

        # Neighbourhood radius of the anchor: distance to its k-th same-class neighbour
        anchor = self.X[class_idx[nearest_local]]
        diff_a = self.X[class_idx] - anchor
        dists_a = np.sqrt((diff_a * diff_a).sum(axis=1))
        dists_a[nearest_local] = np.inf  # exclude self

        k_eff = min(anchor_k, len(class_idx) - 1)
        if k_eff <= 0:
            # Only one sample in this class — use raw distance as disruption
            return d_x_anchor

        r_anchor = float(np.partition(dists_a, k_eff - 1)[k_eff - 1])
        return d_x_anchor / (r_anchor + 1e-10)

    def _argsort_distances(self, x: NDArray) -> NDArray:
        diff = self.X - x
        return np.argsort(np.sqrt((diff * diff).sum(axis=1)))
