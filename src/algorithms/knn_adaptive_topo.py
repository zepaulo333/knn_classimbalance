"""KNN with adaptive-k selection driven by persistent homology (TDA).

Extends KNNClassifier with adaptive-k: for each query point, the local
topology of its neighbourhood (H0 connected components + H1 loops) is used
to classify the query into one of four structural cases, each mapped to a
different k strategy:

  Clean majority  → large k (k_max)         safe, stable region
  Clean minority  → small k (k_min)          avoid bleeding into majority mass
  Boundary        → small k (k_min)          majority vote dilutes minority signal
  Isolated outlier→ medium k (sqrt(k_max))   balance overfit vs dilution

Topology is computed with ripser (Vietoris-Rips filtration, H0 and H1).
All thresholds are expressed as fractions of the neighbourhood diameter so
they transfer across datasets without rescaling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import euclidean
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

from src.algorithms.knn_base import KNNClassifier
from src.utils.config import load_config


def _to_odd_floor(n: int) -> int:
    """Largest odd integer <= n (minimum 1)."""
    return max(1, n if n % 2 == 1 else n - 1)


class KNNAdaptiveTopo(KNNClassifier):
    """Adaptive-k KNN using persistent homology for case classification.

    Parameters
    ----------
    k_min : int
        Smallest k used in minority/boundary/outlier cases.
    k_max : int or None
        Neighbourhood pool size for topology analysis and the k used in clean
        majority regions.  ``None`` → floor(sqrt(n_train)) at fit time.
    h0_threshold : float
        Fraction of neighbourhood diameter above which the longest H0 bar
        signals sub-cluster separation (gap between clouds).
    h1_threshold : float
        Fraction of neighbourhood diameter above which total H1 persistence
        signals a ring/boundary structure.
    minority_threshold : float
        Label fraction below which the neighbourhood is considered majority-
        dominated (used to distinguish cases after topology is classified).
    distance_func : callable
        Distance function (Euclidean by default).
    """

    def __init__(
        self,
        k_min: int | None = None,
        k_max: int | None = None,
        h0_threshold: float | None = None,
        h1_threshold: float | None = None,
        minority_threshold: float | None = None,
        distance_func=euclidean,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config()["knn_adaptive_topo"]
        self._k_min = _to_odd_floor(k_min if k_min is not None else cfg.get("k_min", 1))
        self._k_max_cfg = k_max if k_max is not None else cfg.get("k_max", None)
        self._h0_threshold = h0_threshold if h0_threshold is not None else cfg.get("h0_threshold", 0.5)
        self._h1_threshold = h1_threshold if h1_threshold is not None else cfg.get("h1_threshold", 0.1)
        self._minority_threshold = minority_threshold if minority_threshold is not None else cfg.get("minority_threshold", 0.15)
        self._n_train: int = 0
        self._minority_class = None
        super().__init__(k=1, distance_func=distance_func, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # fit — capture training size and identify minority class
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNAdaptiveTopo":
        super().fit(X, y)
        self._n_train = len(self.X)
        counts = {c: np.sum(self.y == c) for c in self.classes_}
        self._minority_class = min(counts, key=counts.get)
        return self

    # ------------------------------------------------------------------
    # Overridden prediction
    # ------------------------------------------------------------------

    def _predict_x(self, x: NDArray):
        order = self._argsort_distances(x)
        sorted_labels = self.y[order]
        k = self._best_k(order, sorted_labels)
        return self.aggregate(sorted_labels[:k].tolist())

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        order = self._argsort_distances(x)
        sorted_labels = self.y[order]
        k = self._best_k(order, sorted_labels)
        neighbors = sorted_labels[:k]
        counts = np.array([np.sum(neighbors == c) for c in self.classes_], dtype=float)
        total = counts.sum()
        return counts / total if total > 0 else counts

    # ------------------------------------------------------------------
    # Core: topology-driven k selection
    # ------------------------------------------------------------------

    def _best_k(self, order: NDArray, sorted_labels: NDArray) -> int:
        n_avail = len(order)
        k_max_eff = self._k_max_cfg if self._k_max_cfg is not None else max(1, int(np.sqrt(self._n_train)))
        k_max_eff = min(_to_odd_floor(k_max_eff), n_avail)
        k_max_eff = max(self._k_min, k_max_eff)
        k_outlier = max(self._k_min, _to_odd_floor(int(np.sqrt(k_max_eff))))

        pool = self.X[order[:k_max_eff]]
        labels_pool = sorted_labels[:k_max_eff]

        # Pass a precomputed distance matrix so ripser never sees more columns
        # than rows (which happens when n_features > k_max_eff and triggers a
        # spurious transpose warning).
        try:
            dist_matrix = squareform(pdist(pool, metric="euclidean"))
            result = ripser(dist_matrix, maxdim=1, distance_matrix=True)
        except Exception:
            # Degenerate input: fall back to small k
            return self._k_min

        dgms = result["dgms"]

        # ── H0 analysis ──────────────────────────────────────────────
        h0 = dgms[0]
        h0_finite = h0[np.isfinite(h0[:, 1])]  # drop infinite bar

        if len(h0_finite) == 0:
            return self._k_min

        # Total diameter ≈ max finite H0 death (= last merge distance)
        total_diam = float(h0_finite[:, 1].max())
        if total_diam < 1e-10:
            return self._k_min

        h0_pers = h0_finite[:, 1] - h0_finite[:, 0]  # birth is always 0 for H0
        h0_max = float(h0_pers.max())
        h0_ratio = h0_max / total_diam

        # ── H1 analysis ──────────────────────────────────────────────
        h1_ratio = 0.0
        if len(dgms) > 1:
            h1 = dgms[1]
            h1_finite = h1[np.isfinite(h1[:, 1])]
            if len(h1_finite) > 0:
                h1_total = float((h1_finite[:, 1] - h1_finite[:, 0]).sum())
                h1_ratio = h1_total / total_diam

        # ── Label distribution ────────────────────────────────────────
        min_fraction = float(np.mean(labels_pool == self._minority_class))

        # ── Case classification ───────────────────────────────────────
        has_subclusters = h0_ratio > self._h0_threshold
        is_boundary = h1_ratio > self._h1_threshold

        if is_boundary or (has_subclusters and self._minority_threshold < min_fraction < (1 - self._minority_threshold)):
            # Boundary: interleaved classes, ring structure or clear sub-cluster gap
            return self._k_min

        if has_subclusters and min_fraction < self._minority_threshold:
            # Isolated outlier: one large gap + neighbourhood dominated by majority
            return k_outlier

        if min_fraction > (1 - self._minority_threshold):
            # Clean minority region: tight cluster, mostly minority labels
            return self._k_min

        # Clean majority region: tight cluster, mostly majority labels
        return k_max_eff

    # ------------------------------------------------------------------
    # Distance helper
    # ------------------------------------------------------------------

    def _argsort_distances(self, x: NDArray) -> NDArray:
        diff = self.X - x
        return np.argsort(np.sqrt((diff * diff).sum(axis=1)))
