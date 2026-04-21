"""KNN with statistically fair rank correction for class imbalance.

Under class imbalance, the majority class always produces a closer
rank-1 neighbour than the minority class as a pure sampling artifact:
with N_maj >> N_min, more samples means smaller minimum-distance
statistics even when the underlying distributions are equivalent.

This algorithm corrects that bias by comparing rank-1 minority distance
against rank-k_eff majority distance, where k_eff is derived from order
statistics of Poisson-process nearest-neighbour distances:

    E[d_k^c] ∝ (k / N_c)^(1/d)

Setting E[d_{k_eff}^maj] = E[d_1^min] gives the fair rank

    k_eff = r^(1/d)   with   r = N_maj / N_min

where d is the local intrinsic dimensionality at the query point,
estimated via the Levina-Bickel MLE:

    d_local = [ (1/(K-1)) * Σ log(d_K / d_i) ]^(-1)

Two per-class neighbour lists are used so both classes are always
represented, regardless of how imbalanced the dataset is.

No resampling, no vote reweighting, no hyperparameters beyond the
neighbourhood sizes — r and d are both read off the data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist

from src.algorithms.knn_base import KNNClassifier
from src.utils.config import load_config


class KNNFairRank(KNNClassifier):
    """KNN with order-statistics-based rank correction for imbalance.

    Parameters
    ----------
    k_min : int
        Number of nearest minority neighbours to fetch. Used for LID and
        to provide the rank-1 minority distance.
    k_maj_buffer : int
        Additive buffer added to ceil(r) when sizing the majority
        neighbourhood; k_maj = max(ceil(r) + buffer, k_maj_floor).
    k_maj_floor : int
        Lower bound on the majority neighbourhood size, so low-r
        datasets still have enough samples for stable LID.
    k_maj_cap : int or None
        Upper bound on the majority neighbourhood size for very
        imbalanced datasets. None = no cap.
    """

    def __init__(
        self,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank", {})
        self._k_min = k_min if k_min is not None else cfg.get("k_min", 10)
        self._k_maj_buffer = k_maj_buffer if k_maj_buffer is not None else cfg.get("k_maj_buffer", 10)
        self._k_maj_floor = k_maj_floor if k_maj_floor is not None else cfg.get("k_maj_floor", 30)
        self._k_maj_cap = k_maj_cap if k_maj_cap is not None else cfg.get("k_maj_cap", 100)
        self._minority_class = None
        self._majority_class = None
        self._X_min: NDArray | None = None
        self._X_maj: NDArray | None = None
        self._r: float = 1.0
        self._k_maj_eff: int = 0
        super().__init__(k=1, n_jobs=n_jobs)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRank":
        super().fit(X, y)
        counts = {c: int(np.sum(self.y == c)) for c in self.classes_}
        self._minority_class = min(counts, key=counts.get)
        self._majority_class = max(counts, key=counts.get)
        n_min = counts[self._minority_class]
        n_maj = counts[self._majority_class]
        self._r = n_maj / max(1, n_min)

        self._X_min = self.X[self.y == self._minority_class]
        self._X_maj = self.X[self.y == self._majority_class]

        k_maj = max(int(np.ceil(self._r)) + self._k_maj_buffer, self._k_maj_floor)
        if self._k_maj_cap is not None:
            k_maj = min(k_maj, self._k_maj_cap)
        self._k_maj_eff = min(k_maj, len(self._X_maj))
        self._k_min_eff = min(self._k_min, len(self._X_min))
        return self

    def _per_class_distances(self, x: NDArray) -> tuple[NDArray, NDArray]:
        """Return (sorted_min_dists, sorted_maj_dists) trimmed to the
        per-class neighbourhood sizes fixed at fit time."""
        x2 = x.reshape(1, -1)
        d_min_all = cdist(self._X_min, x2, metric="euclidean").ravel()
        d_maj_all = cdist(self._X_maj, x2, metric="euclidean").ravel()

        if self._k_min_eff < len(d_min_all):
            idx_min = np.argpartition(d_min_all, self._k_min_eff - 1)[: self._k_min_eff]
            d_min = np.sort(d_min_all[idx_min])
        else:
            d_min = np.sort(d_min_all)

        if self._k_maj_eff < len(d_maj_all):
            idx_maj = np.argpartition(d_maj_all, self._k_maj_eff - 1)[: self._k_maj_eff]
            d_maj = np.sort(d_maj_all[idx_maj])
        else:
            d_maj = np.sort(d_maj_all)

        return d_min, d_maj

    def _estimate_lid(self, distances: NDArray) -> float:
        """Levina-Bickel MLE of local intrinsic dimensionality."""
        if len(distances) < 2:
            return 1.0
        # Guard against zero distances (duplicate training points)
        d = distances[distances > 0]
        if len(d) < 2:
            return 1.0
        d_k = d[-1]
        log_ratios = np.log(d_k / d[:-1])
        mean_log = np.mean(log_ratios)
        if mean_log <= 0 or not np.isfinite(mean_log):
            return float(len(d))  # fall back to ambient-like
        return float(1.0 / mean_log)

    def _decide(self, x: NDArray) -> int:
        if len(self._X_min) == 0:
            return self._majority_class
        if len(self._X_maj) == 0:
            return self._minority_class

        d_min, d_maj = self._per_class_distances(x)

        if len(d_min) == 0:
            return self._majority_class
        if len(d_maj) == 0:
            return self._minority_class

        merged = np.sort(np.concatenate([d_min, d_maj]))
        d_local = self._estimate_lid(merged)
        d_local = max(d_local, 1.0)

        k_eff_float = self._r ** (1.0 / d_local)
        k_eff = int(max(1, round(k_eff_float)))
        k_eff = min(k_eff, len(d_maj))

        d_min_ref = d_min[0]
        d_maj_ref = d_maj[k_eff - 1]

        if d_min_ref < d_maj_ref:
            return self._minority_class
        return self._majority_class

    def _predict_x(self, x: NDArray):
        return self._decide(x)

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        """Soft score derived from the d_min / (d_min + d_maj) ratio.

        Uses the same k_eff rank correction as the hard decision, so
        the two are consistent. A lower ratio means minority is closer
        in the fair-rank sense, so p(minority) is higher.
        """
        if len(self._X_min) == 0 or len(self._X_maj) == 0:
            prob = np.zeros(len(self.classes_), dtype=float)
            present = self._majority_class if len(self._X_min) == 0 else self._minority_class
            prob[np.where(self.classes_ == present)[0][0]] = 1.0
            return prob

        d_min, d_maj = self._per_class_distances(x)
        merged = np.sort(np.concatenate([d_min, d_maj]))
        d_local = max(self._estimate_lid(merged), 1.0)
        k_eff = int(max(1, round(self._r ** (1.0 / d_local))))
        k_eff = min(k_eff, len(d_maj))

        d_min_ref = d_min[0]
        d_maj_ref = d_maj[k_eff - 1]
        total = d_min_ref + d_maj_ref
        if total <= 0:
            p_min = 0.5
        else:
            # Closer minority → smaller d_min_ref → larger p_min
            p_min = d_maj_ref / total

        prob = np.zeros(len(self.classes_), dtype=float)
        idx_min = int(np.where(self.classes_ == self._minority_class)[0][0])
        idx_maj = int(np.where(self.classes_ == self._majority_class)[0][0])
        prob[idx_min] = p_min
        prob[idx_maj] = 1.0 - p_min
        return prob
