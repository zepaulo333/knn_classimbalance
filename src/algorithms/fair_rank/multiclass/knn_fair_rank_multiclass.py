"""KNN with statistically fair rank correction generalised to K ≥ 2 classes.

Binary FairRank corrects for sampling bias by comparing the minority's rank-1
neighbor to the majority's rank-r neighbor (r = IR).  This class extends that
to K classes simultaneously.

Reference choice
----------------
Using the least-populated class as the reference (min-based) is problematic:
the single-point class always compares at rank 1 (its nearest neighbor) while
every other class compares at a rank proportional to its size.  In algorithm
selection the rarest class is rare because it is genuinely rarely optimal, not
because of sampling bias — correcting for it destroys information.

Instead we use the **median** class size as reference:

    N_ref = median_j N_j
    k_j   = max(1, N_j / N_ref)

Classes smaller than the median all get k_j = 1 (compare at rank 1 — their
nearest neighbor), so no single class monopolises the rank-1 slot.  Classes
larger than the median are penalised proportionally to their size advantage
over the median, which is a much more moderate correction than the min-based
version.

Valid vote rounds
-----------------
The number of rounds is capped at floor(N_j / k_j) = floor(N_ref) for every
class, so rounds never exceed the number of distinct training points a class
can contribute.  No point is reused across rounds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config


class KNNFairRankMulticlass:
    """KNN with median-anchored fair rank correction for K-class imbalance.

    Parameters
    ----------
    n_votes : int
        Maximum number of rank-corrected rounds to aggregate per query.
    k_buffer : int
        Extra neighbours fetched per class beyond worst-case needed rank.
    k_floor : int
        Minimum neighbours fetched per class.
    k_cap : int or None
        Maximum neighbours fetched per class.
    normalize : bool
        If True, fit a StandardScaler on X during fit and apply it during
        predict.  Recommended for distance-based methods.
    n_jobs : int
    """

    def __init__(
        self,
        n_votes: int | None = None,
        k_buffer: int | None = None,
        k_floor: int | None = None,
        k_cap: int | None = None,
        normalize: bool = True,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_multiclass", {})
        self._n_votes  = n_votes  if n_votes  is not None else cfg.get("n_votes",  5)
        self._k_buffer = k_buffer if k_buffer is not None else cfg.get("k_buffer", 5)
        self._k_floor  = k_floor  if k_floor  is not None else cfg.get("k_floor",  5)
        self._k_cap    = k_cap    if k_cap    is not None else cfg.get("k_cap",    500)
        self._normalize = normalize
        self.n_jobs = n_jobs
        self._scaler: StandardScaler | None = None

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_k_ratio(N_per_class: dict, classes) -> tuple[float, dict]:
        """Compute median-anchored k_j ratios for the given class sizes.

        Returns (N_ref, k_ratio_dict) where k_j = max(1.0, N_j / N_ref).
        """
        N_vals = [N_per_class[c] for c in classes]
        N_ref = max(1.0, float(np.median(N_vals)))
        k_ratio = {c: max(1.0, N_per_class[c] / N_ref) for c in classes}
        return N_ref, k_ratio

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankMulticlass":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if self._normalize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        self.classes_ = np.unique(y)

        self._X_per_class: dict[object, NDArray] = {}
        self._N_per_class: dict[object, int] = {}
        for c in self.classes_:
            mask = y == c
            self._X_per_class[c] = X[mask]
            self._N_per_class[c] = int(mask.sum())

        self._N_ref, self._k_ratio = self._compute_k_ratio(
            self._N_per_class, self.classes_
        )

        # Neighbours to fetch: cover n_votes * ceil(k_j) + buffer
        self._k_fetch: dict[object, int] = {}
        for c in self.classes_:
            needed = int(np.ceil(self._k_ratio[c])) * self._n_votes + self._k_buffer
            needed = max(needed, self._k_floor)
            if self._k_cap is not None:
                needed = min(needed, self._k_cap)
            self._k_fetch[c] = min(needed, self._N_per_class[c])

        return self

    # ── distance helpers ──────────────────────────────────────────────────

    def _per_class_distances(self, x: NDArray) -> dict[object, NDArray]:
        """Sorted distances from x to k_fetch nearest neighbours per class."""
        x2 = x.reshape(1, -1)
        result: dict[object, NDArray] = {}
        for c in self.classes_:
            Xc = self._X_per_class[c]
            if len(Xc) == 0:
                result[c] = np.array([])
                continue
            d_all = cdist(Xc, x2, metric="euclidean").ravel()
            k = self._k_fetch[c]
            if k < len(d_all):
                idx = np.argpartition(d_all, k - 1)[:k]
                result[c] = np.sort(d_all[idx])
            else:
                result[c] = np.sort(d_all)
        return result

    @staticmethod
    def _interp_dist(sorted_dists: NDArray, k_float: float) -> float:
        """Distance at fractional 1-indexed rank k_float (linear interpolation)."""
        n = len(sorted_dists)
        if n == 0:
            return float("inf")
        k0 = max(0.0, min(k_float - 1.0, float(n - 1)))
        lo = int(np.floor(k0))
        hi = int(np.ceil(k0))
        if lo == hi:
            return float(sorted_dists[lo])
        frac = k0 - lo
        return float((1.0 - frac) * sorted_dists[lo] + frac * sorted_dists[hi])

    # ── voting ────────────────────────────────────────────────────────────

    def _run_votes(
        self,
        dists: dict[object, NDArray],
        k_ratio: dict[object, float] | None = None,
        active_classes: list | None = None,
    ) -> dict[object, int]:
        """Run up to n_votes rounds and return raw win counts per class.

        Parameters
        ----------
        dists : per-class sorted distance arrays (from _per_class_distances)
        k_ratio : fair-rank ratios to use; defaults to self._k_ratio
        active_classes : classes to include; defaults to all with non-empty dists
        """
        if k_ratio is None:
            k_ratio = self._k_ratio

        if active_classes is None:
            active = [c for c in self.classes_ if len(dists.get(c, [])) > 0]
        else:
            active = [c for c in active_classes if len(dists.get(c, [])) > 0]

        if not active:
            return {c: 0 for c in self.classes_}

        # Max rounds = min over active classes of floor(len / k_j), capped by n_votes.
        # With median-anchored k_j: k_j ≤ N_j for every class, so floor(N_j / k_j) ≥ 1
        # and no point is reused across rounds.
        max_rounds = self._n_votes
        for c in active:
            k_j = k_ratio[c]
            available = int(np.floor(len(dists[c]) / k_j)) if k_j > 0 else 0
            max_rounds = min(max_rounds, available)
        max_rounds = max(1, max_rounds)

        votes: dict[object, int] = {c: 0 for c in self.classes_}
        for i in range(1, max_rounds + 1):
            round_dists = {
                c: self._interp_dist(dists[c], i * k_ratio[c])
                for c in active
            }
            winner = min(round_dists, key=round_dists.get)
            votes[winner] += 1

        return votes

    # ── predict interface ─────────────────────────────────────────────────

    def _predict_x(self, x: NDArray):
        votes = self._run_votes(self._per_class_distances(x))
        return max(votes, key=votes.get)

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        votes = self._run_votes(self._per_class_distances(x))
        total = sum(votes.values())
        if total == 0:
            return np.ones(len(self.classes_)) / len(self.classes_)
        return np.array([votes[c] / total for c in self.classes_])

    def predict(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self._normalize and self._scaler is not None:
            X = self._scaler.transform(X)
        if self.n_jobs == 1:
            return np.array([self._predict_x(x) for x in X])
        from joblib import Parallel, delayed
        return np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_x)(x) for x in X
        ))

    def predict_proba(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self._normalize and self._scaler is not None:
            X = self._scaler.transform(X)
        if self.n_jobs == 1:
            return np.array([self._predict_proba_x(x) for x in X])
        from joblib import Parallel, delayed
        return np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_proba_x)(x) for x in X
        ))
