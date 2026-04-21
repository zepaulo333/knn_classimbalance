"""KNN with statistically fair rank correction for class imbalance.

Under class imbalance, the majority class always produces a closer
rank-1 neighbour than the minority class as a pure sampling artifact:
with N_maj >> N_min, more samples means smaller minimum-distance
statistics even when the underlying distributions are equivalent.

This algorithm corrects that bias by running multiple rank-corrected
comparisons and taking a majority vote. The rank correction is derived
from order statistics of Poisson-process nearest-neighbour distances:

    E[d_k^c] ∝ (k / N_c)^(1/d)

Setting E[d_{k_eff}^maj] = E[d_1^min] and raising both sides to the
power d cancels the dimension, giving the dimension-free fair rank

    k_eff = r   with   r = N_maj / N_min

More generally, rank i on the minority side is statistically equivalent
to rank i * k_eff on the majority side. We therefore run n_votes such
comparisons and vote:

    for i in 1..n_votes:
        vote minority if d_i^min < d_{i * k_eff}^maj

Dimensionality still influences the numerical effect: the ratio
E[d_r^maj] / E[d_1^maj] ≈ r^(1/d) tells us how much the correction
shifts the majority reference. In high dimensions this ratio goes to 1
(concentration of measure), so the correction is self-damping through
the distance values without any need to shrink k_eff.

Two per-class neighbour lists are used so both classes are always
represented, regardless of how imbalanced the dataset is.
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
        Number of nearest minority neighbours to fetch. Used for LID
        estimation and as the upper bound on the number of rank-corrected
        vote comparisons.
    k_maj_buffer : int
        Additive buffer added to ceil(r) when sizing the majority
        neighbourhood; k_maj = max(ceil(r) + buffer, k_maj_floor).
    k_maj_floor : int
        Lower bound on the majority neighbourhood size, so low-r
        datasets still have enough samples for stable LID.
    k_maj_cap : int or None
        Upper bound on the majority neighbourhood size for very
        imbalanced datasets. None = no cap.
    n_votes : int
        Maximum number of rank-corrected comparisons to aggregate into
        the final vote. Actual number used per query is clipped by
        available neighbours.
    """

    def __init__(
        self,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank", {})
        self._k_min = k_min if k_min is not None else cfg.get("k_min", 10)
        self._k_maj_buffer = k_maj_buffer if k_maj_buffer is not None else cfg.get("k_maj_buffer", 10)
        self._k_maj_floor = k_maj_floor if k_maj_floor is not None else cfg.get("k_maj_floor", 30)
        self._k_maj_cap = k_maj_cap if k_maj_cap is not None else cfg.get("k_maj_cap", 1000)
        self._n_votes = n_votes if n_votes is not None else cfg.get("n_votes", 5)
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

        # Majority neighbourhood must cover n_votes * k_eff in the worst case.
        # For the LID-damped variant, k_eff ≤ r, so worst case is n_votes * r.
        # We size k_maj generously but cap it to avoid blowing up on extreme r.
        k_maj = max(int(np.ceil(self._r)) * self._n_votes + self._k_maj_buffer,
                    self._k_maj_floor)
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
        d = distances[distances > 0]
        if len(d) < 2:
            return 1.0
        d_k = d[-1]
        log_ratios = np.log(d_k / d[:-1])
        mean_log = np.mean(log_ratios)
        if mean_log <= 0 or not np.isfinite(mean_log):
            return float(len(d))
        return float(1.0 / mean_log)

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        """Run the rank-corrected vote. Returns (fraction_minority, n_votes).

        For i in 1..n_votes_eff, compares d_i^min to d_{i*k_eff}^maj and
        counts the fraction of comparisons where minority is closer.
        k_eff is the global imbalance ratio r (see module docstring).
        """
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        # Dimension-free fair rank: k_eff = r.
        k_eff = int(max(1, round(self._r)))

        # Number of votes we can actually support:
        # - limited by minority samples (|d_min|)
        # - limited by the largest majority rank we can reach (|d_maj| // k_eff)
        # - user-specified cap (n_votes)
        max_votes_maj = len(d_maj) // k_eff
        n_votes = min(self._n_votes, len(d_min), max_votes_maj)
        if n_votes < 1:
            # Fallback: at least one comparison using what we have
            n_votes = 1
            k_eff = min(k_eff, len(d_maj))

        # Vectorised comparison
        min_refs = d_min[:n_votes]                                # d_1..d_n_votes
        maj_indices = np.arange(1, n_votes + 1) * k_eff - 1       # 0-based
        maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
        maj_refs = d_maj[maj_indices]                             # d_{k_eff}..d_{n_votes*k_eff}

        votes_minority = int(np.sum(min_refs < maj_refs))
        return votes_minority / n_votes, n_votes

    def _decide(self, x: NDArray) -> int:
        if len(self._X_min) == 0:
            return self._majority_class
        if len(self._X_maj) == 0:
            return self._minority_class

        frac_min, n_votes = self._vote_fraction(x)
        if n_votes == 0:
            return self._majority_class
        # Tie goes to majority (conservative; preserves precision).
        return self._minority_class if frac_min > 0.5 else self._majority_class

    def _predict_x(self, x: NDArray):
        return self._decide(x)

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        """Soft score = fraction of rank-corrected votes favouring minority.

        This is naturally consistent with the hard decision: the hard
        rule predicts minority iff this fraction > 0.5.
        """
        prob = np.zeros(len(self.classes_), dtype=float)
        idx_min = int(np.where(self.classes_ == self._minority_class)[0][0])
        idx_maj = int(np.where(self.classes_ == self._majority_class)[0][0])

        if len(self._X_min) == 0:
            prob[idx_maj] = 1.0
            return prob
        if len(self._X_maj) == 0:
            prob[idx_min] = 1.0
            return prob

        frac_min, n_votes = self._vote_fraction(x)
        if n_votes == 0:
            prob[idx_maj] = 1.0
            return prob

        prob[idx_min] = frac_min
        prob[idx_maj] = 1.0 - frac_min
        return prob
