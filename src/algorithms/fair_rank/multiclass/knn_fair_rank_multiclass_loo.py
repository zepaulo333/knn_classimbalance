"""KNNFairRankMulticlass with leave-one-class-out aggregation.

For each query x, runs K LOO predictions — each excluding one algorithm class
— and aggregates the fractional vote shares across all runs.

Why per-run median recomputation matters
-----------------------------------------
When class j is excluded, the remaining K-1 classes have a different size
distribution, so their median shifts.  Using the original median would assign
stale k_j ratios that reflect a class composition that no longer exists for
that run.  Each LOO run therefore recomputes its own median from the K-1
remaining class sizes, producing fresh, self-consistent k_j values.

Why this fixes the reference-class bias
-----------------------------------------
With the min-based reference the single-point class always occupies rank 1 and
dominates the comparison.  With the median-based reference that bias is already
reduced.  The LOO layer adds a second defence: in the one run where the tiny
class is excluded, the remaining classes compete with a fresh (higher) median
reference, giving larger classes a fairer chance.  A class that wins robustly
across most LOO configurations is a reliable choice; a class that only wins
because it happens to sit near the median in one configuration is diluted.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.fair_rank.multiclass.knn_fair_rank_multiclass import KNNFairRankMulticlass
from src.utils.config import load_config


class KNNFairRankMulticlassLOO(KNNFairRankMulticlass):
    """KNNFairRankMulticlass with leave-one-class-out vote aggregation.

    For each query, runs K rounds of FairRank (each excluding one algorithm
    class and recomputing the median reference from the remaining K-1 classes),
    then averages fractional vote shares across all rounds.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankMulticlassLOO":
        super().fit(X, y)
        # Fetch all neighbours per class: LOO runs may produce k_j values
        # larger than those computed at fit time, so we need the full distance
        # arrays to avoid clamping inside _interp_dist.
        for c in self.classes_:
            self._k_fetch[c] = self._N_per_class[c]
        return self

    # ── LOO prediction ────────────────────────────────────────────────────

    def _loo_scores(self, x: NDArray) -> dict[object, float]:
        """Aggregate fractional vote shares across K LOO runs."""
        dists = self._per_class_distances(x)

        totals: dict[object, float] = {c: 0.0 for c in self.classes_}
        n_valid_runs = 0

        for exclude_c in self.classes_:
            remaining = [c for c in self.classes_ if c != exclude_c]
            if len(remaining) < 2:
                continue

            # Fresh median and k_j for the K-1 remaining classes
            _, k_ratio_loo = self._compute_k_ratio(self._N_per_class, remaining)

            votes = self._run_votes(dists, k_ratio=k_ratio_loo, active_classes=remaining)

            total = sum(votes.values())
            if total == 0:
                continue

            for c in remaining:
                totals[c] += votes[c] / total
            n_valid_runs += 1

        if n_valid_runs == 0:
            # Degenerate fallback: use standard median-based prediction
            votes = self._run_votes(dists)
            total = max(1, sum(votes.values()))
            return {c: votes[c] / total for c in self.classes_}

        return totals

    def _predict_x(self, x: NDArray):
        scores = self._loo_scores(x)
        return max(scores, key=scores.get)

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        scores = self._loo_scores(x)
        total = sum(scores.values())
        if total == 0:
            return np.ones(len(self.classes_)) / len(self.classes_)
        return np.array([scores[c] / total for c in self.classes_])
