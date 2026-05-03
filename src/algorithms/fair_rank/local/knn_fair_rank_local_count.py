"""KNNFairRank with per-query k_eff from minority-budget per-class counting.

Replaces the global k_eff = r used by KNNFairRank with a per-query estimate.
Fixing the minority denominator at k_ref, count how many majority training
points fall inside the resulting radius:

    radius      = d_{k_ref, min}(x)
    k_eff(x)    = max(1, round( #{maj : ||x_i^maj - x|| <= radius } / k_ref ))

The minority denominator is exactly k_ref by construction (no Poisson noise
on the denominator), and k_eff(x) reduces to the global r under uniform
density. In minority-dense regions k_eff(x) << r (milder correction); in
majority-dense regions k_eff(x) > r (stronger correction).

Default k_ref = max(3, floor(sqrt(N_min))) per §19.

The estimate is computed inside `_vote_fraction` from the per-class sorted
distances already produced by the base class — one extra binary search per
query, no new k-NN passes, no topology, no inner CV.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.fair_rank.core.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankLocalCount(KNNFairRank):
    """KNNFairRank with per-query k_eff from per-class counting at k_ref.

    Parameters
    ----------
    k_ref : int or None
        Minority-budget denominator for the local density estimate.
        None → adaptive default max(3, floor(sqrt(N_min))) at fit time.
    """

    def __init__(
        self,
        k_ref: int | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_local_count", {})
        self._k_ref_param = k_ref if k_ref is not None else cfg.get("k_ref", None)
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankLocalCount":
        super().fit(X, y)
        n_min = len(self._X_min)
        if self._k_ref_param is None:
            self._k_ref = max(3, int(np.floor(np.sqrt(n_min))))
        else:
            self._k_ref = int(self._k_ref_param)
        self._k_ref = max(1, min(self._k_ref, n_min))
        # d_min must cover both k_ref (radius) and n_votes (voting step).
        needed_min = max(self._k_ref, self._n_votes)
        self._k_min_eff = min(max(self._k_min_eff, needed_min), n_min)
        return self

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        # §20 per-class counting estimate at the k_ref-th minority neighbour.
        k_ref = min(self._k_ref, len(d_min))
        radius = d_min[k_ref - 1]
        n_maj_local = int(np.searchsorted(d_maj, radius, side="right"))
        k_eff = int(max(1, round(n_maj_local / k_ref)))

        # Standard v3 binary voting with the per-query k_eff.
        max_votes_maj = len(d_maj) // k_eff
        n_votes = min(self._n_votes, len(d_min), max_votes_maj)
        if n_votes < 1:
            n_votes = 1
            k_eff = min(k_eff, len(d_maj))

        min_refs = d_min[:n_votes]
        maj_indices = np.arange(1, n_votes + 1) * k_eff - 1
        maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
        maj_refs = d_maj[maj_indices]

        votes_minority = int(np.sum(min_refs < maj_refs))
        return votes_minority / n_votes, n_votes
