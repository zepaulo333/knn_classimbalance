"""KNNFairRank with α-grid ensemble voting (no inner CV).

KNNFairRankCV selects a single α ∈ {0.25, 0.5, 0.75, 1.0} by inner
stratified CV and uses it for every prediction. This variant eliminates
the CV step: every α in the grid contributes votes at predict time and
the plurality determines the class.

For each (α, i) pair — candidate correction strength × minority rank —
a binary vote d_i^min < d_{i·round(r^α)}^maj is cast. All votes across
the full (α × n_votes) grid are summed and divided by the total count,
producing a vote fraction treated identically to v3's.

Properties
----------
- No fit-time CV overhead; cheaper than KNNFairRankCV.
- Majority array sized for α=1 (inherited from KNNFairRank.fit), so no
  additional memory beyond v3.
- Robust to outlier minority distances: a single anomalous d_i^min
  affects only votes at rank i, a fraction 1/n_votes of the grid.
- Under homogeneous Poisson density, votes at α<1 and α=1 partially
  offset each other, recovering behaviour close to v3. Under
  heterogeneous density the consensus tips toward the locally correct
  class without committing to a single correction strength.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.algorithms.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankEnsemble(KNNFairRank):
    """KNNFairRank with ensemble voting over an α grid (no inner CV).

    Parameters
    ----------
    alpha_grid : list[float] or None
        Candidate correction exponents. None → reads from
        ``knn_fair_rank_ensemble.alpha_grid`` in settings.yaml,
        falling back to KNNFairRankCV's default [0.25, 0.5, 0.75, 1.0].
    """

    def __init__(
        self,
        alpha_grid: list[float] | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config()
        fallback = cfg.get("knn_fair_rank_cv", {}).get("alpha_grid", [0.25, 0.5, 0.75, 1.0])
        self._alpha_grid = (
            list(alpha_grid) if alpha_grid is not None
            else list(cfg.get("knn_fair_rank_ensemble", {}).get("alpha_grid", fallback))
        )
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        minority_wins = 0
        total_votes = 0

        for alpha in self._alpha_grid:
            k_eff = int(max(1, round(self._r ** alpha)))
            max_votes_maj = len(d_maj) // k_eff
            n = min(self._n_votes, len(d_min), max_votes_maj)
            if n < 1:
                n = 1
                k_eff = min(k_eff, len(d_maj))

            min_refs = d_min[:n]
            maj_indices = np.arange(1, n + 1) * k_eff - 1
            maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
            maj_refs = d_maj[maj_indices]

            minority_wins += int(np.sum(min_refs < maj_refs))
            total_votes += n

        if total_votes == 0:
            return 0.0, 0
        return minority_wins / total_votes, total_votes
