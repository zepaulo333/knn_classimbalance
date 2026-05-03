"""KNNFairRank ensemble over the full (n_votes, α) grid (no inner CV).

KNNFairRankJointCV selects a single (n_votes, α) pair by inner CV and commits
to it for every query.  This variant sweeps the entire grid at inference time:
every combination contributes binary votes, and the plurality determines the
class.

For each (n_v, α) pair and each rank i in 1..n_v, a binary vote

    d_i^min  <  d_{i · round(r^α)}^maj

is cast.  All votes across the full (n_votes × α) grid are pooled and divided
by the total count, producing a vote fraction identical to v3's decision rule.

Compared to KNNFairRankEnsemble (α-only grid, fixed n_votes):
- Majority neighbourhood is sized for max(n_votes_grid) × max(α_grid = 1),
  i.e. the same generous sizing as KNNFairRankJointCV — no extra memory.
- Pairs with more votes (high n_v) contribute proportionally more to the
  pool, which naturally downweights the noisiest single-vote (n_v=1) pairs.
- When all (n_v, α) combinations agree, the vote fraction is far from 0.5,
  giving a high-confidence prediction.  When they disagree, it is near 0.5
  and the prediction is conservative.
- Under homogeneous Poisson density, votes at α<1 partially cancel votes at
  α=1, recovering behaviour close to v3.  Under heterogeneous density the
  consensus tips toward the locally correct class without committing to a
  single (n_v, α) setting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.algorithms.fair_rank.core.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankEnsemble(KNNFairRank):
    """KNNFairRank ensemble over the full (n_votes, α) grid (no inner CV).

    Parameters
    ----------
    alpha_grid : list[float] or None
        Candidate correction exponents.  None → settings.yaml.
    n_votes_grid : list[int] or None
        Candidate vote counts.  None → settings.yaml.  The majority
        neighbourhood is sized for max(n_votes_grid) so all candidates
        are covered without refitting.
    """

    def __init__(
        self,
        alpha_grid: list[float] | None = None,
        n_votes_grid: list[int] | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config()
        ens_cfg = cfg.get("knn_fair_rank_ensemble", {})
        jcv_cfg = cfg.get("knn_fair_rank_joint_cv", {})

        self._alpha_grid: list[float] = (
            list(alpha_grid) if alpha_grid is not None
            else list(ens_cfg.get("alpha_grid", jcv_cfg.get("alpha_grid", [0.25, 0.5, 0.75, 1.0])))
        )
        self._n_votes_grid: list[int] = (
            list(n_votes_grid) if n_votes_grid is not None
            else list(ens_cfg.get("n_votes_grid", jcv_cfg.get("n_votes_grid", [1, 2, 3, 5, 7, 10])))
        )
        # Size k_maj for the largest n_votes candidate — same trick as JointCV.
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=max(self._n_votes_grid),
            n_jobs=n_jobs,
        )

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        minority_wins = 0
        total_votes = 0

        for n_v in self._n_votes_grid:
            for alpha in self._alpha_grid:
                k_eff = int(max(1, round(self._r ** alpha)))
                max_votes_maj = len(d_maj) // k_eff
                n = min(n_v, len(d_min), max_votes_maj)
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
