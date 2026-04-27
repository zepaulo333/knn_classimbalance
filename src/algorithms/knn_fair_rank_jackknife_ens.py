"""KNNFairRank with LOO jackknife over minority ranks combined with
α-grid ensemble voting.

Combines both robustness mechanisms developed independently:

  KNNFairRankJackknife  — LOO over minority ranks (holds k_eff = r fixed,
                          averages vote fractions across trials that each
                          exclude one minority neighbour).

  KNNFairRankEnsemble   — α-grid ensemble (holds the minority sequence
                          fixed, votes across multiple k_eff = round(r^α)
                          values and averages).

Here, for each LOO trial j (remove d_j^min, shift ranks) the full α-grid
vote is run on the shifted minority sequence. All (j, α, i) vote outcomes
are averaged into a single fraction.

This tests whether the two sources of improvement are additive:
  - LOO removes sensitivity to individual outlier minority distances.
  - α-grid introduces the conservative correction (α < 1 values) that
    raised precision and MCC in KNNFairRankEnsemble.

Fallback
--------
If k_min is too small for any LOO trial (fewer than n_votes + 1 minority
distances), the method falls back to the pure α-grid ensemble vote (no
LOO), which is identical to KNNFairRankEnsemble.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankJackknifeEnsemble(KNNFairRank):
    """LOO jackknife over minority ranks + α-grid ensemble voting.

    Parameters
    ----------
    alpha_grid : list[float] or None
        Candidate correction exponents. None → reads from
        ``knn_fair_rank_ensemble.alpha_grid`` in settings.yaml,
        falling back to [0.25, 0.5, 0.75, 1.0].
    k_probe : int or None
        Number of LOO trials. None → uses n_votes. Capped at
        k_min − n_votes at fit time.
    """

    def __init__(
        self,
        alpha_grid: list[float] | None = None,
        k_probe: int | None = None,
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
        self._k_probe = k_probe if k_probe is not None else cfg.get(
            "knn_fair_rank_jackknife", {}
        ).get("k_probe", None)
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankJackknifeEnsemble":
        super().fit(X, y)
        k_probe = self._k_probe if self._k_probe is not None else self._n_votes
        self._k_probe_eff = min(k_probe, max(0, self._k_min_eff - self._n_votes))
        return self

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        # ── α-grid helper (used both in LOO trials and in the fallback) ──────
        def _ensemble_frac(d_min_seq: NDArray) -> tuple[float, int]:
            minority_wins = 0
            total = 0
            for alpha in self._alpha_grid:
                k_eff = int(max(1, round(self._r ** alpha)))
                max_votes_maj = len(d_maj) // k_eff
                n = min(self._n_votes, len(d_min_seq), max_votes_maj)
                if n < 1:
                    n = 1
                    k_eff = min(k_eff, len(d_maj))
                min_refs = d_min_seq[:n]
                maj_indices = np.arange(1, n + 1) * k_eff - 1
                maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
                minority_wins += int(np.sum(min_refs < d_maj[maj_indices]))
                total += n
            return (minority_wins / total, total) if total > 0 else (0.0, 0)

        # Fallback: not enough minority distances for any LOO trial.
        if self._k_probe_eff <= 0 or len(d_min) <= self._n_votes:
            return _ensemble_frac(d_min)

        # LOO loop: for each excluded minority rank, run the full α-grid vote.
        all_fracs: list[float] = []
        for j in range(self._k_probe_eff):
            d_min_loo = np.concatenate([d_min[:j], d_min[j + 1:]])
            frac, _ = _ensemble_frac(d_min_loo)
            all_fracs.append(frac)

        return float(np.mean(all_fracs)), self._k_probe_eff
