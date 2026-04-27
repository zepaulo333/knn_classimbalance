"""KNNFairRankLocalOdds with LOO jackknife over minority ranks.

For each LOO trial j, the j-th minority neighbour is removed from the
sequence. The per-query k_eff is then re-estimated from the shifted
sequence (rank-pair interpolation, same logic as KNNFairRankLocalOdds),
and the standard binary vote is cast with that estimate.

This tests whether the LOO mechanism helps LocalOdds in the same way it
helped v3: by reducing the sensitivity of the k_eff estimate to
individual outlier minority distances.

In LocalOdds the k_eff estimate is derived from searchsorted ratios
j_i / i. A single anomalously close minority point (small d_j^min)
pushes all ratios j_i / i downward for i <= j, potentially
under-estimating k_eff and weakening the correction. Excluding it in one
LOO trial and averaging with trials that include it dampens this effect.

Fallback
--------
If not enough minority distances are available for any LOO trial, falls
back to the standard LocalOdds vote (no jackknife).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.knn_fair_rank_local_odds import KNNFairRankLocalOdds
from src.utils.config import load_config


class KNNFairRankLocalOddsJackknife(KNNFairRankLocalOdds):
    """KNNFairRankLocalOdds with LOO jackknife over minority ranks.

    Parameters
    ----------
    k_probe : int or None
        Controls both the estimation depth (minority ranks used to
        estimate k_eff) and the number of LOO trials. None → uses
        n_votes. Capped at k_min − n_votes for the LOO side.
    shrinkage : float or None
        Passed through to KNNFairRankLocalOdds unchanged.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankLocalOddsJackknife":
        super().fit(X, y)
        k_probe = self._k_probe if self._k_probe is not None else self._n_votes
        # LOO trials need at least n_votes distances remaining after removal.
        self._jk_probe_eff = min(k_probe, max(0, self._k_min_eff - self._n_votes))
        return self

    # ── helper: one LocalOdds vote on an arbitrary minority sequence ─────────

    def _local_odds_vote(self, d_min_seq: NDArray, d_maj: NDArray) -> tuple[float, int]:
        """LocalOdds vote on d_min_seq (may be a LOO-shifted sequence)."""
        n_probe = min(self._k_probe_eff, len(d_min_seq))
        probe_dists = d_min_seq[:n_probe]
        js = np.searchsorted(d_maj, probe_dists, side="right").astype(float)
        ranks = np.arange(1, n_probe + 1, dtype=float)
        estimates = js / ranks

        raw = float(np.median(estimates))
        lam = self._shrinkage
        k_eff_raw = (raw + lam * self._r) / (1.0 + lam)
        k_eff = int(max(1, round(k_eff_raw)))

        max_votes_maj = len(d_maj) // k_eff
        n = min(self._n_votes, len(d_min_seq), max_votes_maj)
        k_e = k_eff
        if n < 1:
            n = 1
            k_e = min(k_eff, len(d_maj))

        min_refs = d_min_seq[:n]
        maj_indices = np.arange(1, n + 1) * k_e - 1
        maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
        return int(np.sum(min_refs < d_maj[maj_indices])) / n, n

    # ── main vote ────────────────────────────────────────────────────────────

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        # Fallback: not enough minority distances for any LOO trial.
        if self._jk_probe_eff <= 0 or len(d_min) <= self._n_votes:
            return self._local_odds_vote(d_min, d_maj)

        fracs = np.empty(self._jk_probe_eff)
        for j in range(self._jk_probe_eff):
            d_min_loo = np.concatenate([d_min[:j], d_min[j + 1:]])
            fracs[j], _ = self._local_odds_vote(d_min_loo, d_maj)

        return float(np.mean(fracs)), self._jk_probe_eff
