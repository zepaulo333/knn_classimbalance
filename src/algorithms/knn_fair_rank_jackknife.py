"""KNNFairRank with leave-one-out jackknife over minority ranks.

At predict time, for each query x:

  1. Fetch the usual per-class sorted distance arrays.
  2. Run k_probe LOO trials: trial j removes d_j^min from the minority
     distance sequence, shifts all higher ranks down by one, and re-runs
     the standard v3 binary-vote step on the shifted sequence.
  3. Average the vote fractions across all trials and predict minority
     iff the average > 0.5.

Motivation
----------
In v3, minority rank i participates in exactly one comparison. A single
noise minority neighbour (duplicate, mislabelled point, or sampling
artefact) sitting unusually close to the query can dominate vote i and
pull the prediction toward minority regardless of the other votes.

The LOO average dilutes this influence: the anomalous point is absent
from exactly one trial and present in all others. If its removal
changes the vote fraction substantially, the average is less extreme
than the full-data result; if its removal changes nothing, the average
equals v3 exactly.

Unlike KNNFairRankEnsemble (which varies α to probe correction strength),
this variant holds k_eff = round(r) fixed (the v3 theoretical value)
and probes sensitivity to individual minority data points.

Fallback
--------
If k_min is too small to run any LOO trial (fewer than n_votes + 1
minority distances fetched), _vote_fraction falls back to standard v3.
This only happens on extremely small datasets or degenerate CV folds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankJackknife(KNNFairRank):
    """KNNFairRank with LOO jackknife over minority ranks.

    Parameters
    ----------
    k_probe : int or None
        Number of LOO trials — how many minority ranks to try excluding
        (starting from rank 1). None → uses n_votes, i.e. try excluding
        each of the minority neighbours that actively participate in
        votes. Must be ≤ k_min − n_votes so that n_votes distances
        remain after each removal; capped automatically at fit time.
    """

    def __init__(
        self,
        k_probe: int | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_jackknife", {})
        self._k_probe = k_probe if k_probe is not None else cfg.get("k_probe", None)
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankJackknife":
        super().fit(X, y)
        k_probe = self._k_probe if self._k_probe is not None else self._n_votes
        # After removing one point we need at least n_votes remaining.
        self._k_probe_eff = min(k_probe, max(0, self._k_min_eff - self._n_votes))
        return self

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        k_eff = int(max(1, round(self._r)))

        # Not enough minority distances for any LOO trial — fall back to v3.
        if self._k_probe_eff <= 0 or len(d_min) <= self._n_votes:
            max_votes_maj = len(d_maj) // k_eff
            n = min(self._n_votes, len(d_min), max_votes_maj)
            if n < 1:
                n = 1
                k_eff = min(k_eff, len(d_maj))
            min_refs = d_min[:n]
            maj_indices = np.arange(1, n + 1) * k_eff - 1
            maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
            maj_refs = d_maj[maj_indices]
            return int(np.sum(min_refs < maj_refs)) / n, n

        fracs = np.empty(self._k_probe_eff)
        for j in range(self._k_probe_eff):
            # Remove the j-th nearest minority neighbour; shift ranks down.
            d_min_loo = np.concatenate([d_min[:j], d_min[j + 1:]])

            max_votes_maj = len(d_maj) // k_eff
            n = min(self._n_votes, len(d_min_loo), max_votes_maj)
            k_e = k_eff
            if n < 1:
                n = 1
                k_e = min(k_eff, len(d_maj))

            min_refs = d_min_loo[:n]
            maj_indices = np.arange(1, n + 1) * k_e - 1
            maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
            fracs[j] = int(np.sum(min_refs < d_maj[maj_indices])) / n

        return float(np.mean(fracs)), self._k_probe_eff
