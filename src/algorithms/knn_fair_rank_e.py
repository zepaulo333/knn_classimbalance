"""KNNFairRank with per-query k_eff from a shrinkage-stabilised local density
(Modification E, rank-native — analysis.ipynb §13).

v3's global k_eff = r assumes homogeneous Poisson density. This variant
keeps v3's binary-vote framework untouched and replaces the global r
with a per-query k_eff(x) derived from the local class density at x,
combined with v3 via Bayesian shrinkage so that local evidence cannot
collapse the rank correction in pathological neighbourhoods.

Local-density estimator (rank-native)
-------------------------------------
Pick k_target minority points and let the local radius adjust to enclose
exactly that many. Count how many majority points fall within. The
local "raw" rate estimator is n_maj_within / k_target — directly
estimating r at x (i.e. how many majority points sit at the same
distance scale as the k_target-th minority point). No d, no V_d.

Why minority-budget rather than fixed-K mixed. Under severe imbalance
(r = 100, K = 10) the K nearest mixed-class neighbours typically
contain zero minority points — the local odds are degenerate. Fixing
the minority count instead means the count n_maj_within ~ k_target·r,
giving a relative SE ~ 1/sqrt(k_target·r) — well-sampled even when r
is large.

Bayesian shrinkage to v3
------------------------
The raw local estimate is statistically clean but structurally
fragile: in dense minority clusters, n_maj_within can be ≪ k_target·r,
giving k_eff ≪ r and effectively disabling the imbalance correction
for any majority-class query that happens to be near such a cluster.

Treat the global r as a prior. Under uniform density n_maj_within is
Poisson(k_target · r). With a Gamma prior on the local rate centered on
r, the posterior mean (with relative prior strength λ) is

    k_eff(x) = ( n_maj_within  +  λ · k_target · r )
               / ( k_target · (1 + λ) )

λ = 0    →  pure local (raw odds)
λ = 1    →  equal weight: in a pure minority cluster, k_eff(x) = r/2
λ → ∞    →  pure v3 (k_eff(x) = r everywhere)

In expectation under uniform density k_eff(x) = r for any λ — the
shrinkage only kicks in when local evidence diverges from the global
prior, smoothly bounding how far k_eff(x) can drift from v3.

Voting
------
Once k_eff(x) is fixed for the query, the vote is exactly v3's binary
comparison: for i = 1..n_votes compare d_i^min against
d_{i · k_eff(x)}^maj, aggregate as a binary fraction, predict minority
iff > 0.5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankDensity(KNNFairRank):
    """v3 with per-query k_eff(x) from a shrinkage-stabilised local density.

    Parameters
    ----------
    k_target : int
        Minority budget for the local-density estimate. Default reads from
        `knn_fair_rank_density.k_target`, falling back to the base k_min.
    shrinkage : float
        Bayesian prior weight λ pulling k_eff(x) toward the global v3
        value r. λ = 0 is pure local; λ = 1 weights local and global
        evidence equally; large λ recovers v3. Default reads from
        `knn_fair_rank_density.shrinkage`, falling back to 1.0.
    """

    def __init__(
        self,
        k_target: int | None = None,
        shrinkage: float | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_density", {})
        self._k_target = k_target if k_target is not None else cfg.get("k_target", None)
        self._shrinkage = (
            float(shrinkage) if shrinkage is not None
            else float(cfg.get("shrinkage", 1.0))
        )
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )
        if self._k_target is None:
            self._k_target = self._k_min

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankDensity":
        super().fit(X, y)
        # k_target capped by available minority distances.
        self._k_target_eff = min(self._k_target, self._k_min_eff)

        # Bump k_maj_eff so the density estimator can count up to ~k_target·r
        # majority points within the local radius. Safety factor 2× covers
        # minority-poor queries where the radius is much larger than average.
        # This may exceed the parent's k_maj_cap — accepted, density E needs
        # better majority-side coverage than the binary-vote variants.
        k_maj_for_density = int(np.ceil(self._k_target_eff * self._r * 2))
        self._k_maj_eff = min(
            max(self._k_maj_eff, k_maj_for_density),
            len(self._X_maj),
        )
        return self

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        # Local radius enclosing exactly k_target minority points.
        k_t = min(self._k_target_eff, len(d_min))
        radius = float(d_min[k_t - 1])

        # Count majority points within radius (d_maj is sorted ascending).
        # Saturates at len(d_maj); fit() sized the array to make truncation
        # rare in practice.
        n_maj_within = int(np.searchsorted(d_maj, radius, side="right"))

        # Bayesian shrinkage to v3: posterior mean of the local rate under
        # a Gamma prior centered on r with relative weight λ.
        lam = self._shrinkage
        k_eff_raw = (n_maj_within + lam * k_t * self._r) / (k_t * (1.0 + lam))
        k_eff = int(max(1, round(k_eff_raw)))

        # v3 binary voting with the per-query k_eff (logic identical to parent).
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
