"""KNNFairRank with per-query k_eff from rank-pair interpolation
(Modification E, rank-native v2).

The base KNNFairRank uses a global k_eff = r derived under the
homogeneous-Poisson assumption. This variant estimates the local density
ratio λ_maj(x)/λ_min(x) per query, replacing the global r with a
data-driven per-query k_eff(x), then uses the standard v3 binary voting
step unchanged.

Local ratio estimation (rank-pair interpolation)
-------------------------------------------------
For each minority rank i = 1..k_probe, the position of d_i^min in the
sorted majority distance array is found via searchsorted:

    j_i = #{maj training points closer to x than d_i^min}

Under homogeneous Poisson, E[j_i / i] = r. Under locally dense minority
(small d_i^min), j_i / i < r. Under locally dense majority (large d_i^min
relative to majority distances), j_i / i > r.

The k_probe estimates {j_i / i} are combined via the median (robust to
individual outlier minority distances) and Bayesian-shrunk toward the
global r to prevent degenerate k_eff values in sparse regions.

Why this avoids the d-estimation problem
-----------------------------------------
The ratio j_i / i is a counting operation on the per-class sorted
distance arrays — no intrinsic dimensionality d appears. It avoids the
mixed-class neighbourhood bias (using class-separated distances, not
joint K-NN) and the (d_min/d_maj)^d amplification that made the §11
density-ratio formula fragile (analysis.ipynb §12).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankLocalOdds(KNNFairRank):
    """KNNFairRank with per-query k_eff from rank-pair interpolation.

    Parameters
    ----------
    k_probe : int or None
        Number of minority ranks used to estimate the local ratio.
        None → uses k_min (all fetched minority distances contribute).
    shrinkage : float or None
        Bayesian prior weight λ pulling k_eff(x) toward the global r.
        λ=0: pure local estimate. λ=1: equal weight local/global.
        Large λ: recovers v3 everywhere. Default 1.0.
    """

    def __init__(
        self,
        k_probe: int | None = None,
        shrinkage: float | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_local_odds", {})
        self._k_probe = k_probe if k_probe is not None else cfg.get("k_probe", None)
        self._shrinkage = float(
            shrinkage if shrinkage is not None else cfg.get("shrinkage", 1.0)
        )
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )
        if self._k_probe is None:
            self._k_probe = self._k_min

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankLocalOdds":
        super().fit(X, y)
        self._k_probe_eff = min(self._k_probe, self._k_min_eff)
        return self

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        # Vectorised rank-pair interpolation over k_probe minority ranks.
        n_probe = min(self._k_probe_eff, len(d_min))
        probe_dists = d_min[:n_probe]
        js = np.searchsorted(d_maj, probe_dists, side="right").astype(float)
        ranks = np.arange(1, n_probe + 1, dtype=float)
        estimates = js / ranks  # shape (n_probe,)

        # Bayesian shrinkage: posterior mean under a prior centred on r.
        raw = float(np.median(estimates))
        lam = self._shrinkage
        k_eff_raw = (raw + lam * self._r) / (1.0 + lam)
        k_eff = int(max(1, round(k_eff_raw)))

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
