"""KNNFairRank with Bayesian α-shrinkage (§21).

The joint-CV variant selects a global correction exponent α_CV that minimises
cross-validated loss.  It is a "high-bias" estimator: it applies the same
k_eff = round(r^α_CV) everywhere regardless of local density.

This variant refines that global estimate with a per-query local signal from
the §20 per-class counting estimator, blended via a Bühlmann credibility weight:

    1. Local density signal (§20):
           radius      = d_{k_ref}^min(x)
           n_maj_local = #{maj : ||x_i^maj - x|| ≤ radius}
           ρ̂(x)        = n_maj_local / k_ref

    2. Map local density to α-space (α_local s.t. r^α_local = ρ̂):
           α_local(x)  = log(ρ̂(x)) / log(r)

    3. Bühlmann credibility weight (w → 1 = trust global prior):
           w(x)        = k_ref / (k_ref + n_maj_local(x))

       Interpretation: when there is very little majority evidence locally
       (n_maj_local ≈ 0) the denominator shrinks to k_ref and w = 1 — we
       fall back entirely to the global CV prior.  As local majority evidence
       accumulates (n_maj_local ≫ k_ref) w → 0 — we trust the local estimate.
       k_ref is its own prior strength, so no extra hyperparameter is needed.

    4. Blended exponent and final k_eff:
           α_final(x)  = w(x) · α_CV + (1 − w(x)) · α_local(x)
           k_eff(x)    = round(r^α_final(x))

    5. Standard v3 binary voting with the per-query k_eff.

Default k_ref = max(3, floor(sqrt(N_min))) following §19.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.fair_rank.core.knn_fair_rank import KNNFairRank
from src.algorithms.fair_rank.ensemble.knn_fair_rank_joint_cv import KNNFairRankJointCV
from src.utils.config import load_config


class KNNFairRankBayesian(KNNFairRankJointCV):
    """KNNFairRank with Bayesian α-shrinkage toward the CV-selected global prior.

    Parameters
    ----------
    k_ref : int or None
        Minority budget for the local density estimate.
        None → adaptive default max(3, floor(sqrt(N_min))) at fit time.
    All other parameters are forwarded to KNNFairRankJointCV.
    """

    def __init__(
        self,
        k_ref: int | None = None,
        n_votes_grid: list[int] | None = None,
        alpha_grid: list[float] | None = None,
        inner_cv_folds: int | None = None,
        scoring: str | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_bayesian", {})
        self._k_ref_param = k_ref if k_ref is not None else cfg.get("k_ref", None)
        super().__init__(
            n_votes_grid=n_votes_grid,
            alpha_grid=alpha_grid,
            inner_cv_folds=inner_cv_folds,
            scoring=scoring,
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_jobs=n_jobs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankBayesian":
        super().fit(X, y)
        n_min = len(self._X_min)
        if self._k_ref_param is None:
            self._k_ref = max(3, int(np.floor(np.sqrt(n_min))))
        else:
            self._k_ref = int(self._k_ref_param)
        self._k_ref = max(1, min(self._k_ref, n_min))
        # Ensure we fetch enough minority distances for both k_ref (radius) and n_votes.
        needed_min = max(self._k_ref, self._n_votes)
        self._k_min_eff = min(max(self._k_min_eff, needed_min), n_min)
        return self

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        r = self._r  # raw N_maj / N_min

        # §20 per-class counting at the k_ref-th minority neighbour.
        k_ref = min(self._k_ref, len(d_min))
        radius = d_min[k_ref - 1]
        n_maj_local = int(np.searchsorted(d_maj, radius, side="right"))

        # Bühlmann credibility weight: w=1 → pure global prior.
        w = k_ref / (k_ref + n_maj_local)

        # Compute blended exponent; fall back to α_CV when r≤1 or no local signal.
        if r > 1 and n_maj_local > 0:
            rho_hat = n_maj_local / k_ref
            alpha_local = np.log(rho_hat) / np.log(r)
            alpha_final = w * self._alpha + (1.0 - w) * alpha_local
        else:
            alpha_final = self._alpha

        k_eff = int(max(1, round(r ** alpha_final)))

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
