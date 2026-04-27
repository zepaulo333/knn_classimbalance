"""KNNFairRank with LID-CV-derived α — no inner CV.

KNNFairRankCV selects α ∈ {0.25, 0.5, 0.75, 1.0} via inner stratified CV.
This variant derives α analytically from the *coefficient of variation* (CV)
of local intrinsic dimensionality (LID) estimates across the training set,
without any inner CV loop.

Motivation
----------
The FairRank correction k_eff = r^α is theoretically exact when the training
data follows a homogeneous Poisson process (uniform local density). α = 1 is
the principled default; departing from it is only justified when that uniformity
assumption is violated.

The right signal for violation is not the mean LID (which measures
dimensionality, and d cancels in the derivation) but the *spread* of LID
estimates across the dataset — i.e. how much the local geometry varies from
point to point. High LID variance → density is heterogeneous → the global r
over-corrects in some regions → α should be < 1.

α formula
---------
Let CV_LID = std(LID estimates) / mean(LID estimates). Then:

    α = clip(1 - scale * CV_LID,  alpha_min,  1.0)

Properties:
    CV_LID = 0  →  α = 1.0  (perfectly uniform: full Poisson correction)
    CV_LID = 1  →  α = 1 - scale  (scale controls sensitivity)

With the default scale = 0.75:
    CV_LID = 0.33  →  α ≈ 0.75
    CV_LID = 0.67  →  α ≈ 0.50
    CV_LID ≥ 1.0   →  α = 0.25  (clamped at alpha_min)

Why CV of LID rather than mean LID
-----------------------------------
Mean LID tells us the intrinsic dimensionality of the space — but d cancels
in the FairRank derivation, so mean LID is not the right predictor of α.
CV of LID tells us how non-uniform the local geometry is — which is exactly
what determines whether the global correction r should be damped.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist

from src.algorithms.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankLID(KNNFairRank):
    """KNNFairRank with α derived from the coefficient of variation of LID.

    Parameters
    ----------
    alpha_min : float or None
        Lower bound on α. None → ``knn_fair_rank_lid.alpha_min`` (default 0.25).
    scale : float or None
        Sensitivity of α to LID variation: α = 1 - scale * CV_LID.
        None → ``knn_fair_rank_lid.scale`` (default 0.75).
    lid_sample_size : int or None
        Training points sampled for LID estimation.
        None → ``knn_fair_rank_lid.lid_sample_size`` (default 200).
    lid_k : int or None
        Neighbours per point in the Levina-Bickel MLE.
        None → ``knn_fair_rank_lid.lid_k`` (default 10).
    """

    def __init__(
        self,
        alpha_min: float | None = None,
        scale: float | None = None,
        lid_sample_size: int | None = None,
        lid_k: int | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_lid", {})
        self._alpha_min = float(
            alpha_min if alpha_min is not None else cfg.get("alpha_min", 0.25)
        )
        self._scale = float(
            scale if scale is not None else cfg.get("scale", 0.75)
        )
        self._lid_sample_size = int(
            lid_sample_size if lid_sample_size is not None
            else cfg.get("lid_sample_size", 200)
        )
        self._lid_k = int(
            lid_k if lid_k is not None else cfg.get("lid_k", 10)
        )
        self._alpha: float = 1.0
        self.alpha_: float = 1.0
        self.lid_cv_: float = 0.0   # public: CV of LID used to set α
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    # ── LID CV estimation ────────────────────────────────────────────────────

    def _estimate_lid_cv(self, X: NDArray) -> float:
        """Coefficient of variation of Levina-Bickel LID over a subsample."""
        rng = np.random.default_rng(load_config().get("random_seed", 42))
        n = len(X)
        if n <= self._lid_k + 1:
            return 0.0

        idx = rng.choice(n, size=min(self._lid_sample_size, n), replace=False)
        sample = X[idx]

        D = cdist(sample, sample, metric="euclidean")
        np.fill_diagonal(D, np.inf)

        k = min(self._lid_k, len(sample) - 1)
        estimates = []
        for i in range(len(sample)):
            row = np.sort(D[i])[:k]
            est = self._estimate_lid(row)
            if np.isfinite(est) and est > 0:
                estimates.append(est)

        if len(estimates) < 2:
            return 0.0

        arr = np.array(estimates)
        mean = float(np.mean(arr))
        if mean <= 0:
            return 0.0
        return float(np.std(arr) / mean)

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankLID":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        super().fit(X_arr, y_arr)

        cv_lid = self._estimate_lid_cv(X_arr)
        alpha = float(np.clip(1.0 - self._scale * cv_lid, self._alpha_min, 1.0))

        self._alpha = alpha
        self.alpha_ = alpha
        self.lid_cv_ = cv_lid
        return self

    # ── Voting with LID-CV-derived α ─────────────────────────────────────────

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        k_eff = int(max(1, round(self._r ** self._alpha)))
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
