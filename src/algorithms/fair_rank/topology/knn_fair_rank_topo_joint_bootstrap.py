"""KNNFairRankTopoJoint with OOB-validated topology reliability gating.

The previous bootstrap stability approach measured whether topology fires
*consistently* across resamples. That detects sampling noise but cannot detect
topology that is stably wrong — Ward finds the same incorrect partition every
time, so stability is high even when the correction hurts minority recall.

This version uses Out-Of-Bag (OOB) evaluation to directly measure whether the
topology correction produces *better predictions* than a reference correction
on held-out training points.

For each bootstrap b, ~37% of the original training points are left out (OOB).
The bootstrap model — trained without those points — predicts them without
overfitting. For each OOB point we compare:
  - Topology prediction: uses the bootstrap model's per-region k_eff.
  - Baseline prediction: same bootstrap NN structure, but with k_eff fixed to
    r^oob_baseline_alpha (a stable reference, default alpha=0.75).

The differential score (topology_correct − baseline_correct) ∈ {−1, 0, +1}
aggregated over bootstraps gives each training point an OOB reliability score:
  +  topology consistently beats baseline near this point → use topology
  −  baseline consistently beats topology → fall back to KNNFairRankJointCV
  0  tie or no OOB evaluations → use topology (neutral default)

Both predictions are computed in a single vectorised batch using cdist, avoiding
the O(n_b × d) per-point loop that made the naive predict() call prohibitively
slow for high-dimensional datasets.

JointCV is only fitted when at least one training point has negative OOB
reliability — datasets where topology is reliably good everywhere skip it.

Theory: exploration.ipynb §23.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

from src.algorithms.fair_rank.topology.knn_fair_rank_topo_joint import KNNFairRankTopoJoint
from src.algorithms.fair_rank.ensemble.knn_fair_rank_joint_cv import KNNFairRankJointCV
from src.utils.config import load_config


def _predict_oob_batch(
    model_b: KNNFairRankTopoJoint,
    X_oob: NDArray,
    alpha_baseline: float,
) -> tuple[NDArray, NDArray]:
    """Vectorised OOB predictions for topology and baseline.

    Replaces two calls to model_b.predict(X_oob), each of which looped over
    every OOB point calling _vote_fraction (O(n_b × d) per point). This
    function computes all pairwise distances in two cdist calls and applies
    the voting logic with numpy operations only.

    Uses n_votes=1 for speed. Both topology and baseline use the same
    simplified rule so the comparison remains fair.

    Returns (topo_preds, baseline_preds) as int8 arrays of shape (n_oob,).
    """
    n_oob = len(X_oob)

    if len(model_b._X_min) == 0:
        return np.zeros(n_oob, np.int8), np.zeros(n_oob, np.int8)
    if len(model_b._X_maj) == 0:
        return np.ones(n_oob, np.int8), np.ones(n_oob, np.int8)

    # ── Batch pairwise distances ──────────────────────────────────────────────
    D_min = cdist(X_oob, model_b._X_min)   # (n_oob, n_min_b)
    D_maj = cdist(X_oob, model_b._X_maj)   # (n_oob, n_maj_b)
    n_maj_b = D_maj.shape[1]

    # Nearest minority distance for each OOB point (used by both branches).
    d_min_1 = D_min.min(axis=1)            # (n_oob,)

    # Sort majority distances once — reused for both topology and baseline.
    D_maj_sorted = np.sort(D_maj, axis=1)  # (n_oob, n_maj_b)

    # ── Topology: k_eff varies per OOB point based on nearest region ──────────
    min_near_dist = D_min.min(axis=1)
    maj_near_dist = D_maj.min(axis=1)
    min_near_j    = np.argmin(D_min, axis=1)
    maj_near_j    = np.argmin(D_maj, axis=1)

    min_orig = np.where(model_b.y == model_b._minority_class)[0]
    maj_orig = np.where(model_b.y == model_b._majority_class)[0]

    nearest_orig = np.where(
        min_near_dist <= maj_near_dist,
        min_orig[min_near_j],
        maj_orig[maj_near_j],
    )
    nearest_regions = model_b._point_region[nearest_orig]
    k_effs = np.array(
        [model_b._region_k_eff[int(r)] for r in nearest_regions], dtype=float
    )
    k_effs = np.clip(np.round(k_effs).astype(int), 1, n_maj_b)

    d_maj_k_topo  = D_maj_sorted[np.arange(n_oob), k_effs - 1]
    topo_preds    = (d_min_1 < d_maj_k_topo).astype(np.int8)

    # ── Baseline: uniform k_eff = round(r^alpha_baseline) ────────────────────
    k_base        = int(np.clip(round(model_b._r ** alpha_baseline), 1, n_maj_b))
    d_maj_k_base  = D_maj_sorted[:, k_base - 1]
    baseline_preds = (d_min_1 < d_maj_k_base).astype(np.int8)

    return topo_preds, baseline_preds


def _oob_worker(
    X_orig: NDArray,
    y_orig: NDArray,
    bootstrap_idx: NDArray,
    topo_params: dict,
    alpha_baseline: float,
) -> tuple[NDArray, NDArray] | None:
    """Fit one bootstrap topology model and evaluate it on the OOB set.

    Returns (oob_idx, scores) where scores[i] ∈ {-1, 0, +1}:
      +1  topology correct, baseline wrong
       0  both right or both wrong
      -1  baseline correct, topology wrong

    Returns None when topology fell back on this bootstrap (n_regions == 1).
    """
    X_b = X_orig[bootstrap_idx]
    y_b = y_orig[bootstrap_idx]

    model_b = KNNFairRankTopoJoint(**topo_params)
    model_b.fit(X_b, y_b)

    if model_b.n_regions_ == 1:
        return None

    in_bootstrap = np.zeros(len(X_orig), dtype=bool)
    in_bootstrap[np.unique(bootstrap_idx)] = True
    oob_idx = np.where(~in_bootstrap)[0]

    if len(oob_idx) == 0:
        return None

    topo_preds, baseline_preds = _predict_oob_batch(
        model_b, X_orig[oob_idx], alpha_baseline
    )

    y_oob            = y_orig[oob_idx]
    topo_correct     = (topo_preds     == y_oob).astype(np.int8)
    baseline_correct = (baseline_preds == y_oob).astype(np.int8)

    return oob_idx, topo_correct - baseline_correct


class KNNFairRankTopoJointBootstrap(KNNFairRankTopoJoint):
    """KNNFairRankTopoJoint with OOB reliability gating and JointCV fallback.

    Parameters
    ----------
    n_bootstrap : int or None
        Bootstrap resamples for OOB reliability estimation.
        None → settings.yaml (default 20).
    oob_baseline_alpha : float or None
        Exponent for the OOB comparison baseline: k_eff = r^alpha.
        Decoupled from JointCV so the OOB comparison never requires fitting
        JointCV on every fold. None → settings.yaml (default 0.75, the most
        common JointCV selection across the benchmark).
    fallback_alpha : float or None
        Pre-computed JointCV alpha for the inference fallback. When provided
        together with fallback_n_votes, the internal JointCV fit is skipped.
    fallback_n_votes : int or None
        Pre-computed JointCV n_votes for the inference fallback.

    All KNNFairRankTopoJoint parameters are accepted and forwarded.

    Attributes
    ----------
    oob_reliability_ : NDArray of shape (n_train,)
        Per-training-point mean OOB differential score ∈ [−1, +1].
        Positive → topology outperformed baseline on OOB samples near this
        point. Negative → baseline won.
    fallback_alpha_ : float
        Alpha used for the JointCV inference fallback (when fitted).
    fallback_n_votes_ : int
        n_votes for the inference fallback.
    """

    def __init__(
        self,
        n_bootstrap: int | None = None,
        oob_baseline_alpha: float | None = None,
        fallback_alpha: float | None = None,
        fallback_n_votes: int | None = None,
        min_persistence_ratio: float | None = None,
        laplace_smooth: float | None = None,
        min_region_samples: int | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_topo_joint_bootstrap", {})
        self._n_bootstrap: int = (
            n_bootstrap if n_bootstrap is not None
            else int(cfg.get("n_bootstrap", 20))
        )
        self._oob_baseline_alpha: float = (
            oob_baseline_alpha if oob_baseline_alpha is not None
            else float(cfg.get("oob_baseline_alpha", 0.75))
        )
        self._preset_fallback_alpha: float | None   = fallback_alpha
        self._preset_fallback_n_votes: int | None   = fallback_n_votes

        self.oob_reliability_: NDArray  = np.zeros(0)
        self.fallback_alpha_: float     = 1.0
        self.fallback_n_votes_: int     = 1

        super().__init__(
            min_persistence_ratio=min_persistence_ratio,
            laplace_smooth=laplace_smooth,
            min_region_samples=min_region_samples,
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankTopoJointBootstrap":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        # ── 1. Fit base TopoJoint on the full training fold ───────────────────
        super().fit(X_arr, y_arr)
        n = len(self.X)

        # ── 2. Bootstrap OOB evaluation ───────────────────────────────────────
        seed = load_config().get("random_seed", 42)
        rng  = np.random.RandomState(seed)

        boot_indices = [
            self._stratified_bootstrap(y_arr, rng) for _ in range(self._n_bootstrap)
        ]

        topo_params = dict(
            min_persistence_ratio = self._min_persistence_ratio,
            laplace_smooth        = self._laplace_smooth,
            min_region_samples    = self._min_region_samples,
            k_min                 = self._k_min,
            k_maj_buffer          = self._k_maj_buffer,
            k_maj_floor           = self._k_maj_floor,
            k_maj_cap             = self._k_maj_cap,
            n_votes               = self._n_votes,
            n_jobs                = 1,
        )

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_oob_worker)(
                self.X, y_arr, idx, topo_params, self._oob_baseline_alpha
            )
            for idx in boot_indices
        )

        # ── 3. Aggregate per-point OOB differential reliability ───────────────
        score_sums = np.zeros(n, dtype=float)
        oob_counts = np.zeros(n, dtype=float)

        for result in results:
            if result is None:
                continue
            oob_idx, scores = result
            score_sums[oob_idx] += scores.astype(float)
            oob_counts[oob_idx] += 1.0

        self.oob_reliability_ = np.where(
            oob_counts > 0,
            score_sums / np.maximum(oob_counts, 1.0),
            0.0,
        )

        # ── 4. Fit JointCV only if the fallback will actually be used ─────────
        needs_fallback = bool((self.oob_reliability_ < 0.0).any())
        if needs_fallback:
            if (self._preset_fallback_alpha is not None
                    and self._preset_fallback_n_votes is not None):
                self.fallback_alpha_   = float(self._preset_fallback_alpha)
                self.fallback_n_votes_ = int(self._preset_fallback_n_votes)
            else:
                joint_cv = KNNFairRankJointCV(n_jobs=self.n_jobs)
                joint_cv.fit(X_arr, y_arr)
                self.fallback_alpha_   = joint_cv._alpha
                self.fallback_n_votes_ = joint_cv._n_votes
        return self

    @staticmethod
    def _stratified_bootstrap(y: NDArray, rng: np.random.RandomState) -> NDArray:
        indices = []
        for cls in np.unique(y):
            pool = np.where(y == cls)[0]
            indices.append(rng.choice(pool, size=len(pool), replace=True))
        return np.concatenate(indices)

    # ── prediction ───────────────────────────────────────────────────────────

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        diff        = self.X - x
        nearest_idx = int(np.argmin((diff * diff).sum(axis=1)))

        if self.oob_reliability_[nearest_idx] >= 0.0:
            region  = int(self._point_region[nearest_idx])
            k_eff   = int(max(1, round(self._region_k_eff[region])))
            n_votes = self._n_votes
        else:
            k_eff   = int(max(1, round(self._r ** self.fallback_alpha_)))
            n_votes = self.fallback_n_votes_

        max_votes_maj = len(d_maj) // k_eff
        n_votes = min(n_votes, len(d_min), max_votes_maj)
        if n_votes < 1:
            n_votes = 1
            k_eff   = min(k_eff, len(d_maj))

        min_refs    = d_min[:n_votes]
        maj_indices = np.arange(1, n_votes + 1) * k_eff - 1
        maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
        return int(np.sum(min_refs < d_maj[maj_indices])) / n_votes, n_votes
