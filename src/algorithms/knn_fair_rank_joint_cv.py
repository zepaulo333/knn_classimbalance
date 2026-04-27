"""KNNFairRank with n_votes and α selected jointly by inner CV.

KNNFairRankCV selects α (correction strength) with n_votes fixed.
KNNFairRankOptVotes selects n_votes with α fixed at 1.
This variant selects both jointly over the full (n_votes, α) grid.

The key efficiency trick (borrowed from KNNFairRankCV._score_alpha):
α only affects prediction, not the neighbourhood structure built at fit
time.  After fitting, applying a different α is just patching clf._r:

    clf._r = original_r ** alpha   →   k_eff = round(r^α)

So we fit once per (fold, n_votes) and sweep all α values on the same
fitted model.  Cost:

    fits       : n_votes_grid × cv_folds          (same as OptVotes)
    predictions: n_votes_grid × cv_folds × alpha_grid  (cheap numpy ops)

This makes joint CV nearly free compared to running OptVotes alone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from joblib import Parallel, delayed
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.algorithms.knn_fair_rank import KNNFairRank
from src.algorithms.knn_fair_rank_c import KNNFairRankCV
from src.evaluation.metrics import geometric_mean
from src.utils.config import load_config


_SCORERS = {
    "geometric_mean":    geometric_mean,
    "balanced_accuracy": balanced_accuracy_score,
    "f1":                lambda y, yp: f1_score(y, yp, zero_division=0),
    "mcc":               matthews_corrcoef,
    "combined":          lambda y, yp: (matthews_corrcoef(y, yp) + geometric_mean(y, yp)) / 2,
}


class KNNFairRankJointCV(KNNFairRankCV):
    """KNNFairRank with n_votes and α selected jointly by inner stratified k-fold CV.

    Parameters
    ----------
    n_votes_grid : list[int] or None
        Candidate vote counts.  None → value from settings.yaml.
    alpha_grid : list[float] or None
        Candidate correction exponents (k_eff = round(r^α)).
        None → value from settings.yaml.
    inner_cv_folds : int or None
        Folds for the inner CV.  None → value from settings.yaml.
    scoring : str or None
        Metric used to select the joint optimum.  None → settings.yaml.

    Attributes
    ----------
    best_n_votes_ : int
    best_alpha_ : float
    """

    def __init__(
        self,
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
        cfg = load_config().get("knn_fair_rank_joint_cv", {})
        self._n_votes_grid = list(n_votes_grid) if n_votes_grid is not None else list(
            cfg.get("n_votes_grid", [1, 2, 3, 5, 7, 10])
        )
        _alpha_grid = list(alpha_grid) if alpha_grid is not None else list(
            cfg.get("alpha_grid", [0.25, 0.5, 0.75, 1.0])
        )
        self.best_n_votes_: int = self._n_votes_grid[0]
        # Pass the largest n_votes so k_maj is sized generously enough for all
        # candidates (same trick as KNNFairRankOptVotes).
        super().__init__(
            alpha_grid=_alpha_grid,
            inner_cv_folds=inner_cv_folds,
            scoring=scoring,
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=max(self._n_votes_grid),
            n_jobs=n_jobs,
        )

    # ── Inner CV ─────────────────────────────────────────────────────────────

    def _fit_fold(
        self, X_tr: NDArray, y_tr: NDArray, X_va: NDArray, y_va: NDArray, n_votes: int
    ) -> dict[float, float]:
        """Fit one (n_votes, fold) pair and return scores for all α values."""
        score_fn = _SCORERS[self._scoring]
        clf = KNNFairRank(
            k_min=self._k_min,
            k_maj_buffer=self._k_maj_buffer,
            k_maj_floor=self._k_maj_floor,
            k_maj_cap=self._k_maj_cap,
            n_votes=n_votes,
            n_jobs=1,
        )
        clf.fit(X_tr, y_tr)
        original_r = clf._r
        result = {}
        for alpha in self._alpha_grid:
            clf._r = original_r ** alpha
            result[alpha] = score_fn(y_va, clf.predict(X_va))
        clf._r = original_r
        return result  # {alpha: score}

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankJointCV":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        seed = load_config()["random_seed"]
        cv = StratifiedKFold(n_splits=self._inner_cv_folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X_arr, y_arr))

        # Parallelise over all (n_votes, fold) pairs — 18 work items for 16 cores
        work = [(nv, tr, va) for nv in self._n_votes_grid for tr, va in splits]

        if self.n_jobs == 1:
            fold_results = [
                self._fit_fold(X_arr[tr], y_arr[tr], X_arr[va], y_arr[va], nv)
                for nv, tr, va in work
            ]
        else:
            fold_results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._fit_fold)(X_arr[tr], y_arr[tr], X_arr[va], y_arr[va], nv)
                for nv, tr, va in work
            )

        # Aggregate: average over folds for each (n_votes, alpha) pair
        joint_scores: dict[tuple[int, float], list[float]] = {}
        for (nv, _, _), alpha_dict in zip(work, fold_results):
            for alpha, score in alpha_dict.items():
                joint_scores.setdefault((nv, alpha), []).append(score)

        best_nv, best_alpha = max(
            joint_scores, key=lambda k: float(np.mean(joint_scores[k]))
        )
        self.best_n_votes_ = best_nv
        self.best_alpha_ = best_alpha
        self._n_votes = best_nv   # used by _vote_fraction at inference time
        self._alpha = best_alpha  # used by KNNFairRankCV._vote_fraction

        # Fit the final model directly, bypassing KNNFairRankCV.fit() which
        # would re-run its own CV selection.
        KNNFairRank.fit(self, X_arr, y_arr)
        return self
