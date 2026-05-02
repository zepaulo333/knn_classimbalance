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

All (n_votes, fold) pairs are dispatched as independent parallel work items
so all available cores stay busy.  This makes joint CV nearly free compared
to running OptVotes alone.
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
        mcc_weight: float | None = None,
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
        super().__init__(
            alpha_grid=_alpha_grid,
            inner_cv_folds=inner_cv_folds,
            scoring=scoring,
            mcc_weight=mcc_weight,
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
    ) -> "dict[float, float] | dict[float, tuple[float, float]]":
        """Fit one (n_votes, fold) pair and return scores for all α values."""
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
            preds = clf.predict(X_va)
            if self._scoring == "utopia":
                result[alpha] = (matthews_corrcoef(y_va, preds), geometric_mean(y_va, preds))
            else:
                result[alpha] = self._scorer_fn(y_va, preds)
        clf._r = original_r
        return result

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankJointCV":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        seed = load_config()["random_seed"]
        cv = StratifiedKFold(n_splits=self._inner_cv_folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X_arr, y_arr))

        work = [(nv, tr, va) for nv in self._n_votes_grid for tr, va in splits]

        if self.n_jobs == 1:
            fold_results = [
                self._fit_fold(X_arr[tr], y_arr[tr], X_arr[va], y_arr[va], nv)
                for nv, tr, va in work
            ]
        else:
            fold_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_fold)(X_arr[tr], y_arr[tr], X_arr[va], y_arr[va], nv)
                for nv, tr, va in work
            )

        joint_scores: dict[tuple[int, float], list] = {}
        for (nv, _, _), alpha_dict in zip(work, fold_results):
            for alpha, score in alpha_dict.items():
                joint_scores.setdefault((nv, alpha), []).append(score)

        if self._scoring == "utopia":
            # Each entry is (mcc, gmean); select (nv, α) closest to the Utopia point.
            pair_means = {
                k: (float(np.mean([s[0] for s in ss])),
                    float(np.mean([s[1] for s in ss])))
                for k, ss in joint_scores.items()
            }
            u_mcc   = max(v[0] for v in pair_means.values())
            u_gmean = max(v[1] for v in pair_means.values())
            eps = 1e-9
            best_nv, best_alpha = min(
                pair_means,
                key=lambda k: (
                    ((pair_means[k][0] - u_mcc)   / (u_mcc   + eps)) ** 2 +
                    ((pair_means[k][1] - u_gmean) / (u_gmean + eps)) ** 2
                ),
            )
        else:
            best_nv, best_alpha = max(
                joint_scores, key=lambda k: float(np.mean(joint_scores[k]))
            )

        self.best_n_votes_ = best_nv
        self.best_alpha_ = best_alpha
        self._n_votes = best_nv
        self._alpha = best_alpha

        KNNFairRank.fit(self, X_arr, y_arr)
        return self
