"""KNNFairRank variant with n_votes selected by inner CV.

The base algorithm uses a fixed n_votes (default 5), but the empirical
n_votes sweep reveals a strongly bimodal optimal distribution across the
benchmark:

  - 28/45 datasets peak at n_votes ∈ {1, 2}   (few-votes regime)
  -  5/45 datasets peak at n_votes ∈ {3, 5}   (transition)
  - 12/45 datasets peak at n_votes ∈ {7, 10}  (many-votes regime)

n=5 is near-optimal for only 7/45 datasets — it sits in the dead zone
between the two clusters.  No single fixed value covers both regimes.

This variant selects n_votes from a small candidate grid via inner
stratified k-fold CV, exactly as KNNFairRankCV selects α.  The overhead
is modest: distance computation (_per_class_distances / cdist) is the
expensive step, and it is controlled entirely by k_min and k_maj_eff,
both of which are fixed at the start of each inner-fold fit.  Evaluating
different n_votes values amounts to indexing into already-sorted arrays,
so the inner CV pays only for extra fit() calls, not extra cdist calls.

The k_maj neighbourhood is sized for the largest candidate in n_votes_grid
so that all candidates can be evaluated within a single fit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from joblib import Parallel, delayed
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.algorithms.knn_fair_rank import KNNFairRank
from src.evaluation.metrics import geometric_mean
from src.utils.config import load_config


_SCORERS = {
    "geometric_mean":     geometric_mean,
    "balanced_accuracy":  balanced_accuracy_score,
    "f1":                 lambda y, yp: f1_score(y, yp, zero_division=0),
    "mcc":                matthews_corrcoef,
    "combined":           lambda y, yp: (matthews_corrcoef(y, yp) + geometric_mean(y, yp)) / 2,
}


class KNNFairRankOptVotes(KNNFairRank):
    """KNNFairRank with n_votes selected by inner stratified k-fold CV.

    Parameters
    ----------
    n_votes_grid : list[int] or None
        Candidate vote counts.  None → value from settings.yaml.
    inner_cv_folds : int or None
        Folds for the inner CV.  None → value from settings.yaml.
    scoring : str or None
        Metric used to select n_votes.  One of {geometric_mean,
        balanced_accuracy, f1, mcc, combined}.  None → settings.yaml.

    All other parameters are forwarded to KNNFairRank unchanged.

    Attributes
    ----------
    best_n_votes_ : int
        The n_votes selected by inner CV.  Available after fit().
    """

    def __init__(
        self,
        n_votes_grid: list[int] | None = None,
        inner_cv_folds: int | None = None,
        scoring: str | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_opt_votes", {})
        self._n_votes_grid = list(n_votes_grid) if n_votes_grid is not None else list(
            cfg.get("n_votes_grid", [1, 2, 3, 5, 7, 10])
        )
        self._inner_cv_folds = (
            inner_cv_folds if inner_cv_folds is not None else cfg.get("inner_cv_folds", 3)
        )
        self._scoring = scoring if scoring is not None else cfg.get("scoring", "geometric_mean")
        if self._scoring not in _SCORERS:
            raise ValueError(f"Unknown scoring '{self._scoring}'. Valid: {list(_SCORERS)}")
        self.best_n_votes_: int = self._n_votes_grid[0]
        # Pass the largest grid value as n_votes so k_maj is sized generously
        # enough for all candidates.  The inner CV will dial this down as needed.
        super().__init__(
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
    ) -> float:
        """Fit one (n_votes, fold) pair and return its score."""
        clf = KNNFairRank(
            k_min=self._k_min,
            k_maj_buffer=self._k_maj_buffer,
            k_maj_floor=self._k_maj_floor,
            k_maj_cap=self._k_maj_cap,
            n_votes=n_votes,
            n_jobs=1,
        )
        clf.fit(X_tr, y_tr)
        return _SCORERS[self._scoring](y_va, clf.predict(X_va))

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankOptVotes":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        seed = load_config()["random_seed"]
        cv = StratifiedKFold(n_splits=self._inner_cv_folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X_arr, y_arr))

        # Parallelise over all (n_votes, fold) pairs — n_votes_grid × folds work items
        work = [(nv, tr, va) for nv in self._n_votes_grid for tr, va in splits]

        if self.n_jobs == 1:
            fold_scores = [
                self._fit_fold(X_arr[tr], y_arr[tr], X_arr[va], y_arr[va], nv)
                for nv, tr, va in work
            ]
        else:
            fold_scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._fit_fold)(X_arr[tr], y_arr[tr], X_arr[va], y_arr[va], nv)
                for nv, tr, va in work
            )

        # Average fold scores per n_votes
        nv_scores: dict[int, list[float]] = {}
        for (nv, _, _), s in zip(work, fold_scores):
            nv_scores.setdefault(nv, []).append(s)

        best_nv = max(nv_scores, key=lambda nv: float(np.mean(nv_scores[nv])))
        self.best_n_votes_ = best_nv
        self._n_votes = best_nv      # used by _vote_fraction at inference time
        return super().fit(X_arr, y_arr)
