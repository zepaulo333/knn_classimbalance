"""KNNFairRank variant with cross-validated correction strength
(Modification C from Section 8.9).

The base KNNFairRank derives k_eff = r from first principles, assuming
a Poisson-uniform density for the two classes. When that assumption is
violated (clustered minorities, non-uniform local density), k_eff = r
can over- or under-correct on specific datasets.

This variant replaces the fixed k_eff = r with

    k_eff = round(r^α)

and selects α via an inner stratified K-fold CV on the training data,
optimising a configurable scoring metric (default: G-mean). The α grid
is small and deliberately biased toward α = 1 (the theoretical value):

    α ∈ {0.25, 0.5, 0.75, 1.0}

Interpretation of α:
    α = 0    → k_eff = 1 → no rank correction (standard 1-NN style).
    α = 0.5  → k_eff = √r → square-root damping.
    α = 1    → k_eff = r → v3 theoretical value.

Analogous to KNNOptK's treatment of k: it converts a principled but
possibly-over-confident theoretical choice into a data-driven one, at
the cost of introducing one hyperparameter and an inner-CV overhead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from src.algorithms.knn_fair_rank import KNNFairRank
from src.evaluation.metrics import geometric_mean
from src.utils.config import load_config


_SCORERS = {
    "geometric_mean": geometric_mean,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": lambda y, yp: f1_score(y, yp, zero_division=0),
    "mcc": matthews_corrcoef,
    "combined": lambda y, yp: (matthews_corrcoef(y, yp) + geometric_mean(y, yp)) / 2,
}


class KNNFairRankCV(KNNFairRank):
    """KNNFairRank with α tuned via inner stratified K-fold CV.

    Parameters
    ----------
    alpha_grid : list[float] or None
        Candidate exponents. None → value from settings.yaml.
    inner_cv_folds : int or None
        Folds for the inner CV. None → value from settings.yaml.
    scoring : str or None
        One of {"geometric_mean", "balanced_accuracy", "f1"}. None →
        value from settings.yaml.
    """

    def __init__(
        self,
        alpha_grid: list[float] | None = None,
        inner_cv_folds: int | None = None,
        scoring: str | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_cv", {})
        self._alpha_grid = list(alpha_grid) if alpha_grid is not None else list(
            cfg.get("alpha_grid", [0.25, 0.5, 0.75, 1.0])
        )
        self._inner_cv_folds = (
            inner_cv_folds if inner_cv_folds is not None else cfg.get("inner_cv_folds", 3)
        )
        self._scoring = scoring if scoring is not None else cfg.get("scoring", "geometric_mean")
        if self._scoring not in _SCORERS:
            raise ValueError(
                f"Unknown scoring '{self._scoring}'. Valid: {list(_SCORERS)}"
            )
        self._alpha: float = 1.0  # selected during fit()
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    # ── Inner CV selection ───────────────────────────────────────────────────

    def _fit_fold(
        self, X_tr: NDArray, y_tr: NDArray, X_va: NDArray, y_va: NDArray, alpha: float
    ) -> float:
        clf = KNNFairRank(
            k_min=self._k_min,
            k_maj_buffer=self._k_maj_buffer,
            k_maj_floor=self._k_maj_floor,
            k_maj_cap=self._k_maj_cap,
            n_votes=self._n_votes,
            n_jobs=1,
        )
        clf.fit(X_tr, y_tr)
        # Dial the effective r without changing the fitted neighbourhoods.
        # k_maj was sized with α=1 (i.e. k_maj ≥ r·n_votes + buffer), so
        # for α ≤ 1 we have strictly more majority neighbours than we
        # need — no under-sizing risk.
        clf._r = clf._r ** alpha
        return _SCORERS[self._scoring](y_va, clf.predict(X_va))

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankCV":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        seed = load_config()["random_seed"]
        cv = StratifiedKFold(n_splits=self._inner_cv_folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X_arr, y_arr))

        # Parallelise over all (alpha, fold) pairs.
        work = [(alpha, tr, va) for alpha in self._alpha_grid for tr, va in splits]

        if self.n_jobs == 1:
            fold_scores = [
                self._fit_fold(X_arr[tr], y_arr[tr], X_arr[va], y_arr[va], alpha)
                for alpha, tr, va in work
            ]
        else:
            fold_scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._fit_fold)(X_arr[tr], y_arr[tr], X_arr[va], y_arr[va], alpha)
                for alpha, tr, va in work
            )

        # Average fold scores per alpha, pick the best.
        alpha_scores: dict[float, list[float]] = {}
        for (alpha, _, _), s in zip(work, fold_scores):
            alpha_scores.setdefault(alpha, []).append(s)

        best_alpha = max(alpha_scores, key=lambda a: float(np.mean(alpha_scores[a])))
        self._alpha = best_alpha
        self.best_alpha_ = best_alpha  # public alias for inspection

        # Fit the outer model on all training data; _vote_fraction uses α.
        super().fit(X_arr, y_arr)
        return self

    # ── Voting with α-adjusted k_eff ─────────────────────────────────────────

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
