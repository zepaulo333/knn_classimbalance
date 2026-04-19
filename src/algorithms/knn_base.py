"""KNN base classes.

Core algorithm adapted from rushter/MLAlgorithms:
  https://github.com/rushter/MLAlgorithms/blob/master/mla/knn.py

Modifications over the original:
- sklearn-compatible fit/predict/predict_proba interface
- classes_ attribute set on fit
- predict_proba support (not in original)
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist, euclidean
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.utils.config import load_config


class KNNBase:
    """Base class for KNN variants.

    Core neighbor-finding logic adapted from rushter/MLAlgorithms.

    Parameters
    ----------
    k : int
        Number of neighbours. ``0`` means use all training examples.
    distance_func : callable
        A distance function ``f(a, b) -> float``. Any function from
        ``scipy.spatial.distance`` will do. Defaults to Euclidean.
    """

    def __init__(self, k: int = 5, distance_func=euclidean, n_jobs: int = 1) -> None:
        self.k = None if k == 0 else k
        self.distance_func = distance_func
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNBase":
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.classes_ = np.unique(self.y)
        return self

    def aggregate(self, neighbors_targets):
        raise NotImplementedError()

    def predict(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self.n_jobs == 1:
            return np.array([self._predict_x(x) for x in X])
        return np.array(
            Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._predict_x)(x) for x in X
            )
        )

    def predict_proba(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self.n_jobs == 1:
            return np.array([self._predict_proba_x(x) for x in X])
        return np.array(
            Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._predict_proba_x)(x) for x in X
            )
        )

    def _predict_x(self, x: NDArray):
        """Predict the label of a single instance.

        Adapted from rushter/MLAlgorithms KNNBase._predict_x.
        """
        distances = (self.distance_func(x, example) for example in self.X)
        neighbors = sorted(
            zip(distances, self.y),
            key=lambda item: item[0],
        )
        neighbors_targets = [target for (_, target) in neighbors[: self.k]]
        return self.aggregate(neighbors_targets)

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        distances = (self.distance_func(x, example) for example in self.X)
        neighbors = sorted(
            zip(distances, self.y),
            key=lambda item: item[0],
        )
        neighbors_targets = np.array([target for (_, target) in neighbors[: self.k]])
        counts = np.array(
            [np.sum(neighbors_targets == c) for c in self.classes_], dtype=float
        )
        total = counts.sum()
        return counts / total if total > 0 else counts


class KNNClassifier(KNNBase):
    """K-Nearest Neighbours classifier.

    Adapted from rushter/MLAlgorithms KNNClassifier.
    Ties are broken arbitrarily.
    """

    def aggregate(self, neighbors_targets):
        """Return the most common target label."""
        return Counter(neighbors_targets).most_common(1)[0][0]


class KNNClassifierFast(KNNClassifier):
    """KNN classifier with vectorised distance computation via scipy.cdist.

    Replaces the per-example ``scipy.euclidean`` loop from
    ``KNNBase._predict_x`` with a single ``scipy.cdist`` call, which
    computes all distances in one C-level operation using the same
    algorithm as ``scipy.euclidean`` — so predictions are bit-for-bit
    identical to ``KNNClassifier``, only execution speed differs.

    This is our first modification of the rushter/MLAlgorithms base code.
    """

    def _predict_x(self, x: NDArray):
        dists = cdist(self.X, x.reshape(1, -1), metric='euclidean').ravel()
        idx = np.argsort(dists)[: self.k]
        return self.aggregate(self.y[idx].tolist())

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        dists = cdist(self.X, x.reshape(1, -1), metric='euclidean').ravel()
        idx = np.argsort(dists)[: self.k]
        neighbors = self.y[idx]
        counts = np.array([np.sum(neighbors == c) for c in self.classes_], dtype=float)
        total = counts.sum()
        return counts / total if total > 0 else counts


class KNNOptK:
    """KNN with k selected by inner cross-validation — the proper baseline.

    For each call to ``fit``, runs a stratified inner CV over a data-driven
    range of odd k values from 1 to sqrt(n_train), scoring by balanced
    accuracy (appropriate for imbalanced problems).  The best k is then used
    to fit a final ``KNNClassifierFast`` on the full training set.

    The k range is derived from the training data at fit time, so it scales
    automatically with dataset size instead of relying on a hardcoded list.

    Parameters
    ----------
    k_max : int or None
        Upper bound on k. ``None`` means use ``floor(sqrt(n_train))``.
    inner_cv_folds : int
        Number of folds for the inner CV used to select k.
    """

    def __init__(
        self,
        k_max: int | None = None,
        inner_cv_folds: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        cfg = load_config()["knn_opt_k"]
        self.k_max = k_max if k_max is not None else cfg.get("k_max", None)
        self.inner_cv_folds = inner_cv_folds if inner_cv_folds is not None else cfg["inner_cv_folds"]
        self.n_jobs = n_jobs if n_jobs is not None else cfg.get("n_jobs", 8)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNOptK":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        seed = load_config()["random_seed"]
        cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=seed)

        # Derive k range from data: odd values 1..sqrt(n), capped at k_max
        k_upper = self.k_max if self.k_max is not None else max(1, int(np.sqrt(len(X))))
        k_range = [k for k in range(1, k_upper + 1, 2)]  # odd values only (avoids ties)
        if not k_range:
            k_range = [1]
        self.k_range_ = k_range  # expose for inspection

        def _eval_k(X_tr, X_val, y_tr, y_val, k):
            if k >= len(X_tr):
                return k, None
            clf = KNNClassifierFast(k=k)
            clf.fit(X_tr, y_tr)
            return k, balanced_accuracy_score(y_val, clf.predict(X_val))

        splits = [
            (X[tr], X[val], y[tr], y[val])
            for tr, val in cv.split(X, y)
        ]
        jobs = [
            (X_tr, X_val, y_tr, y_val, k)
            for X_tr, X_val, y_tr, y_val in splits
            for k in k_range
        ]
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_eval_k)(X_tr, X_val, y_tr, y_val, k)
            for X_tr, X_val, y_tr, y_val, k in jobs
        )

        k_scores: dict[int, list[float]] = {k: [] for k in k_range}
        for k, score in results:
            if score is not None:
                k_scores[k].append(score)

        valid = {k: np.mean(v) for k, v in k_scores.items() if v}
        self.best_k_ = max(valid, key=valid.get)

        self._clf = KNNClassifierFast(k=self.best_k_)
        self._clf.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        return self._clf.predict(X)

    def predict_proba(self, X: ArrayLike) -> NDArray:
        return self._clf.predict_proba(X)
