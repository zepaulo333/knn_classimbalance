"""Discriminant Adaptive Nearest Neighbours (DANN) — baseline.

Hastie & Tibshirani (1996).  The local metric is adapted using the
between- and within-class covariance of the k-neighbourhood so that the
distance function stretches toward discriminant directions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from joblib import Parallel, delayed

from src.utils.config import load_config


class DANN:
    """Discriminant Adaptive Nearest Neighbours classifier.

    Parameters
    ----------
    k : int
        Number of neighbours for classification.
    sigma : float
        Regularisation / neighbourhood size for local metric estimation.
    metric : str
        Base distance metric used before adaptation.
    """

    def __init__(
        self,
        k: int = 5,
        sigma: float = 1.0,
        metric: str = "euclidean",
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config()["dann"]
        self.k = k
        self.sigma = sigma
        self.metric = metric
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DANN":
        self._X_train = np.asarray(X, dtype=float)
        self._y_train = np.asarray(y)
        self.classes_ = np.unique(self._y_train)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self.n_jobs == 1:
            return np.array([self._predict_single(x) for x in X])
        return np.array(
            Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._predict_single)(x) for x in X
            )
        )

    def predict_proba(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self.n_jobs == 1:
            return np.array([self._predict_proba_single(x) for x in X])
        return np.array(
            Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._predict_proba_single)(x) for x in X
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _euclidean(self, x: NDArray) -> NDArray:
        diff = self._X_train - x
        return np.sqrt((diff * diff).sum(axis=1))

    def _local_metric(self, x: NDArray) -> NDArray:
        """Compute the DANN local metric matrix W at query point x."""
        dists = self._euclidean(x)
        # Use top-50 neighbours (or all) to estimate local metric
        n_local = min(50, len(self._X_train))
        local_idx = np.argsort(dists)[:n_local]
        X_local = self._X_train[local_idx]
        y_local = self._y_train[local_idx]

        p = self._X_train.shape[1]

        # Within-class scatter W
        W = np.zeros((p, p))
        for c in self.classes_:
            Xc = X_local[y_local == c]
            if len(Xc) < 2:
                continue
            centred = Xc - Xc.mean(axis=0)
            W += centred.T @ centred

        # Between-class scatter B
        overall_mean = X_local.mean(axis=0)
        B = np.zeros((p, p))
        for c in self.classes_:
            Xc = X_local[y_local == c]
            if len(Xc) == 0:
                continue
            diff = Xc.mean(axis=0) - overall_mean
            B += len(Xc) * np.outer(diff, diff)

        # W* = W + epsilon * I  (regularise)
        W_reg = W + self.sigma * np.eye(p)
        try:
            W_inv = np.linalg.inv(W_reg)
        except np.linalg.LinAlgError:
            return np.eye(p)

        # Metric: W^{-1} B W^{-1}  (simplified DANN)
        M = W_inv @ B @ W_inv
        return M

    def _dann_distances(self, x: NDArray) -> NDArray:
        M = self._local_metric(x)
        diff = self._X_train - x
        # Mahalanobis-style: sqrt(d^T M d)
        Md = diff @ M
        dists = np.sqrt(np.maximum((Md * diff).sum(axis=1), 0.0))
        return dists

    def _predict_single(self, x: NDArray):
        dists = self._dann_distances(x)
        idx = np.argsort(dists)[: self.k]
        neighbour_labels = self._y_train[idx]
        votes = {c: np.sum(neighbour_labels == c) for c in self.classes_}
        return max(votes, key=votes.get)

    def _predict_proba_single(self, x: NDArray) -> NDArray:
        dists = self._dann_distances(x)
        idx = np.argsort(dists)[: self.k]
        neighbour_labels = self._y_train[idx]
        counts = np.array([np.sum(neighbour_labels == c) for c in self.classes_], dtype=float)
        return counts / counts.sum()
