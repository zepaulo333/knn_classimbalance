"""Combined Modification B + C variant (algorithm_design.ipynb §6.5).

Inherits the inner-CV α-tuning from `KNNFairRankCV` (Modification C) and
the magnitude-aware soft voting from `KNNFairRankMagnitude`
(Modification B). The two modifications are independent: B replaces the
binary vote with a continuous score, C dials the rank-correction
strength via an exponent α tuned by inner CV. Section 6.5 flags this
combination as the natural follow-up since B already wins ROC AUC and
C already wins G-mean in isolation.

The inner CV that selects α uses the magnitude-aware voting too — it
would be inconsistent to pick α with binary votes and then deploy with
soft votes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from src.algorithms.knn_fair_rank_b import KNNFairRankMagnitude
from src.algorithms.knn_fair_rank_c import KNNFairRankCV, _SCORERS
from src.utils.config import load_config


class KNNFairRankMagnitudeCV(KNNFairRankCV):
    """Magnitude-aware votes (B) with α tuned by inner CV (C)."""

    def _score_alpha(self, X: NDArray, y: NDArray, alpha: float) -> float:
        """Same trick as the parent: fit once with α=1's k_maj sizing, then
        rescale `_r` so the magnitude-aware vote uses k_eff = round(r^α)."""
        seed = load_config()["random_seed"]
        cv = StratifiedKFold(n_splits=self._inner_cv_folds, shuffle=True, random_state=seed)
        score_fn = _SCORERS[self._scoring]
        scores = []
        for tr, va in cv.split(X, y):
            clf = KNNFairRankMagnitude(
                k_min=self._k_min,
                k_maj_buffer=self._k_maj_buffer,
                k_maj_floor=self._k_maj_floor,
                k_maj_cap=self._k_maj_cap,
                n_votes=self._n_votes,
                n_jobs=1,
            )
            clf.fit(X[tr], y[tr])
            clf._r = clf._r ** alpha
            y_pred = clf.predict(X[va])
            scores.append(score_fn(y[va], y_pred))
        return float(np.mean(scores)) if scores else -np.inf

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

        denom = min_refs + maj_refs
        safe = denom > 0
        scores = np.where(safe, maj_refs / np.where(safe, denom, 1.0), 0.5)
        return float(np.mean(scores)), n_votes
