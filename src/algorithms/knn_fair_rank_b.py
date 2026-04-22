"""KNNFairRank variant with magnitude-aware voting (Modification B).

Replaces the binary vote 1[d_i^min < d_{i·k_eff}^maj] with a continuous
confidence score

    s_i = d_{i·k_eff}^maj / (d_i^min + d_{i·k_eff}^maj)  ∈ [0, 1]

The mean of the s_i over i = 1..n_votes is returned as the minority
score; the hard decision still predicts minority iff mean(s_i) > 0.5.
Probability estimates come out naturally calibrated in [0, 1] rather
than quantised to {i/n_votes : i = 0..n_votes} as in the binary vote.

Empirically this is the only variant of the Sec. 8.9 ablation that
produces a statistically significant improvement — a roughly +1.6 point
ROC AUC gain over v3 (Holm-corrected Wilcoxon, p ≈ 2e-4). Point-estimate
metrics (F1, BA, G-mean) are within noise of v3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.algorithms.knn_fair_rank import KNNFairRank


class KNNFairRankMagnitude(KNNFairRank):
    """KNNFairRank with continuous-score magnitude-aware voting."""

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        k_eff = int(max(1, round(self._r)))
        max_votes_maj = len(d_maj) // k_eff
        n_votes = min(self._n_votes, len(d_min), max_votes_maj)
        if n_votes < 1:
            n_votes = 1
            k_eff = min(k_eff, len(d_maj))

        min_refs = d_min[:n_votes]
        maj_indices = np.arange(1, n_votes + 1) * k_eff - 1
        maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
        maj_refs = d_maj[maj_indices]

        # Continuous score. Guard against both distances = 0 (x coincides
        # with training points of both classes, numerically unlikely but
        # possible after StandardScaler on duplicate rows).
        denom = min_refs + maj_refs
        safe = denom > 0
        scores = np.where(safe, maj_refs / np.where(safe, denom, 1.0), 0.5)
        return float(np.mean(scores)), n_votes
