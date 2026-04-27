"""KNNFairRankMagnitude with α-grid ensemble voting (no inner CV).

Combines magnitude-aware scoring (Modification B) with the α-grid
ensemble approach from KNNFairRankEnsemble, replacing the inner-CV
α-selection of KNNFairRankMagnitudeCV.

For each (α, i) pair a continuous score
    s = d_{i·round(r^α)}^maj / (d_i^min + d_{i·round(r^α)}^maj)
is computed. All scores across the full grid are averaged; the hard
decision predicts minority iff mean > 0.5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.algorithms.knn_fair_rank_b import KNNFairRankMagnitude
from src.utils.config import load_config


class KNNFairRankMagnitudeEnsemble(KNNFairRankMagnitude):
    """KNNFairRankMagnitude with ensemble voting over an α grid (no inner CV).

    Parameters
    ----------
    alpha_grid : list[float] or None
        Candidate correction exponents. None → reads from
        ``knn_fair_rank_magnitude_ensemble.alpha_grid`` in settings.yaml,
        falling back to KNNFairRankCV's default [0.25, 0.5, 0.75, 1.0].
    """

    def __init__(
        self,
        alpha_grid: list[float] | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config()
        fallback = cfg.get("knn_fair_rank_cv", {}).get("alpha_grid", [0.25, 0.5, 0.75, 1.0])
        self._alpha_grid = (
            list(alpha_grid) if alpha_grid is not None
            else list(cfg.get("knn_fair_rank_magnitude_ensemble", {}).get("alpha_grid", fallback))
        )
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        all_scores: list[float] = []

        for alpha in self._alpha_grid:
            k_eff = int(max(1, round(self._r ** alpha)))
            max_votes_maj = len(d_maj) // k_eff
            n = min(self._n_votes, len(d_min), max_votes_maj)
            if n < 1:
                n = 1
                k_eff = min(k_eff, len(d_maj))

            min_refs = d_min[:n]
            maj_indices = np.arange(1, n + 1) * k_eff - 1
            maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
            maj_refs = d_maj[maj_indices]

            denom = min_refs + maj_refs
            safe = denom > 0
            scores = np.where(safe, maj_refs / np.where(safe, denom, 1.0), 0.5)
            all_scores.extend(scores.tolist())

        if not all_scores:
            return 0.0, 0
        return float(np.mean(all_scores)), len(all_scores)
