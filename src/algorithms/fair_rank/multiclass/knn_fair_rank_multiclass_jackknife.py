"""KNNFairRankMulticlass with LOO jackknife over the nearest training points.

For each query x:
  1. Compute per-class sorted distance arrays (as in the base class).
  2. Build a global sorted list of all k_probe nearest training points.
  3. For each, remove it from its class's distance array and re-run the
     multiclass vote on the modified distances.
  4. Average vote fractions across all LOO trials; predict the class with
     the highest average fraction.

Motivation
----------
With very few training points per class (as in meta-learning with ~40
datasets), a single training point that sits unusually close to the query
can dominate the vote.  The LOO average dilutes this influence: the
anomalous point is absent from exactly one trial.  If its removal changes
the outcome substantially, the average is less extreme; if it changes
nothing, the average equals the base prediction.

k_probe=None (default) runs a full LOO over all training points, which is
the most robust choice when training sets are small.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.algorithms.fair_rank.multiclass.knn_fair_rank_multiclass import KNNFairRankMulticlass
from src.utils.config import load_config


class KNNFairRankMulticlassJackknife(KNNFairRankMulticlass):
    """KNNFairRankMulticlass with LOO jackknife over training points.

    Parameters
    ----------
    k_probe : int or None
        Number of nearest training points to probe per query.
        None → probe all training points (full jackknife).
    """

    def __init__(
        self,
        k_probe: int | None = None,
        n_votes: int | None = None,
        k_buffer: int | None = None,
        k_floor: int | None = None,
        k_cap: int | None = None,
        normalize: bool = True,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_multiclass_jackknife", {})
        self._k_probe_param = (
            k_probe if k_probe is not None else cfg.get("k_probe", None)
        )
        super().__init__(
            n_votes=n_votes,
            k_buffer=k_buffer,
            k_floor=k_floor,
            k_cap=k_cap,
            normalize=normalize,
            n_jobs=n_jobs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankMulticlassJackknife":
        super().fit(X, y)
        total_train = sum(self._N_per_class.values())
        if self._k_probe_param is None:
            self._k_probe_eff = total_train  # full LOO
        else:
            self._k_probe_eff = min(self._k_probe_param, total_train)
        return self

    # ── jackknife core ────────────────────────────────────────────────────

    def _jackknife_scores(self, x: NDArray) -> dict[object, float]:
        """Average vote fractions across k_probe LOO trials."""
        dists = self._per_class_distances(x)

        if self._k_probe_eff <= 0:
            votes = self._run_votes(dists)
            total = max(1, sum(votes.values()))
            return {c: votes[c] / total for c in self.classes_}

        # Global sorted list: (distance, class, index_in_class_array)
        all_entries: list[tuple[float, object, int]] = []
        for c in self.classes_:
            for rank_0, d in enumerate(dists[c]):
                all_entries.append((d, c, rank_0))
        all_entries.sort(key=lambda e: e[0])

        n_probe = min(self._k_probe_eff, len(all_entries))
        if n_probe == 0:
            votes = self._run_votes(dists)
            total = max(1, sum(votes.values()))
            return {c: votes[c] / total for c in self.classes_}

        fracs: list[dict[object, float]] = []
        for trial_idx in range(n_probe):
            _, c_remove, rank_remove = all_entries[trial_idx]

            # Remove the trial point from its class distance array
            d_c = dists[c_remove]
            dists_loo = dict(dists)
            dists_loo[c_remove] = np.concatenate(
                [d_c[:rank_remove], d_c[rank_remove + 1:]]
            )

            votes_trial = self._run_votes(dists_loo)
            total = max(1, sum(votes_trial.values()))
            fracs.append({c: votes_trial[c] / total for c in self.classes_})

        return {
            c: float(np.mean([f[c] for f in fracs]))
            for c in self.classes_
        }

    # ── predict interface ─────────────────────────────────────────────────

    def _predict_x(self, x: NDArray):
        scores = self._jackknife_scores(x)
        return max(scores, key=scores.get)

    def _predict_proba_x(self, x: NDArray) -> NDArray:
        scores = self._jackknife_scores(x)
        total = sum(scores.values())
        if total == 0:
            return np.ones(len(self.classes_)) / len(self.classes_)
        return np.array([scores[c] / total for c in self.classes_])
