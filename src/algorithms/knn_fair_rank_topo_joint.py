"""KNNFairRank with per-query k_eff from joint-cloud topology.

Standard FairRank applies the same global correction k_eff = r to every
query. This assumes uniform spatial density for both classes (Poisson-
uniform). When that fails — minority concentrated in sub-regions, majority
spread unevenly — the single global r over- or under-corrects depending
on where the query falls.

This variant partitions the training space into regions via Ward linkage
on the *joint* point cloud (all N = N_maj + N_min points) and uses the
local imbalance ratio in each region:

    k_eff(x) = min( (N_maj(R_j) + λ) / (N_min(R_j) + λ),  r )

where R_j is the region containing x and λ is a Laplace smoothing term.

Two design choices prevent over-correction:

1.  Cap k_eff at the global r. The Poisson derivation assumes uniform
    local density; in a region with no minority the assumption fails and
    the predict-time vote uses cross-region distances that no longer
    share a common density. Letting k_eff exceed r there flips the vote
    toward minority — the opposite of what we want. The cap ensures the
    algorithm is ≥ FairRank by construction: minority-rich regions get a
    milder correction (the gain), majority-dominated regions fall back to
    the global value.

2.  Bound the number of regions (max_regions). The largest gap in the
    Ward dendrogram is meant to be the cloud's dominant H0 scale (one or
    two big pieces). On data with tied distances (integer/categorical
    features) Ward produces degenerate dendrograms where the largest gap
    can land at a fine-grained merge, fragmenting the cloud into many
    micro-regions whose per-region statistics are sampling noise. The gap
    search is restricted to the last max_regions - 1 merges.

Why not per-query local odds (the 3-zone idea)?
    A continuous per-query local ratio k_eff = clip(n_maj_knn/n_min_knn, 1, r)
    reduces k_eff even in "mixed" zones where minority is only slightly
    above average — which includes almost every zone for scattered-minority
    datasets. This causes systematic under-correction and collapses minority
    recall on datasets without genuine spatial concentration. Ward is more
    conservative: it only deviates from r when a *large*, statistically
    clear structural gap exists. That stability makes it broadly safer.

Graceful degradation: if no gap exceeds min_persistence_ratio of the
cloud diameter, the algorithm falls back to a single region, recovering
standard FairRank exactly.

Theory: analysis.ipynb §15.4 / §16.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import linkage, fcluster

from src.algorithms.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankTopoJoint(KNNFairRank):
    """KNNFairRank with topology-aware per-query k_eff.

    Parameters
    ----------
    min_persistence_ratio : float or None
        Minimum gap / cloud_diameter to accept a topological split.
        Below this threshold the data is treated as uniform and the
        algorithm falls back to standard FairRank. None → settings.yaml.
    laplace_smooth : float or None
        Additive smoothing λ in k_eff = (N_maj + λ) / (N_min + λ).
        Prevents division by zero and shrinks extremes toward 1.
        None → settings.yaml.
    max_regions : int or None
        Upper bound on the number of Ward partitions. Prevents degenerate
        fragmentation on tied-distance data. None → settings.yaml.

    Attributes
    ----------
    n_regions_ : int
        Number of topological regions found at fit time. 1 = fallback.
    point_k_eff_ : NDArray of shape (n_train,)
        Per-training-point effective k, for visualisation.
    zone_counts_ : dict
        Counts of training points in majority / mixed / minority zones.
    """

    def __init__(
        self,
        min_persistence_ratio: float | None = None,
        laplace_smooth: float | None = None,
        max_regions: int | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_topo_joint", {})
        self._min_persistence_ratio: float = (
            min_persistence_ratio if min_persistence_ratio is not None
            else float(cfg.get("min_persistence_ratio", 0.05))
        )
        self._laplace_smooth: float = (
            laplace_smooth if laplace_smooth is not None
            else float(cfg.get("laplace_smooth", 1.0))
        )
        self._max_regions: int = (
            max_regions if max_regions is not None
            else int(cfg.get("max_regions", 5))
        )
        self._point_region: NDArray = np.zeros(0, dtype=int)
        self._region_k_eff: dict[int, float] = {}
        self.n_regions_: int = 1
        self.point_k_eff_: NDArray = np.zeros(0, dtype=float)
        self.zone_counts_: dict[str, int] = {}
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankTopoJoint":
        super().fit(X, y)

        n = len(self.X)
        if n < 2:
            self._fallback()
            return self

        # ── Ward dendrogram on the joint cloud ───────────────────────────────
        Z = linkage(self.X, method="ward")
        merge_dists = Z[:, 2]           # n-1 values, ascending

        max_dist = float(merge_dists[-1])
        if max_dist < 1e-10:
            self._fallback()
            return self

        gaps = np.diff(merge_dists)
        if len(gaps) == 0:
            self._fallback()
            return self

        # Restrict search to last max_regions-1 gaps → at most max_regions clusters.
        lo = max(0, len(merge_dists) - self._max_regions)
        best_idx = lo + int(np.argmax(gaps[lo:]))
        best_gap = float(gaps[best_idx])

        if best_gap / max_dist < self._min_persistence_ratio:
            self._fallback()
            return self

        eps_star = (merge_dists[best_idx] + merge_dists[best_idx + 1]) / 2.0
        labels = fcluster(Z, t=eps_star, criterion="distance").astype(int)
        self._point_region = labels
        self.n_regions_ = int(np.unique(labels).size)

        # Per-region k_eff, capped at global r (see module docstring §1).
        lam = self._laplace_smooth
        self._region_k_eff = {}
        for rid in np.unique(labels):
            mask = labels == rid
            n_maj = float(np.sum(self.y[mask] == self._majority_class))
            n_min = float(np.sum(self.y[mask] == self._minority_class))
            self._region_k_eff[int(rid)] = min(
                (n_maj + lam) / (n_min + lam), self._r
            )

        # Expose per-point k_eff and zone counts for visualisation.
        self.point_k_eff_ = np.array(
            [self._region_k_eff[int(r)] for r in labels], dtype=float
        )
        tol = 0.01
        is_maj = self.point_k_eff_ >= self._r - tol
        is_min = self.point_k_eff_ <= 1.0 + tol
        self.zone_counts_ = {
            "majority": int(is_maj.sum()),
            "minority": int(is_min.sum()),
            "mixed":    int((~is_maj & ~is_min).sum()),
        }
        return self

    def _fallback(self) -> None:
        n = len(self.X) if len(self.X) > 0 else 0
        self._point_region = np.zeros(n, dtype=int)
        self._region_k_eff = {0: float(self._r)}
        self.n_regions_ = 1
        self.point_k_eff_ = np.full(n, float(self._r))
        self.zone_counts_ = {"majority": n, "minority": 0, "mixed": 0}

    # ── prediction ───────────────────────────────────────────────────────────

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        # Nearest training point → its region → local k_eff.
        diff = self.X - x
        nearest_idx = int(np.argmin((diff * diff).sum(axis=1)))
        region = int(self._point_region[nearest_idx])
        k_eff = int(max(1, round(self._region_k_eff[region])))

        max_votes_maj = len(d_maj) // k_eff
        n_votes = min(self._n_votes, len(d_min), max_votes_maj)
        if n_votes < 1:
            n_votes = 1
            k_eff = min(k_eff, len(d_maj))

        min_refs = d_min[:n_votes]
        maj_indices = np.arange(1, n_votes + 1) * k_eff - 1
        maj_indices = np.clip(maj_indices, 0, len(d_maj) - 1)
        return int(np.sum(min_refs < d_maj[maj_indices])) / n_votes, n_votes
