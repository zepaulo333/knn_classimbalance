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

1.  Cap k_eff at the global r (see module docstring in original).

2.  Minimum region size (min_region_samples). A candidate Ward cut is
    only accepted if every resulting region contains at least
    min_region_samples training points. This directly controls the
    statistical reliability of the per-region k_eff estimate: a region
    of 5 points produces a meaningless ratio regardless of how large the
    dendrogram gap looks.

    The search scans all cuts that produce between 2 and
    floor(n / min_region_samples) regions. Among the cuts where every
    region meets the size requirement, the one with the largest relative
    gap (gap / cloud_diameter) is selected. If the best gap falls below
    min_persistence_ratio the algorithm falls back to a single region,
    recovering standard FairRank exactly.

    Default: None → adaptive max(10, floor(sqrt(n))) at fit time.

Why not per-query local odds (the 3-zone idea)?
    See original module docstring.

Theory: analysis.ipynb §15.4 / §16 / §17.1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import linkage, fcluster

from src.algorithms.fair_rank.core.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankTopoJoint(KNNFairRank):
    """KNNFairRank with topology-aware per-query k_eff.

    Parameters
    ----------
    min_persistence_ratio : float or None
        Minimum gap / cloud_diameter to accept a topological split.
        None → settings.yaml.
    laplace_smooth : float or None
        Additive smoothing λ in k_eff = (N_maj + λ) / (N_min + λ).
        None → settings.yaml.
    min_region_samples : int or None
        Every region produced by a candidate Ward cut must contain at
        least this many training points. Prevents accepting cuts whose
        per-region k_eff estimates are too noisy to trust.
        None → adaptive max(10, floor(sqrt(n))) at fit time.

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
        min_region_samples: int | None = None,
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
        # None → adaptive default computed at fit time from n
        self._min_region_samples: int | None = (
            min_region_samples if min_region_samples is not None
            else cfg.get("min_region_samples", None)
        )
        if self._min_region_samples is not None:
            self._min_region_samples = int(self._min_region_samples)

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
        if n < 4:
            self._fallback()
            return self

        # ── Ward dendrogram on the joint cloud ───────────────────────────────
        Z = linkage(self.X, method="ward")
        merge_dists = Z[:, 2]           # n-1 values, ascending

        max_dist = float(merge_dists[-1])
        if max_dist < 1e-10:
            self._fallback()
            return self

        gaps = np.diff(merge_dists)     # n-2 values; gaps[i] = merge_dists[i+1] - merge_dists[i]
        if len(gaps) == 0:
            self._fallback()
            return self

        # ── Resolve min_region_samples ───────────────────────────────────────
        min_samples: int = (
            max(10, int(np.floor(np.sqrt(n))))
            if self._min_region_samples is None
            else self._min_region_samples
        )

        # Maximum number of regions where every region can have >= min_samples.
        max_k = n // min_samples
        if max_k < 2:
            self._fallback()
            return self

        # ── Search for the best valid cut ────────────────────────────────────
        # A cut at gap index i produces (n - i - 1) regions.
        # We only consider cuts with 2 .. max_k regions:
        #   n - i - 1 <= max_k  →  i >= n - max_k - 1
        #   n - i - 1 >= 2      →  i <= n - 3
        i_lo = max(0, n - max_k - 1)
        i_hi = n - 3             # cut at i_hi gives exactly 2 regions

        if i_lo > i_hi:
            self._fallback()
            return self

        best_gap_ratio = 0.0
        best_labels: NDArray | None = None

        for i in range(i_lo, i_hi + 1):
            gap_ratio = float(gaps[i]) / max_dist

            # Skip gaps that cannot improve the current best (avoid fcluster).
            if gap_ratio <= best_gap_ratio:
                continue

            eps_star = (merge_dists[i] + merge_dists[i + 1]) / 2.0
            labels_i = fcluster(Z, t=eps_star, criterion="distance").astype(int)

            # Every region must be large enough for a reliable k_eff estimate.
            sizes = np.bincount(labels_i)[1:]   # labels are 1-indexed
            if not np.all(sizes >= min_samples):
                continue

            best_gap_ratio = gap_ratio
            best_labels = labels_i.copy()

        if best_labels is None or best_gap_ratio < self._min_persistence_ratio:
            self._fallback()
            return self

        labels = best_labels
        self._point_region = labels
        self.n_regions_ = int(np.unique(labels).size)

        # Per-region k_eff, capped at global r.
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
