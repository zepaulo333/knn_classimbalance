"""KNNFairRank with per-query k_eff from dual-class cluster density estimates.

Motivation
──────────
The standard FairRank derivation assumes a Poisson-uniform process for each
class with GLOBAL densities λ_maj = N_maj/V, λ_min = N_min/V.  Setting

    E[d_{k_eff}^maj(x)] = E[d_1^min(x)]

under Poisson gives k_eff = λ_maj / λ_min = r globally.  When the data is
non-uniform (minority sub-clusters, majority blobs, sparse background), r is
the right correction on average but wrong locally — exactly the failure mode
all single-density-ratio approaches suffer from.

The theoretically correct local version is simply:

    k_eff(x) = λ_maj(x) / λ_min(x)

where λ_c(x) is the LOCAL density of class c at query x.  The challenge is
estimating both densities reliably from finite samples.

Algorithm
─────────
Fit
  1. Estimate ρ̂(x_i) = #{maj in ball(x_i, d_{k_ref}^min\i)} / k_ref at every
     minority training point (leave-one-out for the minority self-distance).
     This uses the stable §20 per-class counting formula.

  2. Gap-detect a partition of the sorted ρ̂ values into K regions, each
     containing ≥ k_ref minority training points (reliability guard from §17).

  3. Compute one minority centroid per region:
         C_k^min = mean(X_min in region k)      shape (K, d)
     and store cluster size  n_k^min.

  4. Run k-means on X_maj with K clusters (same K, so the two class
     representations have comparable granularity):
         C_j^maj = k-means centroids            shape (K, d)
         n_j^maj = cluster sizes

Inference
  5. For query x, find the nearest minority centroid k* and nearest majority
     centroid j* (O(K·d) — K is small, typically 1–4):

         d_k = ‖x − C_{k*}^min‖,   n_k = n_{k*}^min
         d_j = ‖x − C_{j*}^maj‖,   n_j = n_{j*}^maj

  6. Apply the local Poisson density-ratio formula (d=1 approximation):

         k_eff(x) = (n_j / d_j) / (n_k / d_k)
                   = (n_j × d_k) / (n_k × d_j)

     Derivation: under a locally homogeneous Poisson process the density at
     distance d from a cluster of n points scales as λ ∝ n/d (d=1 kernel).
     The ratio of the two class densities gives the local correction.

     When the distribution is globally uniform with K_min = K_maj = K and
     equal cluster sizes, d_k ≈ d_j and n_j/n_k ≈ r, recovering k_eff = r.

  7. Standard v3 binary voting with k_eff(x).

Graceful degradation
  If fewer than 2·k_ref minority points exist, or no ρ̂ gap exceeds
  min_persistence_ratio, K=1 and the single minority centroid equals the
  minority mean.  The formula then reduces to
      k_eff(x) = r × (d_to_minority_mean / d_to_majority_mean)
  which is a principled continuous adjustment around the global FairRank
  correction — requiring no region structure in the data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from src.algorithms.knn_fair_rank import KNNFairRank
from src.utils.config import load_config


class KNNFairRankDensityRegion(KNNFairRank):
    """KNNFairRank with per-query k_eff from dual-class cluster densities.

    At inference, k_eff(x) = (n_maj_nearest × d_min_nearest) /
                              (n_min_nearest × d_maj_nearest)
    using cluster centroids fit separately for each class, so both densities
    inform the local Poisson correction rather than only the minority side.

    Parameters
    ----------
    k_ref : int or None
        Minority budget for ρ̂ estimation and per-region reliability guard.
        None → adaptive max(3, floor(sqrt(N_min))) at fit time.
    min_persistence_ratio : float or None
        Normalised ρ̂ gap threshold to accept a region boundary.
        None → settings.yaml (default 0.1).

    Attributes
    ----------
    n_regions_ : int
        Number of minority density regions found (K).  1 = no structure found.
    min_centroids_ : NDArray, shape (K, d)
        Minority region centroids in feature space.
    maj_centroids_ : NDArray, shape (K, d)
        Majority k-means centroids in feature space.
    min_cluster_sizes_ : NDArray, shape (K,)
    maj_cluster_sizes_ : NDArray, shape (K,)
    """

    def __init__(
        self,
        k_ref: int | None = None,
        min_persistence_ratio: float | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_density_region", {})
        self._k_ref_param = k_ref if k_ref is not None else cfg.get("k_ref", None)
        self._min_persistence_ratio = float(
            min_persistence_ratio if min_persistence_ratio is not None
            else cfg.get("min_persistence_ratio", 0.1)
        )
        self.n_regions_: int = 1
        self.min_centroids_: NDArray = np.zeros((1, 1))
        self.maj_centroids_: NDArray = np.zeros((1, 1))
        self.min_cluster_sizes_: NDArray = np.array([1.0])
        self.maj_cluster_sizes_: NDArray = np.array([1.0])
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankDensityRegion":
        super().fit(X, y)
        n_min = len(self._X_min)
        n_maj = len(self._X_maj)

        # k_ref: capped to n_min-1 so LOO always has a valid k_ref-th neighbour
        if self._k_ref_param is None:
            self._k_ref = max(3, int(np.floor(np.sqrt(n_min))))
        else:
            self._k_ref = int(self._k_ref_param)
        self._k_ref = max(1, min(self._k_ref, n_min - 1))

        # ── Step 1-2: minority density regions via ρ̂ gap detection ───────────
        rho_hat = self._compute_rho_hat_at_minority()
        region_labels = self._assign_minority_regions(rho_hat)   # (N_min,) ints
        K = int(region_labels.max()) + 1
        self.n_regions_ = K

        # ── Step 3: minority region centroids + sizes ─────────────────────────
        self.min_centroids_ = np.array([
            self._X_min[region_labels == k].mean(axis=0) for k in range(K)
        ])                                                          # (K, d)
        self.min_cluster_sizes_ = np.array([
            float((region_labels == k).sum()) for k in range(K)
        ])                                                          # (K,)

        # ── Step 4: majority k-means with K clusters ──────────────────────────
        K_maj = min(K, n_maj)  # guard: cannot have more clusters than points
        if K_maj == 1:
            self.maj_centroids_ = self._X_maj.mean(axis=0, keepdims=True)
            self.maj_cluster_sizes_ = np.array([float(n_maj)])
        else:
            seed = load_config()["random_seed"]
            km = KMeans(n_clusters=K_maj, n_init=10, random_state=seed)
            km.fit(self._X_maj)
            self.maj_centroids_ = km.cluster_centers_               # (K, d)
            self.maj_cluster_sizes_ = np.bincount(
                km.labels_, minlength=K_maj
            ).astype(float)                                         # (K,)

        # _vote_fraction needs d_min up to at least k_ref for ρ̂ computation
        self._k_min_eff = min(max(self._k_min_eff, self._k_ref), n_min)
        return self

    # ── Internal fit helpers ──────────────────────────────────────────────────

    def _compute_rho_hat_at_minority(self) -> NDArray:
        """§20 counting estimate ρ̂(x_i) for each minority training point (LOO)."""
        k_ref = self._k_ref
        D_mm = cdist(self._X_min, self._X_min)    # (N_min, N_min)
        D_mm.sort(axis=1)
        radius_col = min(k_ref, D_mm.shape[1] - 1)
        radii = D_mm[:, radius_col]                # (N_min,)
        D_mj = cdist(self._X_min, self._X_maj)    # (N_min, N_maj)
        return (D_mj <= radii[:, None]).sum(axis=1) / k_ref   # (N_min,)

    def _assign_minority_regions(self, rho_hat: NDArray) -> NDArray:
        """Gap-detect K regions in sorted ρ̂ space; return per-point labels."""
        k_ref = self._k_ref
        n = len(rho_hat)
        order = np.argsort(rho_hat)
        sorted_rho = rho_hat[order]

        cut_points = self._find_cuts(sorted_rho, k_ref)

        segs = [0] + cut_points + [n]
        labels = np.empty(n, dtype=int)
        for k, (lo, hi) in enumerate(zip(segs, segs[1:])):
            labels[order[lo:hi]] = k
        return labels

    def _find_cuts(self, sorted_rho: NDArray, k_ref: int) -> list[int]:
        """Greedy gap-detection on sorted ρ̂ values; returns split positions."""
        n = len(sorted_rho)
        if n < 2 * k_ref:
            return []
        rho_range = float(sorted_rho[-1] - sorted_rho[0])
        if rho_range < 1e-9:
            return []

        norm_gaps = np.diff(sorted_rho) / rho_range
        cut_points: list[int] = []
        for gap_idx in np.argsort(norm_gaps)[::-1]:
            if norm_gaps[gap_idx] < self._min_persistence_ratio:
                break
            candidate = sorted(cut_points + [int(gap_idx) + 1])
            segs = [0] + candidate + [n]
            if all(segs[i + 1] - segs[i] >= k_ref for i in range(len(segs) - 1)):
                cut_points = candidate
        return cut_points

    # ── Inference ────────────────────────────────────────────────────────────

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        # ── Nearest minority cluster ──────────────────────────────────────────
        d_to_min = np.linalg.norm(self.min_centroids_ - x, axis=1)   # (K,)
        k_star = int(np.argmin(d_to_min))
        d_k = float(d_to_min[k_star])
        n_k = float(self.min_cluster_sizes_[k_star])

        # ── Nearest majority cluster ──────────────────────────────────────────
        d_to_maj = np.linalg.norm(self.maj_centroids_ - x, axis=1)   # (K,)
        j_star = int(np.argmin(d_to_maj))
        d_j = float(d_to_maj[j_star])
        n_j = float(self.maj_cluster_sizes_[j_star])

        # ── Local Poisson density-ratio: k_eff = λ_maj(x) / λ_min(x) ─────────
        # λ_c(x) ≈ n_c_nearest / d_c_nearest  (d=1 kernel)
        # k_eff = (n_j / d_j) / (n_k / d_k) = (n_j × d_k) / (n_k × d_j)
        eps = 1e-9
        k_eff = int(max(1, round((n_j * (d_k + eps)) / (n_k * (d_j + eps)))))

        # ── Standard v3 binary voting ─────────────────────────────────────────
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
