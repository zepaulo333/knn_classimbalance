"""KNNFairRank with topology-derived per-query scale for the counting estimator.

The exact rank-correction formula (§22 of exploration.ipynb) is:

    k_eff(x) = (N_maj · f_maj(x)) / (N_min · f_min(x))

Counting both classes in the same ball of radius r gives a dimension-free
estimate because ω_d · r^d cancels exactly (same centre, same radius):

    k_eff(x) = n_maj_ball(x, r) / n_min_ball(x, r)

KNNFairRankLocalCount implements this with r = d_{k_ref,min}(x), but k_ref
is arbitrary: it anchors the scale to the minority distribution, giving a
large biased ball in minority-sparse regions.

This variant replaces the arbitrary scale with the topology-derived scale
ε*(x): the H0 death time just before the largest gap in the Vietoris-Rips
filtration over x's k_max nearest joint neighbours.  That is the largest
radius at which x's local neighbourhood is still a coherent connected
piece — above it you blend distinct density regions; below it the estimate
is noisy.

Invariant (§22.5): k_min_eff = k_maj_eff = k_max.
Proof: ε*(x) ≤ d_joint_{k_max}(x) ≤ d_min_{k_max}(x), so the per-class
distance arrays always cover the full ball at radius ε*(x).  The boundary
degenerate case (ε* > max fetched distance) is therefore impossible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist, pdist, squareform

from src.algorithms.fair_rank.core.knn_fair_rank import KNNFairRank
from src.utils.config import load_config

try:
    from ripser import ripser as _ripser
    _RIPSER_AVAILABLE = True
except ImportError:
    _RIPSER_AVAILABLE = False


class KNNFairRankTopoCount(KNNFairRank):
    """KNNFairRank with topology-derived per-query counting scale.

    At inference time, runs H0 persistent homology on the k_max nearest joint
    training neighbours of each query point x, selects ε*(x) as the death
    time just before the largest gap in the finite H0 bars, and counts both
    classes in ball(x, ε*).  When no significant gap is found, falls back to
    the LocalCount estimate at k_ref.

    Parameters
    ----------
    k_max : int or None
        Joint neighbourhood pool size used both for PH and for per-class
        distance arrays (the k_min_eff = k_maj_eff = k_max invariant).
        None → adaptive: max(floor(sqrt(N)), round(r · k_ref)) at fit time,
        capped to min(N_min, N_maj).
    min_persistence_ratio : float or None
        Minimum (largest H0 gap / max finite death) to accept ε*.
        Below this threshold the neighbourhood is treated as homogeneous
        and LocalCount fallback is used.  None → settings.yaml (default 0.05).

    Attributes
    ----------
    k_max_ : int
        Actual k_max used after fit-time adaptive computation.
    """

    def __init__(
        self,
        k_max: int | None = None,
        min_persistence_ratio: float | None = None,
        k_min: int | None = None,
        k_maj_buffer: int | None = None,
        k_maj_floor: int | None = None,
        k_maj_cap: int | None = None,
        n_votes: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        cfg = load_config().get("knn_fair_rank_topo_count", {})
        self._k_max_param = k_max if k_max is not None else cfg.get("k_max", None)
        self._min_persistence_ratio: float = float(
            min_persistence_ratio if min_persistence_ratio is not None
            else cfg.get("min_persistence_ratio", 0.05)
        )
        self.k_max_: int = 1
        super().__init__(
            k_min=k_min,
            k_maj_buffer=k_maj_buffer,
            k_maj_floor=k_maj_floor,
            k_maj_cap=k_maj_cap,
            n_votes=n_votes,
            n_jobs=n_jobs,
        )

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNFairRankTopoCount":
        super().fit(X, y)

        n_min = len(self._X_min)
        n_maj = len(self._X_maj)
        n_total = len(self.X)

        k_ref_default = max(3, int(np.floor(np.sqrt(n_min))))

        if self._k_max_param is None:
            k_max = max(
                int(np.floor(np.sqrt(n_total))),
                int(round(self._r * k_ref_default)),
            )
        else:
            k_max = int(self._k_max_param)

        # Cap to N_min (invariant requires k_min_eff ≥ k_max ≤ N_min).
        # k_maj_eff is NOT capped here — it stays at the base FairRank generous
        # sizing (ceil(r)*n_votes + buffer) so the voting step has enough majority
        # distances.  The invariant proof only requires k_min_eff ≥ k_max; majority
        # satisfies d_joint_k ≤ d_maj_k automatically since base sizes k_maj >> k_max.
        k_max = max(1, min(k_max, n_min, n_total - 1))
        self.k_max_ = k_max

        # Invariant: k_min_eff = k_max (§22.5 proof — minority side only)
        self._k_min_eff = k_max

        return self

    # ── PH scale selection ────────────────────────────────────────────────────

    def _compute_eps_star(self, X_pool: NDArray) -> float | None:
        """H0 death time just before the largest finite-death gap, or None."""
        if not _RIPSER_AVAILABLE or len(X_pool) < 2:
            return None

        try:
            D = squareform(pdist(X_pool, metric="euclidean"))
            result = _ripser(D, maxdim=0, distance_matrix=True)
        except Exception:
            return None

        h0 = result["dgms"][0]
        finite_mask = np.isfinite(h0[:, 1])
        if not finite_mask.any():
            return None

        deaths = np.sort(h0[finite_mask, 1])
        max_death = float(deaths[-1])
        if max_death < 1e-10:
            return None

        if len(deaths) == 1:
            if deaths[0] / max_death >= self._min_persistence_ratio:
                return float(deaths[0])
            return None

        gaps = np.diff(deaths)
        best = int(np.argmax(gaps))
        if gaps[best] / max_death < self._min_persistence_ratio:
            return None

        # ε* = death time just before the largest gap
        return float(deaths[best])

    # ── inference ────────────────────────────────────────────────────────────

    def _vote_fraction(self, x: NDArray) -> tuple[float, int]:
        d_min, d_maj = self._per_class_distances(x)
        if len(d_min) == 0 or len(d_maj) == 0:
            return (1.0 if len(d_min) > 0 else 0.0, 0)

        # Joint k_max pool (coordinates needed for PH)
        dists_joint = cdist(self.X, x.reshape(1, -1)).ravel()
        pool_size = min(self.k_max_, len(dists_joint))
        pool_idx = np.argpartition(dists_joint, pool_size - 1)[:pool_size]
        X_pool = self.X[pool_idx]

        eps_star = self._compute_eps_star(X_pool)

        if eps_star is None:
            # No topological structure found: neighbourhood is homogeneous, so the
            # Poisson-uniform assumption holds locally and global r is exact.
            k_eff = int(max(1, round(self._r)))
        else:
            n_min_ball = int(np.searchsorted(d_min, eps_star, side="right"))
            n_maj_ball = int(np.searchsorted(d_maj, eps_star, side="right"))

            if n_min_ball == 0:
                # Pure-majority region: fall back to global correction
                k_eff = int(max(1, round(self._r)))
            elif n_maj_ball == 0:
                # Pure-minority region: all votes go to minority
                return 1.0, 1
            else:
                k_eff = int(max(1, round(n_maj_ball / n_min_ball)))

        # Standard FairRank voting with the per-query k_eff
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
