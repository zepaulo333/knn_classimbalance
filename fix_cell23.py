import json

with open('notebooks/exploration.ipynb') as f:
    nb = json.load(f)

section_23 = """\
---

## 23. KNNFairRankTopoJointBootstrap — Bootstrap-Stabilised Topology with JointCV Fallback

### 23.1 Where We Are: Two Fixes, One Remaining Problem

`KNNFairRankTopoJoint` went through two iterations before reaching this point.

**Iteration 1 — `max_regions` (original).** Restricted the Ward gap search to the last
`max_regions − 1 = 4` dendrogram merges. Average G-mean: 0.7649.

**Iteration 2 — `min_region_samples` (§17.1, implemented).** Replaced the region-count
cap with a statistical-reliability guard: a candidate Ward cut is only accepted if every
resulting region contains at least `max(10, ⌊√n⌋)` training points. This prevents the
per-region $k_\\text{eff}$ estimate from being computed from so few points that it is
pure noise. Average G-mean: **0.7687** (+0.0039).

Head-to-head against `KNNFairRank` with the new version: **15 datasets won, 13 lost**.
The average (0.7687) still sits below `KNNFairRank` (0.7872) for one reason only — three
datasets cause catastrophic losses that drag the mean down:

| Dataset | TopoJoint G-mean | FairRank G-mean | Δ |
|---------|-----------------|-----------------|---|
| analcatdata_neavote | 0.000 | 0.429 | −0.429 |
| ar1 | 0.318 | 0.493 | −0.175 |
| kc1-top5 | 0.377 | 0.516 | −0.139 |

On every other dataset TopoJoint is already competitive with or superior to `KNNFairRank`.
The bootstrap wrapper targets exactly these three remaining failures.

---

### 23.2 Why `min_region_samples` Did Not Fix the Catastrophic Cases

The two classes of topology failure have different root causes:

**Class A — Small splinter regions (fixed by `min_region_samples`).** Ward produces a
degenerate cut where one cluster contains 5–15 points with a skewed class ratio. The
$k_\\text{eff}$ estimate from this region is high-variance noise. `min_region_samples`
rejects the cut entirely, either accepting a coarser valid cut or falling back to global
$r$. Datasets mc1 (+0.061), ar6 (+0.057), backache (+0.053) were in this category.

**Class B — Large structurally-wrong regions (still present).** Ward finds a
plausible-looking gap in the dendrogram and creates large, well-populated regions. The
$k_\\text{eff}$ estimates are statistically stable (enough points). But the regional
structure is fold-specific: the gap exists because this particular 80% subsample happens
to leave a spatial hole that would be filled by the missing 20%. On the next fold, the
hole is elsewhere, the regions are different, and the per-query $k_\\text{eff}$ is wrong.
neavote, ar1, kc1-top5 are in this category — `min_region_samples` leaves them untouched
because their regions are large enough to pass the size check.

The residual problem is not statistical noise within a region. It is that the *existence*
of the region itself is fold-specific. No threshold on region size can detect this —
only a test of whether the structure survives on data the model has not seen can.

---

### 23.3 Why Not Fix This via Inner CV?

Adding an inner CV loop that decides whether to use topology (binary: topology vs.
fallback) would make `KNNFairRankJointCV` win almost every inner-fold comparison for the
same reason it wins in the outer benchmark — the topology is fold-specific and fails on
held-out inner folds just as it fails on the outer test fold. Inner CV would suppress
topology nearly everywhere, recovering `KNNFairRankJointCV` at extra computational cost.

The issue is not *whether* to use topology globally for a dataset. It is *where* in the
feature space the topology is trustworthy. A query near a genuinely-structured region
should use topology; a query near a fold-specific artefact should not. This requires a
per-query or per-region reliability estimate, not a global binary switch.

---

### 23.4 The Theoretical Gap: Perturbation Stability vs Subsampling Stability

The PH stability theorem (Cohen-Steiner, Edelsbrunner, Harer 2007) states: if two point
clouds differ by at most $\\varepsilon$ in Hausdorff distance, their persistence diagrams
differ by at most $\\varepsilon$ in bottleneck distance. This covers *perturbations* —
every point is shifted by at most $\\varepsilon$.

Subsampling is categorically different. Removing 20% of points is not a small
perturbation; it is deleting entire points. The Hausdorff distance between the 80%
sample and the full cloud can be large wherever the removed points are spatially isolated.
The stability theorem gives a vacuous bound in this regime.

**The density ratio is actually preserved.** Under stratified k-fold CV both classes are
sampled at ~80% uniformly. The local density ratio $f_\\text{maj}(x)/f_\\text{min}(x)$ is
therefore unbiased in expectation:

$$\\frac{\\hat{f}_\\text{maj}(x)}{\\hat{f}_\\text{min}(x)} = \\frac{0.8 \\cdot f_\\text{maj}(x)}{0.8 \\cdot f_\\text{min}(x)} = \\frac{f_\\text{maj}(x)}{f_\\text{min}(x)}$$

The bias is not in the ratio but in the *topology*: the cluster structure changes as
points are removed, and Ward clustering is sensitive to spatial gaps that are sampling
artefacts. What we need is not perturbation stability but **subsampling stability**: does
this topological feature persist under repeated subsampling of the training fold?

---

### 23.5 Bootstrap as Subsampling Stability Estimator

Bootstrap resampling answers this question directly. Sample with replacement $B$ times
from the training fold. For each sample, run the Ward topology with `min_region_samples`.
A structural feature (a distinct region) is *stable* if it appears consistently.

Formally, define the **stability score** of training point $p$:

$$s(p) = \\frac{1}{B} \\sum_{b=1}^{B} \\mathbf{1}\\bigl[\\text{topology fires on bootstrap } b \\text{ and } p \\text{ is assigned to a distinct region}\\bigr]$$

$s(p) \\approx 1$: regardless of which subsample is drawn, $p$ reliably belongs to a
structurally distinct region. The topology around $p$ is real.

$s(p) \\approx 0$: the region $p$ lands in changes with every resample. The structural
gap was a sampling artefact.

For the three failing datasets: neavote, ar1, and kc1-top5 all have Ward finding a gap
that does not exist in the true distribution. On repeated bootstrap resamples of the
training fold, the gap will appear at different places or not at all — resulting in low
$s(p)$ for most training points. The bootstrap detects exactly the failure mode that
`min_region_samples` cannot.

---

### 23.6 The Algorithm: KNNFairRankTopoJointBootstrap

The base model is now `KNNFairRankTopoJoint` with `min_region_samples` (the §17.1
version), not the original `max_regions` version.

#### Fit time

1. Fit `KNNFairRankJointCV` on $(X_\\text{train}, y_\\text{train})$. Record $\\alpha^*$,
   $n_\\text{votes}^*$. This is the fallback.
2. Fit `KNNFairRankTopoJoint` (with `min_region_samples`) on the full training fold.
   This gives the primary region assignments for stable queries.
3. Run $B$ bootstrap iterations:
   - Sample $(X_b, y_b)$ with replacement from the training fold (same size, stratified
     if possible to preserve class proportions).
   - Fit `KNNFairRankTopoJoint` on $(X_b, y_b)$.
   - For each original training point $p_i$, find its nearest neighbour in $X_b$ and
     record whether topology fired and what $k_\\text{eff}$ it received.
4. For each training point $p_i$, compute:
   - $s(p_i)$: fraction of bootstraps where it landed in a distinct region.
   - $\\bar{k}_\\text{eff}(p_i)$: mean $k_\\text{eff}$ across bootstraps where topology
     fired (bootstraps that fell back to global $r$ contribute nothing to the average).

#### Inference for query $x$

1. Find the nearest training point $p^* = \\arg\\min_i \\|x - p_i\\|$.
2. Evaluate $s(p^*)$ against threshold $\\tau$ (e.g. 0.7).
3. **If $s(p^*) \\geq \\tau$**: topology is verified. Set
   $k_\\text{eff}(x) = \\text{round}(\\bar{k}_\\text{eff}(p^*))$ and run standard FairRank
   voting.
4. **If $s(p^*) < \\tau$**: topology is unreliable at this location. Fall back to
   `KNNFairRankJointCV`: use $k_\\text{eff} = r^{\\alpha^*}$ with $n_\\text{votes}^*$.

The threshold $\\tau$ is the one free hyperparameter. It can be set conservatively
(0.6–0.7) or selected by an additional inner CV pass.

---

### 23.7 Expected Outcome

The bootstrap wrapper should have two distinct effects:

**Eliminating the catastrophic losses.** For neavote, ar1, kc1-top5 the Ward topology
fires on the full training fold but is a sampling artefact. On bootstrap resamples the
gap will appear inconsistently → $s(p^*) < \\tau$ for most query points → fallback to
`KNNFairRankJointCV`. Result: those three datasets recover to at least JointCV-level
performance, eliminating the −0.43, −0.18, −0.14 losses.

**Preserving the genuine wins.** Datasets where topology genuinely helps (mc1, ar6,
backache, oil_spill, arsenic-male-bladder) have real spatial imbalance structure that
persists across bootstrap resamples → $s(p^*) \\geq \\tau$ for queries near stable regions
→ topology $k_\\text{eff}$ is used and the improvement is kept.

**Net effect.** Removing the three catastrophic losses while keeping the 15 per-dataset
wins against `KNNFairRank` should push the average G-mean clearly above both
`KNNFairRank` (0.7872) and `KNNFairRankJointCV` (0.7988)."""

nb['cells'][-1]['source'] = section_23

with open('notebooks/exploration.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Done.')
