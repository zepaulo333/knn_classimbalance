# KNN Variants for Class Imbalance

**Machine Learning I (CC2008) — University of Porto, 2025/2026**
Dataset Group 2: Binary classification under class imbalance.

We implement and benchmark KNN-based classifiers modified to handle class
imbalance, **without using scikit-learn for the core algorithm**. The main
contribution is **KNNFairRank**, a rank-correction mechanism derived from the
order statistics of Poisson nearest-neighbour distances: instead of comparing
$d_1^{\text{min}}$ against $d_1^{\text{maj}}$ (the standard KNN choice), we
compare $d_1^{\text{min}}$ against $d_r^{\text{maj}}$ where $r=N_{\text{maj}}/N_{\text{min}}$.

---

## Final deliverable

The main executable notebook is **`notebooks/entrega_final.ipynb`**.

### Notebook navigator

| Notebook | Role | When to open |
|---|---|---|
| `entrega_final.ipynb` | **Main report** — full pipeline, theory, results, statistical validation, discussion | This is the deliverable. Start here. |
| `meta_analysis.ipynb` | Instance Space Analysis pipeline — PCA, footprints, LODO-CV, stress test, Kendall trend | Open to reproduce the meta-analysis (§9 in main report references it) |
| `algorithm_documentation.ipynb` | Per-algorithm technical reference (math derivations, complexity, code walk-through) | Open as a companion when reading code in `src/algorithms/` |
| `phase1_baseline.ipynb` | Phase 1 exploratory analysis (KNNOptK baseline, IR sweep, k-selection) | Open if reviewer asks about Phase 1 checkpoint |
| `phase2_benchmark.ipynb` | Benchmark execution pipeline (calls `run_benchmark()`, generates `benchmark_5rep.csv`) | Open to re-run the benchmark from scratch |
| `algorithm_design.ipynb`, `algorithm_reference.ipynb` | Development narrative + earlier reference (legacy) | Historical context only |

Sandbox / scratch notebooks live in `notebooks/sandbox/`.

---

## Algorithms in the final benchmark (10 algorithms × 49 datasets × 50 splits)

| Algorithm | Source file | Role |
|---|---|---|
| `KNNOptK` | `src/algorithms/baseline/knn_base.py` | KNN with $k$ selected by inner 3-fold CV — **principled baseline** |
| `KNNWeighted` | `src/algorithms/baseline/knn_weighted.py` | KNN with `balanced` class-frequency weighting — **sanity-check baseline** |
| `SMOTE+KNN` | wrapper (see `notebooks/phase2_benchmark.ipynb`) | SMOTE oversampling + KNN($k=5$) — **state-of-the-art baseline** |
| `KNNFairRank` | `src/algorithms/fair_rank/core/knn_fair_rank.py` | Core proposal: $k_{\text{eff}}=r$ from Poisson derivation |
| `KNNFairRankCV` | `src/algorithms/fair_rank/core/knn_fair_rank_c.py` | + inner CV over $\alpha$ in $k_{\text{eff}}=r^\alpha$ |
| `KNNFairRankJointCV` | `src/algorithms/fair_rank/ensemble/knn_fair_rank_joint_cv.py` | + joint inner CV over $(n_{\text{votes}},\alpha)$ |
| `KNNFairRankEnsemble` | `src/algorithms/fair_rank/ensemble/knn_fair_rank_ens.py` | Vote-fraction averaging across $\alpha$-grid (no CV) |
| `KNNFairRankOptVotes` | `src/algorithms/fair_rank/ensemble/knn_fair_rank_opt_votes.py` | Inner CV over $n_{\text{votes}}$ only |
| `KNNFairRankJackknife` | `src/algorithms/fair_rank/resampling/knn_fair_rank_jackknife.py` | LOO over minority ranks (variance reduction) |
| `KNNFairRankTopoJointBootstrap` | `src/algorithms/fair_rank/topology/knn_fair_rank_topo_joint_bootstrap.py` | Bootstrap-stabilised topology-aware variant |

> **Reference adaptation:** the core KNN loop in `knn_base.py` is adapted from
> [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms/blob/master/mla/knn.py)
> with modifications documented in the file header. No scikit-learn KNN
> implementation is used at any point.

> **Additional variants** explored during development (LID-derived $\alpha$,
> Bayesian shrinkage, density regions, topology-count, etc.) live in
> `src/algorithms/fair_rank/{local,topology,resampling}/` but were excluded
> from the final benchmark on focus grounds — they did not consistently beat
> KNNFairRankJointCV. They are documented in
> `notebooks/sandbox/algorithm_reference.ipynb`.

---

## Headline results (49 datasets, 5 reps × 10-fold StratifiedCV)

- **Friedman test:** $p < 10^{-12}$ on all five metrics — algorithms differ significantly.
- **G-Mean champion:** `KNNFairRankJointCV` (mean 0.799 vs 0.631 for `KNNOptK`).
- **ROC-AUC & PR-AUC champion:** `KNNFairRankEnsemble`, **statistically beating
  SMOTE+KNN** under Holm correction (ROC-AUC: $p<10^{-4}$, $\Delta=+0.034$;
  PR-AUC: $p<10^{-4}$, $\Delta=+0.081$).
- **Trade-off discovered:** threshold-based metrics favour `JointCV` (sharp
  decisions via inner-CV-selected $\alpha$); ranking metrics favour `Ensemble`
  (smoother probabilities via $\alpha$-grid averaging).
- **Instance Space Analysis** (notebook §9.4): two KNN-graph structural
  meta-features correlate strongly and *in the theoretically predicted direction*
  with the FairRank advantage $\Delta\text{G-Mean} = \text{JointCV} - \text{KNNOptK}$:
  `borderline_fraction` $\rho = +0.77$ ($p<10^{-4}$) and
  `minority_nbr_purity` $\rho = -0.76$ ($p<10^{-4}$).
  FairRank helps most where minorities sit on the class boundary and least where
  they already form pure clusters — direct empirical confirmation of the
  structural-bias mechanism predicted by the Poisson derivation in §2.

---

## Project structure

```
.
├── config/
│   └── settings.yaml              # All hyperparameters, paths, seeds
├── class_imbalance/               # Benchmark dataset bundle (49 binary CSVs)
├── notebooks/
│   ├── entrega_final.ipynb        # ★ The deliverable
│   ├── phase1_baseline.ipynb      # Phase 1 baseline analysis
│   ├── phase2_benchmark.ipynb     # Phase 2 benchmark execution
│   └── sandbox/                   # Development notebooks (analysis,
│                                  #   exploration, algorithm_design, etc.)
├── results/
│   ├── figures/                   # Auto-generated plots (PDF + PNG)
│   └── tables/                    # benchmark_5rep.csv (canonical),
│                                  #   wilcoxon_*, dataset_summary.csv, etc.
├── src/
│   ├── algorithms/
│   │   ├── baseline/              # KNN base, KNNOptK, KNNWeighted
│   │   ├── fair_rank/             # FairRank family (core, ensemble,
│   │   │                          #   local, resampling, topology)
│   │   └── adaptive_k/            # Exploratory adaptive-k variants
│   ├── data/                      # loader.py, preprocessing.py
│   ├── evaluation/                # benchmarking.py, metrics.py,
│   │                              #   statistical_tests.py
│   └── utils/                     # config.py, visualization.py
├── tests/                         # pytest suite (knn_base, fair_rank,
│                                  #   metrics, preprocessing, benchmarking)
├── environment.yml                # Conda environment (name: ml1-assignment)
├── pyproject.toml                 # Package metadata + black/ruff/pytest config
└── Makefile                       # make env / make notebook / make test
```

---

## Reproducibility

### One-shot pipeline

```bash
make env                           # creates conda env "ml1-assignment"
conda activate ml1-assignment
python -m ipykernel install --user --name ml1-assignment \
       --display-name "Python 3 (ml1-assignment)"
jupyter lab notebooks/entrega_final.ipynb
```

### Re-running the benchmark from scratch (~2–3 hours on Apple M1)

```bash
# All results in entrega_final.ipynb are read from results/tables/benchmark_5rep.csv.
# To regenerate it, run the benchmark cell in notebooks/phase2_benchmark.ipynb
# with quick_run=false in config/settings.yaml.
```

### Computational environment

- Python 3.11 (see `environment.yml` for full dependency list)
- 50 splits per (algorithm, dataset) = 5 repetitions × 10-fold StratifiedKFold
- Random seed `42`, set in `config/settings.yaml`
- Wall-clock on Apple M1 / 16 GB: ≈ 2.5 hours for the full 49 × 10 algorithm benchmark
- `KNNFairRankJointCV` and `KNNFairRankOptVotes` use inner CV (`n_jobs=8`),
  dominant cost contributors

### Expected output

- `results/tables/benchmark_5rep.csv` — 24,500 rows (49 ds × 10 algos × 50 splits)
- `results/tables/benchmark_5rep_degenerate.csv` — 4,500 rows for the 9
  datasets with $N_{\text{min}}<20$ (analysed separately in §8 of the notebook)
- All Friedman $\chi^2$ statistics with $p<10^{-9}$ on five metrics
- `KNNFairRankJointCV` ranking #1 in G-Mean and MCC
- `KNNFairRankEnsemble` ranking #1 in ROC-AUC and PR-AUC

---

## Running the test suite

```bash
make test                          # runs pytest on src/
pytest --cov=src tests/            # with coverage (requires pytest-cov)
```

---

## Configuration

All tunable parameters live in `config/settings.yaml`. No hard-coded values
appear in `src/`. Edit that file to change CV folds, random seed, $\alpha$
grids, dataset paths, etc.

---

## Algorithm interface

Every classifier follows the sklearn estimator convention:

| Method | Signature |
|---|---|
| `fit` | `(X: ndarray, y: ndarray) → self` |
| `predict` | `(X: ndarray) → ndarray` |
| `predict_proba` | `(X: ndarray) → ndarray (n_samples, n_classes)` |

This allows every estimator to be plugged directly into `run_benchmark()`.

---

## Known limitations

**Categorical NaN imputation.** Six datasets contain NaN values in categorical
feature columns (`dataset_1000_hypothyroid`, `dataset_38_sick`,
`dataset_1002_ipums_la_98-small`, `dataset_1018_ipums_la_99-small`,
`dataset_1023_soybean`, `dataset_968_analcatdata_birthday`). Inside each
training fold they are imputed with the most-frequent value of that column
**before** `OneHotEncoder` is fitted (see `src/evaluation/benchmarking.py`,
lines 102-107). The imputer and encoder are both fit on the training split
only and `transform`-ed on the test split, so there is no leakage. The residual
limitation is that most-frequent imputation collapses systematic missingness
patterns into the modal category — if NaN was itself informative, that signal
is lost. Numerical NaN values are imputed with training-fold column medians
following the same train-only protocol.

**Zero-feature datasets after preprocessing.** Three datasets
(`dataset_1023_soybean`, `dataset_867_visualizing_livestock`,
`dataset_875_analcatdata_chlamydia`) have all-categorical raw inputs with no
numerical columns, leaving `X.shape[1] = 0` before the categorical encoding
step. These are silently skipped by the meta-feature loops in §2.3 and §9.1
of the notebook (`continue` guards on `X.shape[1] == 0`).

**Degenerate datasets.** Nine datasets in the suite have fewer than 20
minority samples (`dataset_1013_analcatdata_challenger`,
`dataset_1045_kc1-top5`, `dataset_1059_ar1`, `dataset_1064_ar6`,
`dataset_450_analcatdata_lawsuit`, `dataset_865_analcatdata_neavote`,
`dataset_875_analcatdata_chlamydia`, `dataset_950_arsenic-female-lung`,
`dataset_951_arsenic-male-lung`). With stratified 10-fold CV these can yield
test folds with zero minority examples, making G-Mean undefined. We separate
them into a descriptive-only analysis (§8 of the notebook) rather than letting
them distort the main statistical battery.

**Poisson-uniform assumption.** The derivation $k_{\text{eff}}=r$ assumes
homogeneous Poisson processes for both classes locally. Where minority points
form tight clusters (non-uniform density), the assumption is approximately
violated; the `KNNFairRankCV` family relaxes this by tuning the exponent
$\alpha$ in $k_{\text{eff}}=r^\alpha$ via inner CV. Empirically, $\alpha^*=1$
is selected by the CV in the majority of folds, supporting the assumption
in the bulk of our 49-dataset suite.

---

## Cleaning results

```bash
make clean                         # removes generated figures and tables
```

---

## References

- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple data sets.* JMLR 7, 1–30.
- Smith-Miles, K. & Muñoz, M.A. (2023). *Instance Space Analysis: A toolkit for the assessment of algorithmic power.* ACM Computing Surveys 55(12).
- Hastie, T. & Tibshirani, R. (1996). *Discriminant adaptive nearest-neighbor classification.* IEEE TPAMI 18(6), 607–616.
- Chawla, N.V. et al. (2002). *SMOTE: Synthetic minority over-sampling technique.* JAIR 16, 321–357.
- [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms) — base KNN loop adapted with sklearn-compatible interface.
