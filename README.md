# KNN Variants for Class Imbalance

**Machine Learning I (CC2008) — University of Porto, 2025/2026**

We implement and benchmark KNN-based classifiers modified to handle class imbalance in binary classification, without using scikit-learn for the core algorithm.

---

## Project structure

```
.
├── config/
│   └── settings.yaml          # All hyperparameters, paths, seeds — edit here
├── data/
│   ├── raw/                   # Original CSVs (not committed)
│   └── processed/             # Cleaned/cached datasets (not committed)
├── class_imbalance/           # Benchmark dataset bundle (provided)
├── notebooks/
│   └── analysis.ipynb         # Single executable notebook (all sections)
├── results/
│   ├── figures/               # Auto-generated plots
│   └── tables/                # Auto-generated CSV result tables
├── src/
│   ├── algorithms/
│   │   ├── knn_base.py            # Standard KNN from scratch
│   │   ├── knn_adaptive_entropy.py # Adaptive-k via local Shannon entropy
│   │   ├── knn_adaptive_eigen.py   # Adaptive-k via local eigenvalue structure
│   │   ├── dann.py                # DANN baseline (Hastie & Tibshirani 1996)
│   │   └── dann_adaptive.py       # DANN + adaptive-k (proposed contribution)
│   ├── data/
│   │   ├── loader.py          # Dataset discovery and loading
│   │   └── preprocessing.py   # Scaling, label binarisation, feature cleaning
│   ├── evaluation/
│   │   ├── metrics.py         # F1, G-mean, ROC-AUC, balanced accuracy
│   │   ├── benchmarking.py    # Repeated stratified k-fold pipeline
│   │   └── statistical_tests.py # Friedman, Wilcoxon/Holm, CD diagram
│   └── utils/
│       ├── config.py          # YAML config loader (cached singleton)
│       └── visualization.py   # Reusable plotting functions
├── tests/                     # pytest test suite
├── environment.yml            # Conda environment (name: ml1-assignment)
├── pyproject.toml             # Project metadata, black, ruff, pytest config
└── Makefile                   # Convenience commands
```

---

## Setup

**1. Create and activate the conda environment**

```bash
make env
conda activate ml1-assignment
```

**2. Register the kernel so the notebook finds it**

```bash
python -m ipykernel install --user --name ml1-assignment --display-name "Python 3 (ml1-assignment)"
```

**3. Launch the notebook**

```bash
make notebook
```

---

## Running tests

```bash
make test
```

---

## Configuration

All tunable parameters live in `config/settings.yaml`.  
No values are hard-coded anywhere in `src/`.  
Edit that file to change k ranges, CV folds, random seed, dataset paths, etc.

---

## Algorithm interface

Every classifier follows the sklearn estimator convention:

| Method | Signature |
|---|---|
| `fit` | `(X: ndarray, y: ndarray) → self` |
| `predict` | `(X: ndarray) → ndarray` |
| `predict_proba` | `(X: ndarray) → ndarray (n_samples, n_classes)` |

This means any estimator can be plugged directly into `run_benchmark()`.

---

## Cleaning results

```bash
make clean
```
