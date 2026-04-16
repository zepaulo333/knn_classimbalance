"""End-to-end benchmarking pipeline.

Takes a dict of {name: estimator} and a list of Dataset objects, runs
repeated stratified k-fold CV for every combination, and returns a tidy
DataFrame with one row per (algorithm, dataset, fold, repeat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import Dataset
from src.data.preprocessing import binarise_labels, remove_constant_features
from src.evaluation.metrics import compute_all_metrics
from src.utils.config import load_config


def run_benchmark(
    estimators: dict,
    datasets: list[Dataset],
) -> pd.DataFrame:
    """Run full benchmark and return tidy results DataFrame.

    Parameters
    ----------
    estimators : dict
        ``{algorithm_name: fitted_estimator_class}`` — classes, not instances.
        Each class is instantiated fresh per fold.
    datasets : list[Dataset]
        Datasets returned by ``load_all_datasets()``.

    Returns
    -------
    pd.DataFrame
        Columns: algorithm, dataset, fold, repeat, imbalance_ratio, + metric columns.
    """
    cfg = load_config()["evaluation"]
    cv = RepeatedStratifiedKFold(
        n_splits=cfg["cv_folds"],
        n_repeats=cfg["n_repetitions"],
        random_state=load_config()["random_seed"],
    )

    rows = []
    for ds in datasets:
        X = remove_constant_features(ds.X)
        y = binarise_labels(ds.y)

        for repeat_fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            fold = repeat_fold_idx % cfg["cv_folds"]
            repeat = repeat_fold_idx // cfg["cv_folds"]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale inside the fold
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            for alg_name, EstimatorClass in estimators.items():
                estimator = EstimatorClass()
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)

                y_proba = None
                if hasattr(estimator, "predict_proba"):
                    proba = estimator.predict_proba(X_test)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        y_proba = proba[:, 1]

                metrics = compute_all_metrics(y_test, y_pred, y_proba)
                rows.append(
                    {
                        "algorithm": alg_name,
                        "dataset": ds.name,
                        "fold": fold,
                        "repeat": repeat,
                        "imbalance_ratio": ds.imbalance_ratio,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)
