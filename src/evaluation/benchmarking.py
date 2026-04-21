"""End-to-end benchmarking pipeline.

Takes a dict of {name: estimator_class} and a list of Dataset objects, runs
repeated stratified k-fold CV for every combination, and returns a tidy
DataFrame with one row per (algorithm, dataset, fold, repeat).

Features
--------
- Incremental saving: appends results after each dataset so progress is
  preserved on interrupt.
- Resume at (dataset, algorithm) granularity: adding a new algorithm to an
  existing results file only runs the missing algorithm — already-done pairs
  are never re-computed.
- Parallel outer loop: n_jobs controls dataset-level parallelism.
- Error isolation: a failed (algorithm, fold) writes NaN metrics instead of
  crashing the whole run.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.loader import Dataset
from src.data.preprocessing import binarise_labels, remove_constant_features
from src.evaluation.metrics import compute_all_metrics
from src.utils.config import load_config


# ── Resume helpers ────────────────────────────────────────────────────────────

def _load_existing(output_path: Path) -> pd.DataFrame | None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(output_path)
    except Exception:
        return None


def _completed_pairs(output_path: Path, estimators: dict, n_folds: int) -> set[tuple[str, str]]:
    """Return (dataset, algorithm) pairs that already have >= n_folds rows."""
    existing = _load_existing(output_path)
    if existing is None:
        return set()
    alg_names = set(estimators.keys())
    counts = existing.groupby(["dataset", "algorithm"]).size()
    return {
        (ds, alg)
        for (ds, alg), count in counts.items()
        if count >= n_folds and alg in alg_names
    }


# ── Core per-dataset runner ───────────────────────────────────────────────────

def _run_dataset(
    ds: Dataset,
    estimators: dict,
    cv: RepeatedStratifiedKFold,
    cv_folds: int,
    skip_pairs: set[tuple[str, str]],
) -> list[dict]:
    """Run all algorithms × all folds for one dataset, skipping done pairs."""
    algs_to_run = {
        name: cls for name, cls in estimators.items()
        if (ds.name, name) not in skip_pairs
    }
    if not algs_to_run:
        return []

    X = remove_constant_features(ds.X)
    y = binarise_labels(ds.y)
    rows = []

    for repeat_fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold = repeat_fold_idx % cv_folds
        repeat = repeat_fold_idx // cv_folds

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for alg_name, EstimatorClass in algs_to_run.items():
            row_base = {
                "algorithm": alg_name,
                "dataset": ds.name,
                "fold": fold,
                "repeat": repeat,
                "imbalance_ratio": ds.imbalance_ratio,
            }
            try:
                estimator = EstimatorClass()
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)

                y_proba = None
                if hasattr(estimator, "predict_proba"):
                    proba = estimator.predict_proba(X_test)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        y_proba = proba[:, 1]

                metrics = compute_all_metrics(y_test, y_pred, y_proba)
                rows.append({**row_base, **metrics})

            except Exception:
                rows.append({
                    **row_base,
                    "f1": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "balanced_accuracy": float("nan"),
                    "geometric_mean": float("nan"),
                    "roc_auc": float("nan"),
                    "error": traceback.format_exc(limit=2),
                })

    return rows


def _append_chunk(df_chunk: pd.DataFrame, output_path: Path) -> None:
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    df_chunk.to_csv(output_path, mode="a", header=write_header, index=False)


# ── Public API ────────────────────────────────────────────────────────────────

def _backup_csv(output_path: Path) -> Path | None:
    """Copy output_path to a timestamped .bak file; return the backup path."""
    import shutil
    from datetime import datetime

    if not output_path.exists() or output_path.stat().st_size == 0:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_path.with_name(f"{output_path.stem}_backup_{ts}{output_path.suffix}")
    shutil.copy2(output_path, backup_path)
    print(f"  Backup saved to {backup_path.name}")
    return backup_path


def _drop_algorithm(output_path: Path, algorithm_name: str) -> Path | None:
    """Backup the CSV, then remove all rows for `algorithm_name` in-place.

    Returns the backup path (or None if no backup was made).
    """
    backup_path = _backup_csv(output_path)
    existing = _load_existing(output_path)
    if existing is None:
        return backup_path
    filtered = existing[existing["algorithm"] != algorithm_name]
    filtered.to_csv(output_path, index=False)
    n_dropped = len(existing) - len(filtered)
    if n_dropped:
        print(f"  Dropped {n_dropped} existing rows for '{algorithm_name}'.")
    return backup_path


def run_benchmark(
    estimators: dict,
    datasets: list[Dataset],
    output_path: Path | None = None,
    n_jobs: int = 1,
    replace_algorithm: str | None = None,
) -> pd.DataFrame:
    """Run benchmark with incremental saving and (dataset, algorithm) resume.

    Parameters
    ----------
    estimators : dict
        ``{algorithm_name: estimator_class}`` — classes, not instances.
    datasets : list[Dataset]
    output_path : Path or None
        CSV file for incremental saving and resume.  Results are appended
        after each dataset completes.
    n_jobs : int
        Outer dataset-level parallelism.  Use 1 (safe) or 4 (faster).
        Note: KNNOptK uses 8 threads internally; set n_jobs=1 when running
        it to avoid oversubscription.
    replace_algorithm : str or None
        If given, drop all existing rows for this algorithm name from
        output_path before running, forcing a full re-run of that algorithm
        only.  All other algorithms are left untouched.
    """
    if replace_algorithm is not None and output_path is not None:
        _drop_algorithm(output_path, replace_algorithm)

    cfg = load_config()["evaluation"]
    n_folds = cfg["cv_folds"] * cfg["n_repetitions"]
    cv = RepeatedStratifiedKFold(
        n_splits=cfg["cv_folds"],
        n_repeats=cfg["n_repetitions"],
        random_state=load_config()["random_seed"],
    )

    # (dataset, algorithm) pairs already present in output_path
    done_pairs = _completed_pairs(output_path, estimators, n_folds) if output_path else set()

    # Datasets where at least one algorithm still needs running
    remaining = [
        ds for ds in datasets
        if any((ds.name, alg) not in done_pairs for alg in estimators)
    ]

    n_done_ds = len(datasets) - len(remaining)
    n_done_pairs = len(done_pairs)
    if n_done_pairs:
        print(f"  Resuming: {n_done_pairs} (dataset, algorithm) pairs already done"
              f" ({n_done_ds} datasets fully complete, {len(remaining)} partially/not started).")

    if not remaining:
        print("  All combinations already complete — loading from cache.")
        return pd.read_csv(output_path)

    # Seed return frames with existing data
    all_frames: list[pd.DataFrame] = []
    existing = _load_existing(output_path) if output_path else None
    if existing is not None:
        all_frames.append(existing)

    if n_jobs == 1:
        for i, ds in enumerate(remaining):
            chunk_rows = _run_dataset(ds, estimators, cv, cfg["cv_folds"], done_pairs)
            if not chunk_rows:
                continue
            df_chunk = pd.DataFrame(chunk_rows)
            all_frames.append(df_chunk)
            if output_path:
                _append_chunk(df_chunk, output_path)
            n_errors = df_chunk.get("error", pd.Series(dtype=str)).notna().sum()
            err_str = f"  ⚠ {n_errors} error(s)" if n_errors else ""
            n_algs_ran = df_chunk["algorithm"].nunique()
            print(f"  [{i+1}/{len(remaining)}] {ds.name}  ({n_algs_ran} alg(s)){err_str}")
    else:
        gen = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator_unordered")(
            delayed(_run_dataset)(ds, estimators, cv, cfg["cv_folds"], done_pairs)
            for ds in remaining
        )
        for i, chunk_rows in enumerate(gen):
            if not chunk_rows:
                continue
            df_chunk = pd.DataFrame(chunk_rows)
            all_frames.append(df_chunk)
            if output_path:
                _append_chunk(df_chunk, output_path)
            n_errors = df_chunk.get("error", pd.Series(dtype=str)).notna().sum()
            err_str = f"  ⚠ {n_errors} error(s)" if n_errors else ""
            ds_name = df_chunk["dataset"].iloc[0]
            print(f"  [{i+1}/{len(remaining)}] {ds_name}{err_str}")

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
