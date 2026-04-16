"""Dataset loader for the class-imbalance benchmark suite.

Scans the raw data directory for CSV files, applies basic sanity checks
(binary target, imbalance ratio within configured bounds), and returns
a list of named (X, y) pairs ready for the evaluation pipeline.

Target column detection order:
  1. Column matching ``datasets.target_column`` in settings.yaml
  2. Any column whose name matches a known target alias
  3. Last column (standard convention for these benchmark CSVs)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_config

# Known target column names across the benchmark suite
_TARGET_ALIASES = {
    "binaryClass", "class", "Class", "target", "label",
    "c", "defects", "DL", "Urban", "Damaged", "Laid.off",
    "col_33", "class14",
}


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    imbalance_ratio: float  # minority_count / majority_count


def load_all_datasets() -> list[Dataset]:
    """Load every valid dataset from the configured raw data directory."""
    cfg = load_config()
    root = get_project_root()
    data_dir = root / cfg["datasets"]["directory"]
    preferred_col = cfg["datasets"]["target_column"]
    min_ratio = cfg["datasets"]["min_imbalance_ratio"]
    max_samples = cfg["datasets"]["max_samples"]

    datasets = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        ds = _load_single(csv_path, preferred_col, min_ratio, max_samples)
        if ds is not None:
            datasets.append(ds)

    return datasets


def _detect_target_column(df: pd.DataFrame, preferred: str) -> str | None:
    """Return the name of the target column, or None if undecidable."""
    # 1. Preferred name from config
    if preferred in df.columns:
        return preferred
    # 2. Known alias
    for alias in _TARGET_ALIASES:
        if alias in df.columns:
            return alias
    # 3. Last column fallback (standard for these benchmark CSVs)
    return df.columns[-1]


def _load_single(
    path: Path,
    preferred_col: str,
    min_ratio: float,
    max_samples: int,
) -> Dataset | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty or len(df.columns) < 2:
        return None

    target_col = _detect_target_column(df, preferred_col)
    if target_col is None:
        return None

    # Drop rows where target is NaN
    df = df.dropna(subset=[target_col])

    if len(df) > max_samples or len(df) < 10:
        return None

    y_raw = df[target_col].values

    # Must be exactly 2 classes
    if len(np.unique(y_raw)) != 2:
        return None

    # Feature matrix — numeric columns only, excluding target
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values

    if X.shape[1] == 0:
        return None

    # Impute missing values with column medians
    if np.isnan(X).any():
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Compute imbalance ratio
    classes, counts = np.unique(pd.factorize(y_raw)[0], return_counts=True)
    ratio = float(counts.min() / counts.max())

    if ratio < min_ratio:
        return None

    return Dataset(
        name=path.stem,
        X=X.astype(float),
        y=y_raw,
        imbalance_ratio=ratio,
    )
