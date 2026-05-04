"""Non-parametric statistical tests for algorithm comparison.

Implements the Demšar (2006) pipeline:
  1. Friedman test (global null: all algorithms perform equally).
  2. Post-hoc Wilcoxon signed-rank tests with Holm correction.
  3. Average rank computation for critical-difference diagrams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import load_config


def average_ranks(results: pd.DataFrame, metric: str = "f1") -> pd.Series:
    """Compute average rank per algorithm across datasets (lower = better)."""
    pivot = results.groupby(["dataset", "algorithm"])[metric].mean().unstack("algorithm")
    # rank within each dataset row (ascending: lower score = higher/worse rank)
    ranked = pivot.rank(axis=1, ascending=False)  # rank 1 = best
    return ranked.mean().sort_values()


def friedman_test(results: pd.DataFrame, metric: str = "f1") -> tuple[float, float]:
    """Friedman test across algorithms.

    Returns
    -------
    statistic : float
    p_value : float
    """
    pivot = results.groupby(["dataset", "algorithm"])[metric].mean().unstack("algorithm")
    pivot = pivot.dropna()
    return stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])


def pairwise_wilcoxon(
    results: pd.DataFrame,
    baseline: str,
    metric: str = "f1",
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests vs a baseline with Holm correction.

    Parameters
    ----------
    results : pd.DataFrame
        Output of ``run_benchmark()``.
    baseline : str
        Algorithm name to compare all others against.
    metric : str
        Column to use as performance measure.

    Returns
    -------
    pd.DataFrame
        Columns: algorithm, statistic, p_raw, p_corrected, significant.
    """
    cfg = load_config()["statistical_tests"]
    alpha = cfg["alpha"]

    pivot = results.groupby(["dataset", "algorithm"])[metric].mean().unstack("algorithm")

    if baseline not in pivot.columns:
        raise ValueError(f"Baseline '{baseline}' not found in results.")

    rows = []
    for alg in pivot.columns:
        if alg == baseline:
            continue
        pair = pivot[[baseline, alg]].dropna()
        if len(pair) < 2:
            rows.append({"algorithm": alg, "statistic": float("nan"), "p_raw": float("nan")})
            continue
        stat, p = stats.wilcoxon(pair[baseline].values, pair[alg].values, zero_method="wilcox")
        rows.append({"algorithm": alg, "statistic": stat, "p_raw": p})

    df = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)

    # Holm correction — only over rows with a valid p-value
    df = df.sort_values("p_raw").reset_index(drop=True)
    valid = df["p_raw"].notna()
    m = valid.sum()
    p_corr = df["p_raw"].copy()
    valid_idx = df.index[valid].tolist()
    for rank, idx in enumerate(valid_idx):
        p_corr.iloc[idx] = min(df["p_raw"].iloc[idx] * (m - rank), 1.0)
    df["p_corrected"] = p_corr
    df["significant"] = df["p_corrected"].fillna(1.0) < alpha
    return df


def critical_difference(
    results: pd.DataFrame,
    metric: str = "f1",
    alpha: float | None = None,
) -> float:
    """Nemenyi critical difference (Demšar 2006, Table 5).

    CD = q_alpha * sqrt(k(k+1) / 6N)

    where k = number of algorithms, N = number of datasets.
    """
    if alpha is None:
        alpha = load_config()["statistical_tests"]["alpha"]

    pivot = results.groupby(["dataset", "algorithm"])[metric].mean().unstack("algorithm")
    k = pivot.shape[1]
    N = pivot.shape[0]

    # Critical values for alpha=0.05 (two-tailed, from Demšar 2006 Table 5)
    _q_alpha_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q = _q_alpha_005.get(k, 2.569)  # fallback
    return q * np.sqrt(k * (k + 1) / (6 * N))
