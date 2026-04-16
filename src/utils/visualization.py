"""Reusable plotting utilities for the benchmarking notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import get_project_root, load_config

_cfg = load_config()
_FIG_DIR = get_project_root() / _cfg["paths"]["results_figures"]


def _save(fig: plt.Figure, filename: str) -> None:
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIG_DIR / filename, bbox_inches="tight", dpi=150)


def plot_class_distribution(y, dataset_name: str = "", save: bool = False) -> plt.Figure:
    """Bar chart of class frequencies."""
    labels, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([str(l) for l in labels], counts)
    ax.set_title(f"Class distribution — {dataset_name}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    if save:
        _save(fig, f"class_dist_{dataset_name}.png")
    return fig


def plot_metric_comparison(
    results: pd.DataFrame,
    metric: str = "f1",
    save: bool = False,
) -> plt.Figure:
    """Box plot comparing algorithms across datasets for one metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    order = results.groupby("algorithm")[metric].median().sort_values(ascending=False).index
    sns.boxplot(data=results, x="algorithm", y=metric, order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(f"{metric.upper()} by algorithm")
    if save:
        _save(fig, f"comparison_{metric}.png")
    return fig


def plot_critical_difference(
    ranks: pd.Series,
    cd: float,
    title: str = "Critical Difference Diagram",
    save: bool = False,
) -> plt.Figure:
    """Minimal critical difference diagram (Demšar 2006)."""
    fig, ax = plt.subplots(figsize=(8, 3))
    names = ranks.index.tolist()
    vals = ranks.values

    ax.scatter(vals, np.zeros_like(vals), zorder=3)
    for name, v in zip(names, vals):
        ax.annotate(name, (v, 0), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    best = vals.min()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.hlines(0, best, best + cd, colors="red", linewidths=2, label=f"CD = {cd:.3f}")

    ax.set_yticks([])
    ax.set_xlabel("Average rank (lower = better)")
    ax.set_title(title)
    ax.legend()
    if save:
        _save(fig, "critical_difference.png")
    return fig


def plot_imbalance_vs_metric(
    results: pd.DataFrame,
    metric: str = "f1",
    save: bool = False,
) -> plt.Figure:
    """Scatter of imbalance ratio vs metric score, coloured by algorithm."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg, grp in results.groupby("algorithm"):
        ax.scatter(grp["imbalance_ratio"], grp[metric], label=alg, alpha=0.6, s=20)
    ax.set_xlabel("Imbalance ratio (minority / majority)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs imbalance ratio")
    ax.legend(fontsize=7)
    if save:
        _save(fig, f"imbalance_vs_{metric}.png")
    return fig
