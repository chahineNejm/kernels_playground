"""
Visualization helpers for the kernel-operator playground.

Existing (from notebook):
    plot_predictions_for_domain - tall strip of per-sample prediction plots

New convenience functions:
    plot_series            - quick single-series (history + future) view
    plot_series_grid       - grid of N sample series at a glance
    plot_distribution      - histogram + box-plot of a 1-D array
    plot_train_test_split  - visual confirmation of train / test partition
    plot_kernel_heatmap    - heatmap of a Gram or kernel matrix
    plot_residuals         - residuals scatter + histogram
    plot_metric_summary    - bar chart of RMSE / relRMSE per sample
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from utils.config import DEFAULT_CONFIG


# ===================================================================
# SINGLE-SERIES QUICK VIEW
# ===================================================================

def plot_series(
    history: np.ndarray,
    future: np.ndarray | None = None,
    *,
    title: str = "",
    history_context: int | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Quick plot of a single time series split into history (grey) and future (black).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3.5))
    else:
        fig = ax.figure

    h = history if history_context is None else history[-history_context:]
    h_x = np.arange(-len(h), 0)
    ax.plot(h_x, h, color="0.6", lw=1.5, label="history")

    if future is not None:
        f_x = np.arange(len(future))
        ax.plot(f_x, future, color="black", lw=2, label="future")
        ax.axvline(0, color="tab:blue", ls=":", lw=1)

    ax.set_title(title or "series preview")
    ax.set_xlabel("time index (relative to split)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ===================================================================
# GRID OF SAMPLES
# ===================================================================

def plot_series_grid(
    examples: Sequence[dict],
    *,
    indices: Sequence[int] | None = None,
    cols: int = 3,
    history_context: int | None = None,
    figsize_per_cell: tuple[float, float] = (4.5, 2.5),
    show: bool = True,
):
    """
    Show a grid of sample series for a quick visual overview.

    Changes from original:
    - Defaults to showing ALL history (history_context=None).
    - Defaults to plotting ALL examples provided.
    - Prevents double-plotting in Jupyter notebooks.
    - Accepts 'indices' to filter specific samples while keeping true titles.
    """
    # 1. Filter by indices if requested, keeping track of the true index
    if indices is not None:
        plot_examples = []
        for idx in indices:
            ex = examples[idx].copy()
            if "sample_idx" not in ex:
                ex["sample_idx"] = idx
            plot_examples.append(ex)
    else:
        plot_examples = examples

    n = len(plot_examples)
    if n == 0:
        print("No examples to plot.")
        return

    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_cell[0] * cols, figsize_per_cell[1] * rows),
        squeeze=False,
    )

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]

        if i >= n:
            ax.set_visible(False)
            continue

        ex = plot_examples[i]

        if history_context is None:
            h = ex["history"]
        else:
            h = ex["history"][-history_context:]

        h_x = np.arange(-len(h), 0)
        f_x = np.arange(len(ex["future"]))

        ax.plot(h_x, h, color="0.6", lw=1)
        ax.plot(f_x, ex["future"], color="black", lw=1.4)
        ax.axvline(0, color="tab:blue", ls=":", lw=0.8)

        original_idx = ex.get("sample_idx", i if indices is None else indices[i])
        ax.set_title(f"sample {original_idx}", fontsize=9)

        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Sample series grid", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if show:
        plt.show()
        return None

    return fig


# ===================================================================
# DISTRIBUTION PLOT
# ===================================================================

def plot_distribution(
    values: np.ndarray,
    *,
    title: str = "distribution",
    bins: int = 50,
    show: bool = True,
) -> plt.Figure:
    """Histogram + horizontal box-plot of a 1-D array."""
    values = np.asarray(values, dtype=float).ravel()
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(10, 4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_hist.hist(values[np.isfinite(values)], bins=bins, color="steelblue", edgecolor="white", lw=0.4)
    ax_hist.set_ylabel("count")
    ax_hist.set_title(title)
    ax_box.boxplot(values[np.isfinite(values)], vert=False, widths=0.6)
    ax_box.set_xlabel("value")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ===================================================================
# TRAIN / TEST SPLIT VISUALIZATION
# ===================================================================

def plot_train_test_split(
    n_total: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    title: str = "train / test split",
    show: bool = True,
) -> plt.Figure:
    """Colour-coded bar showing which indices are train vs test."""
    fig, ax = plt.subplots(figsize=(12, 1.2))
    colours = np.full(n_total, 0.85)
    colours[train_indices] = 0.35
    colours[test_indices] = 0.0
    ax.imshow(
        colours[None, :], aspect="auto", cmap="gray", vmin=0, vmax=1,
        extent=[0, n_total, 0, 1],
    )
    ax.set_yticks([])
    ax.set_xlabel("sample index")
    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ===================================================================
# KERNEL / GRAM MATRIX HEATMAP
# ===================================================================

def plot_kernel_heatmap(
    matrix: np.ndarray,
    *,
    title: str = "kernel matrix",
    cmap: str = "viridis",
    show: bool = True,
) -> plt.Figure:
    """Heatmap of a square kernel or Gram matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ===================================================================
# RESIDUAL DIAGNOSTICS
# ===================================================================

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "residuals",
    show: bool = True,
) -> plt.Figure:
    """Scatter of residuals vs predicted + marginal histogram."""
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig, (ax_sc, ax_hist) = plt.subplots(
        1, 2, figsize=(12, 4),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    ax_sc.scatter(y_pred, res, s=8, alpha=0.5, color="tab:orange")
    ax_sc.axhline(0, color="black", lw=0.8)
    ax_sc.set_xlabel("predicted")
    ax_sc.set_ylabel("residual")
    ax_sc.set_title(title)
    ax_sc.grid(True, alpha=0.2)

    ax_hist.hist(res, bins=40, orientation="horizontal", color="tab:orange", edgecolor="white", lw=0.4)
    ax_hist.axhline(0, color="black", lw=0.8)
    ax_hist.set_xlabel("count")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ===================================================================
# METRIC BAR CHART
# ===================================================================

def plot_metric_summary(
    predictions: Sequence[dict],
    *,
    metric: str = "relative_rmse",
    title: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Bar chart of a per-sample metric (e.g. relRMSE) from a predictions list."""
    vals = [p[metric] for p in predictions]
    fig, ax = plt.subplots(figsize=(max(6, len(vals) * 0.25), 4))
    ax.bar(range(len(vals)), vals, color="tab:orange", edgecolor="white", lw=0.4)
    ax.set_xlabel("test sample")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} per test sample")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ===================================================================
# ORIGINAL NOTEBOOK PLOT (kept for compatibility)
# ===================================================================

def plot_predictions_for_domain(
    domain: str,
    records: Sequence[dict],
    predictions: Sequence[dict],
    *,
    history_context: int = DEFAULT_CONFIG["HISTORY_CONTEXT_TO_PLOT"],
    show: bool = True,
) -> plt.Figure:
    """Tall multi-panel figure: one subplot per test sample."""
    n = len(records)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.2 * n), squeeze=False)
    axes = axes[:, 0]

    for ax, record, pred in zip(axes, records, predictions):
        h = record["history"]
        future = pred["future_true_model"]
        pred_f = pred["future_pred"]
        ctx = h[-min(history_context, len(h)):]
        h_x = np.arange(-len(ctx), 0)
        f_x = np.arange(len(future))

        ax.plot(h_x, ctx, color="0.75", lw=1.5, label="History tail")
        ax.plot(f_x, future, color="black", lw=2, label="Actual future")
        ax.plot(f_x, pred_f, color="tab:orange", ls="--", lw=2, label="Predicted future")
        ax.axvline(0, color="tab:blue", ls=":", lw=1.2)
        ax.set_title(
            f"{domain} | sample {record['sample_idx']} "
            f"| RMSE={pred['rmse']:.4f} | relRMSE={pred['relative_rmse']:.4f} "
            f"| l={pred['lengthscale']:.4f}"
        )
        ax.set_xlabel("time index relative to T1")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.suptitle(f"{domain}: kernel operator forecast", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if show:
        plt.show()
    return fig
