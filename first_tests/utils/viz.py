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
        future = pred.get("future_true", pred.get("future_true_resampled"))
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
            f"| relRMSE={pred['relative_rmse']:.4f} | MASE={pred.get('mase', float('nan')):.4f}"
            + (f" | l={pred['lengthscale']:.4f}" if pred.get('lengthscale') is not None else "")
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


# ===================================================================
# FULL EVAL REPORT FROM TrainEvalResult
# ===================================================================

def report_eval(
    result,
    *,
    n_preview: int = 6,
    history_context: int = DEFAULT_CONFIG["HISTORY_CONTEXT_TO_PLOT"],
    show: bool = True,
):
    """
    One-call diagnostic dashboard for a TrainEvalResult.

    Produces (in order):
        1. Summary stats printout
        2. Train / test split bar
        3. Grid of a few test predictions (actual vs predicted)
        4. relRMSE bar chart across all test samples
        5. Residual scatter + histogram (aggregated over all test samples)
        6. RMSE vs relRMSE scatter (per test sample)

    Parameters
    ----------
    result : TrainEvalResult
        Output of pipeline.train_eval.
    n_preview : int
        How many test samples to show in the prediction grid.
    history_context : int
        How much history tail to show in prediction panels.
    show : bool
        Call plt.show() after each figure.

    Returns
    -------
    dict of str -> Figure  (keyed by plot name, for further tweaking)
    """
    domain = result.domain or "eval"
    preds = result.predictions
    figs = {}

    # -- 1. print summary -------------------------------------
    s = result.summary()
    print(f"{'=' * 50}")
    print(f"  {domain}")
    print(f"  train: {s['n_train']}   test: {s['n_test']}")
    ls_str = f"{s['lengthscale']:.4f}" if s.get('lengthscale') is not None else "n/a"
    print(f"  kernel: {s.get('kernel', 'RBF')}   lengthscale: {ls_str}   gamma: {s['gamma']:.2e}")
    print(f"  mean RMSE:      {s['mean_rmse']:.4f}")
    print(f"  mean relRMSE:   {s['mean_relRMSE']:.4f}")
    print(f"  median relRMSE: {s['median_relRMSE']:.4f}")
    print(f"  mean MASE:      {s.get('mean_MASE', float('nan')):.4f}")
    print(f"  median MASE:    {s.get('median_MASE', float('nan')):.4f}")
    print(f"{'=' * 50}")

    # -- 2. train / test split bar -----------------------------
    figs["split"] = plot_train_test_split(
        len(result.examples),
        result.train_indices,
        result.test_indices,
        title=f"{domain} -- train / test split",
        show=show,
    )

    # -- 3. prediction preview grid ----------------------------
    n_show = min(n_preview, len(result.test_records))
    rows_grid = int(np.ceil(n_show / 3))
    fig_prev, axes_prev = plt.subplots(
        rows_grid, 3,
        figsize=(14, 3.5 * rows_grid),
        squeeze=False,
    )
    for i in range(rows_grid * 3):
        ax = axes_prev[i // 3, i % 3]
        if i >= n_show:
            ax.set_visible(False)
            continue
        rec = result.test_records[i]
        pred = preds[i]
        h = rec["history"][-min(history_context, len(rec["history"])):]
        h_x = np.arange(-len(h), 0)
        f_true = pred.get("future_true", pred.get("future_true_resampled"))
        f_x = np.arange(len(f_true))

        ax.plot(h_x, h, color="0.65", lw=1, label="history")
        ax.plot(f_x, f_true, color="black", lw=1.4, label="actual")
        ax.plot(f_x, pred["future_pred"], color="tab:orange", ls="--", lw=1.4, label="predicted")
        ax.axvline(0, color="tab:blue", ls=":", lw=0.8)
        ax.set_title(
            f"sample {rec.get('sample_idx', i)} "
            f"| relRMSE={pred['relative_rmse']:.3f} | MASE={pred.get('mase', float('nan')):.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig_prev.suptitle(f"{domain} -- prediction preview", fontsize=12)
    fig_prev.tight_layout(rect=(0, 0, 1, 0.96))
    if show:
        plt.show()
    figs["prediction_preview"] = fig_prev

    # -- 4. relRMSE bar chart ---------------------------------
    figs["metric_bars"] = plot_metric_summary(
        preds,
        metric="relative_rmse",
        title=f"{domain} -- relRMSE per test sample",
        show=show,
    )

    # -- 4b. MASE bar chart -----------------------------------
    if preds and "mase" in preds[0]:
        figs["mase_bars"] = plot_metric_summary(
            preds,
            metric="mase",
            title=f"{domain} -- MASE per test sample",
            show=show,
        )

    # -- 5. aggregated residuals -------------------------------
    all_true = np.concatenate([p.get("future_true", p.get("future_true_resampled")) for p in preds])
    all_pred = np.concatenate([p["future_pred"] for p in preds])
    figs["residuals"] = plot_residuals(
        all_true, all_pred,
        title=f"{domain} -- residuals (all test samples)",
        show=show,
    )

    # -- 6. RMSE vs relRMSE scatter ----------------------------
    rmses = [p["rmse"] for p in preds]
    rel_rmses = [p["relative_rmse"] for p in preds]
    fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
    ax_sc.scatter(rmses, rel_rmses, s=20, alpha=0.6, color="tab:orange")
    ax_sc.set_xlabel("RMSE")
    ax_sc.set_ylabel("relative RMSE")
    ax_sc.set_title(f"{domain} -- RMSE vs relRMSE")
    ax_sc.grid(True, alpha=0.2)
    fig_sc.tight_layout()
    if show:
        plt.show()
    figs["rmse_vs_relrmse"] = fig_sc

    return figs


def compare_results(
    results: dict,
    *,
    metric: str = "relative_rmse",
    show: bool = True,
) -> plt.Figure:
    """
    Side-by-side box plots comparing a metric across multiple domains.

    Parameters
    ----------
    results : dict[str, TrainEvalResult]
        Output of pipeline.run_all_domains.
    metric : str
        Key in each prediction dict to compare.
    show : bool
        Call plt.show().

    Returns
    -------
    Figure
    """
    labels = []
    data = []
    for domain, result in results.items():
        vals = [p[metric] for p in result.predictions]
        labels.append(f"{domain}\n(n={len(vals)})")
        data.append(vals)

    fig, ax = plt.subplots(figsize=(max(4, 2.5 * len(labels)), 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across domains")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
