"""
Data loading, cleaning, and quick‑access helpers for GiftEvalParquet.

Core helpers (from the original notebook):
    clean_series, extract_history_future, normalize_by_history, resample_series

Fast dataset access:
    load_gift_dataset   – cached HuggingFace loader (returns a datasets.Dataset)
    quick_peek          – grab a single sample's history/future arrays
    dataset_summary     – length / NaN / range stats for the first N samples
    build_examples      – full pipeline: load → clean → normalise → split
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from datasets import load_dataset

from utils.config import DEFAULT_CONFIG


# ═══════════════════════════════════════════════════════════════
# CLEANING / TRANSFORM HELPERS
# ═══════════════════════════════════════════════════════════════

def clean_series(x: np.ndarray) -> np.ndarray:
    """Interpolate NaN / Inf values in a 1‑D series."""
    x = np.asarray(x, dtype=float).ravel().copy()
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if np.all(finite):
        return x
    if not np.any(finite):
        return np.zeros_like(x)
    idx = np.arange(x.size)
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return x


def extract_history_future(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (history, future) arrays from a GiftEval sample dict."""
    if "history_value" in sample and "future_value" in sample:
        return clean_series(sample["history_value"]), clean_series(sample["future_value"])
    if "target" in sample:
        x = clean_series(sample["target"])
        split = int(round(0.8 * len(x)))
        return x[:split], x[split:]
    raise KeyError(f"Unsupported sample format. Keys: {list(sample.keys())}")


def normalize_by_history(
    history: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Z‑normalise history & future using history statistics."""
    mu = float(np.mean(history))
    sigma = float(np.std(history))
    if sigma < 1e-8:
        sigma = 1.0
    return (history - mu) / sigma, (future - mu) / sigma, mu, sigma


def resample_series(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample *x* to *target_len* points."""
    x = np.asarray(x, dtype=float).ravel()
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if x.size == target_len:
        return x.copy()
    if x.size == 1:
        return np.full(target_len, x[0], dtype=float)
    src = np.linspace(0.0, 1.0, x.size)
    dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst, src, x)


# ═══════════════════════════════════════════════════════════════
# FAST DATASET ACCESS
# ═══════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=16)
def load_gift_dataset(
    config: str = "electricity_H_long",
    n_samples: int | None = None,
    dataset_name: str = DEFAULT_CONFIG["DATASET_NAME"],
) -> Any:
    """
    Load a GiftEvalParquet config from HuggingFace (result is cached in‑process).

    Parameters
    ----------
    config : str
        Dataset configuration name (e.g. "electricity_H_long").
    n_samples : int or None
        Limit the number of rows loaded. None → full split.
    dataset_name : str
        HuggingFace dataset identifier.

    Returns
    -------
    datasets.Dataset
    """
    split = f"train[:{n_samples}]" if n_samples else "train"
    return load_dataset(dataset_name, config, split=split)


def quick_peek(
    index: int = 0,
    config: str = "electricity_H_long",
    normalize: bool = False,
    dataset_name: str = DEFAULT_CONFIG["DATASET_NAME"],
) -> dict[str, np.ndarray | float]:
    """
    Grab a single sample and return its history / future arrays.

    Only loads the one row you ask for — no bulk download.

    Returns a dict with keys:
        history, future, (and if normalize=True: history_n, future_n, mu, sigma)
    """
    ds = load_dataset(dataset_name, config, split=f"train[{index}:{index + 1}]")
    sample = ds[0]
    history, future = extract_history_future(sample)
    result: dict[str, Any] = {"history": history, "future": future}
    if normalize:
        h_n, f_n, mu, sigma = normalize_by_history(history, future)
        result.update(history_n=h_n, future_n=f_n, mu=mu, sigma=sigma)
    return result


def dataset_summary(
    config: str = "electricity_H_long",
    n_samples: int = 500,
    max_display: int = 20,
) -> list[dict[str, Any]]:
    """
    Return quick stats for the first *max_display* samples:
    history_len, future_len, history_mean/std, future_mean/std, n_nans.
    """
    ds = load_gift_dataset(config, n_samples)
    rows: list[dict[str, Any]] = []
    for i in range(min(len(ds), max_display)):
        sample = ds[i]
        h, f = extract_history_future(sample)
        rows.append({
            "index": i,
            "item_id": sample.get("item_id", "?"),
            "history_len": len(h),
            "future_len": len(f),
            "history_mean": float(np.mean(h)),
            "history_std": float(np.std(h)),
            "future_mean": float(np.mean(f)),
            "future_std": float(np.std(f)),
            "history_nans": int(np.sum(~np.isfinite(h))),
            "future_nans": int(np.sum(~np.isfinite(f))),
        })
    return rows


def build_examples(
    config: str = "electricity_H_long",
    n_samples: int = DEFAULT_CONFIG["N_SAMPLES_TO_LOAD"],
    subsample_step: int = 5,
    dataset_name: str = DEFAULT_CONFIG["DATASET_NAME"],
) -> list[dict[str, Any]]:
    """
    Full pipeline: load → clean → align lengths → normalise.

    Returns a list of dicts, each containing:
        sample_idx, history, future,
        history_model, future_model,
        history_n, future_n, mu, sigma
    """
    ds = load_gift_dataset(config, n_samples, dataset_name)
    ds = ds.select(range(0, len(ds), subsample_step))

    raw: list[dict[str, Any]] = []
    for idx, sample in enumerate(ds):
        h, f = extract_history_future(sample)
        if len(h) == 0 or len(f) == 0:
            continue
        raw.append({"sample_idx": idx, "history": h, "future": f})

    if not raw:
        return []

    history_len = min(len(r["history"]) for r in raw)
    future_len = min(len(r["future"]) for r in raw)

    examples: list[dict[str, Any]] = []
    for rec in raw:
        hm = resample_series(rec["history"], history_len)
        fm = resample_series(rec["future"], future_len)
        hn, fn, mu, sigma = normalize_by_history(hm, fm)
        examples.append({
            "sample_idx": rec["sample_idx"],
            "history": rec["history"],
            "future": rec["future"],
            "history_model": hm,
            "future_model": fm,
            "history_n": hn,
            "future_n": fn,
            "mu": mu,
            "sigma": sigma,
        })
    return examples
