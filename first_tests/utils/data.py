"""
Data loading, cleaning, and quick-access helpers for GiftEval datasets.

Works with both:
    - Salesforce/GiftEvalParquet  (eval — 97 named configs)
    - Salesforce/GiftEvalPretrain (pretrain — subsets loaded via data_dir)

Core helpers:
    clean_series, extract_history_future, normalize_by_history, resample_series

Dataset access:
    load_gift_dataset   - cached HuggingFace loader
    quick_peek          - grab a single sample
    dataset_summary     - stats for the first N samples
    build_examples      - load + clean only (raw lengths preserved)
    prepare_examples    - resample to target lengths + normalize (call before training)
    list_pretrain_subsets - discover available pretrain subsets
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from datasets import load_dataset

from utils.config import DEFAULT_CONFIG, DATASETS


# ===================================================================
# CLEANING / TRANSFORM HELPERS
# ===================================================================

def clean_series(x: np.ndarray) -> np.ndarray:
    """Interpolate NaN / Inf values in a 1-D series."""
    x = np.asarray(x, dtype=np.float16).ravel().copy()
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
        x = clean_series(np.asarray(sample["target"], dtype=float))
        return x, np.array([], dtype=float)
    raise KeyError(f"Unsupported sample format. Keys: {list(sample.keys())}")


def normalize_by_history(
    history: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Z-normalise history & future using history statistics."""
    mu = float(np.mean(history))
    sigma = float(np.std(history))
    if sigma < 1e-8:
        sigma = 1.0
    history_n = (history - mu) / sigma
    future_n = (future - mu) / sigma
    return history_n, future_n, mu, sigma


def resample_series(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample x to target_len points."""
    x = np.asarray(x, dtype=np.float16).ravel()
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if x.size == target_len:
        return x.copy()
    if x.size == 1:
        return np.full(target_len, x[0], dtype=float)
    src_grid = np.linspace(0.0, 1.0, x.size)
    dst_grid = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst_grid, src_grid, x)


# ===================================================================
# CONFIG NAME RESOLUTION
# ===================================================================

def resolve_config(name: str) -> str:
    """
    Accept either a friendly name ("Energy") or a raw HF config string
    ("electricity_H_long") and return the raw config string.
    """
    configs = DEFAULT_CONFIG.get("CONFIGS", {})
    if name in configs:
        return configs[name]
    return name


# ===================================================================
# PRETRAIN SUBSET DISCOVERY
# ===================================================================

_pretrain_subsets_cache: list[str] | None = None


def list_pretrain_subsets() -> list[str]:
    """
    Discover available subset names in GiftEvalPretrain via HfFileSystem.

    Returns a sorted list like ['SHMETRO', 'alibaba_cluster_trace_2018', ...].
    Result is cached after the first call.
    """
    global _pretrain_subsets_cache
    if _pretrain_subsets_cache is not None:
        return _pretrain_subsets_cache

    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    root_items = fs.ls(f"datasets/{DATASETS['pretrain']}", detail=True)
    subsets = sorted([
        item["name"].split("/")[-1]
        for item in root_items
        if item["type"] == "directory"
        and not item["name"].split("/")[-1].startswith(".")
    ])
    _pretrain_subsets_cache = subsets
    return subsets


# ===================================================================
# INTERNAL: smart load_dataset wrapper
# ===================================================================

def _load_hf(
    dataset_name: str,
    config: str,
    split: str,
) -> Any:
    """
    Wrapper around load_dataset that handles both eval and pretrain.

    - Eval (GiftEvalParquet): config is the HF BuilderConfig name
      e.g. load_dataset("Salesforce/GiftEvalParquet", "electricity_H_long", ...)

    - Pretrain (GiftEvalPretrain): there's only a "default" BuilderConfig,
      but each subset lives in its own subdirectory. We pass config as
      data_dir so only that subset's arrow files are downloaded.
      e.g. load_dataset("Salesforce/GiftEvalPretrain", data_dir="electricity", ...)
    """
    if dataset_name == DATASETS["pretrain"]:
        # config is the subset name (subdirectory), not a BuilderConfig.
        # Use data_files to download only that subset's arrow files.
        # load_dataset with data_files may return a DatasetDict; we
        # grab the first (only) split, then apply any row slicing.
        ds = load_dataset(
            dataset_name,
            data_files=f"{config}/*.arrow",
        )
        # ds is a DatasetDict like {"train": Dataset}
        if hasattr(ds, "keys"):
            actual_split = list(ds.keys())[0]
            ds = ds[actual_split]
        # now apply row slicing from split (e.g. "train[0:100]")
        if "[" in split:
            slice_str = split[split.index("[") + 1 : split.index("]")]
            parts = slice_str.split(":")
            start_idx = int(parts[0]) if parts[0] else 0
            end_idx = int(parts[1]) if len(parts) > 1 and parts[1] else len(ds)
            end_idx = min(end_idx, len(ds))
            ds = ds.select(range(start_idx, end_idx))
        return ds
    else:
        # eval dataset — config is the real BuilderConfig name
        return load_dataset(dataset_name, config, split=split)


# ===================================================================
# FAST DATASET ACCESS
# ===================================================================

@functools.lru_cache(maxsize=16)
def load_gift_dataset(
    config: str = "electricity_H_long",
    n_samples: int | None = None,
    dataset_name: str = DEFAULT_CONFIG["DATASET_NAME"],
) -> Any:
    """
    Load a GiftEval config from HuggingFace (result is cached in-process).

    For eval:     config is an HF config name (e.g. "electricity_H_long")
    For pretrain: config is a subset name   (e.g. "electricity")
    """
    config = resolve_config(config)
    split = f"train[:{n_samples}]" if n_samples else "train"
    return _load_hf(dataset_name, config, split)


def quick_peek(
    index: int = 0,
    config: str = "electricity_H_long",
    normalize: bool = False,
    dataset_name: str = DEFAULT_CONFIG["DATASET_NAME"],
) -> dict[str, np.ndarray | float]:
    """
    Grab a single sample. Only loads one row.
    config can be a friendly name or raw HF config string.
    """
    config = resolve_config(config)
    split = f"train[{index}:{index + 1}]"
    ds = _load_hf(dataset_name, config, split)
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
    dataset_name: str = DEFAULT_CONFIG["DATASET_NAME"],
) -> list[dict[str, Any]]:
    """
    Quick stats for the first max_display samples.
    """
    config = resolve_config(config)
    ds = load_gift_dataset(config, n_samples, dataset_name)
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


# ===================================================================
# LOAD + CLEAN (no normalization, no resampling)
# ===================================================================

def build_examples(
    config: str = "electricity_H_long",
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    dataset_name: str = DEFAULT_CONFIG["DATASET_NAME"],
    extra_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Load and clean samples. No resampling, no normalization.

    Each sample keeps its original lengths. Returns a list of dicts:
        sample_idx  - global row index in the HF dataset
        history     - cleaned 1-D array (original length)
        future      - cleaned 1-D array (original length)
        + any fields listed in extra_keys, pulled from the raw HF sample

    Parameters
    ----------
    config : str
        Friendly name ("Energy") or raw HF config string.
    start, stop : int
        Row range to load. stop defaults to start + 1000.
    step : int
        Keep every step-th row (1 = keep all).
    extra_keys : list[str], optional
        Additional columns to extract from each HF sample dict
        (e.g. ["item_id", "freq", "seasonal_period"]).
        Values are stored as-is in each example dict.
    """
    config = resolve_config(config)
    if stop is None:
        stop = start + 1000
    split = f"train[{start}:{stop}]"
    ds = _load_hf(dataset_name, config, split)

    indices = range(0, len(ds), step)
    ds = ds.select(indices)

    examples: list[dict[str, Any]] = []
    for local_idx, sample in enumerate(ds):
        h, f = extract_history_future(sample)
        if len(h) == 0:
            continue
        global_idx = start + indices[local_idx]
        entry: dict[str, Any] = {
            "sample_idx": global_idx,
            "history": h,
            "future": f,
        }
        if extra_keys:
            for key in extra_keys:
                if key in sample:
                    val = sample[key]
                    entry[key] = np.asarray(val, dtype=float) if isinstance(val, list) else val
        examples.append(entry)
    return examples


# ===================================================================
# RESAMPLE + NORMALIZE (call before training)
# ===================================================================

def prepare_examples(
    examples: list[dict[str, Any]],
    *,
    history_len: int | None = None,
    future_len: int | None = None,
    min_history: int | None = None,
    min_future: int | None = None,
    feature_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Resample to uniform lengths and Z-normalize. Call this on a subset
    of build_examples output right before training.

    Parameters
    ----------
    examples : list[dict]
        Output of build_examples. Each dict must have 'history' and 'future'.
    history_len : int, optional
        Target history length. If None, uses the median across examples.
    future_len : int, optional
        Target future length. If None, uses the median across examples.
    min_history : int, optional
        Drop any sample with history shorter than this (before resampling).
    min_future : int, optional
        Drop any sample with future shorter than this (before resampling).
    feature_fn : callable, optional
        A function  f(example_dict) -> np.ndarray  that builds the kernel
        input from each prepared example.  The dict it receives already has
        history_n, future_n, mu, sigma, plus any extra_keys from build_examples.

        If None, ``x_model`` defaults to ``history_n`` (1-D, same as before).

        Examples::

            # append summary stats to the history
            def with_stats(rec):
                h = rec["history_n"]
                stats = [np.mean(h), np.std(h), np.min(h), np.max(h)]
                return np.concatenate([h, stats])

            # stack multiple channels into (seq_len, n_channels)
            def multivariate(rec):
                return np.column_stack([rec["history_n"], rec["temperature_n"]])

    Returns
    -------
    list[dict] with keys:
        sample_idx, history, future,
        history_model, future_model,
        history_n, future_n, mu, sigma,
        x_model   (kernel input — built by feature_fn or = history_n)
        + any extra keys carried forward from build_examples
    """
    # -- filter by minimum lengths -----------------------------
    filtered = examples
    if min_history is not None:
        filtered = [e for e in filtered if len(e["history"]) >= min_history]
    if min_future is not None:
        filtered = [e for e in filtered if len(e["future"]) >= min_future]

    if not filtered:
        return []

    # -- resolve target lengths --------------------------------
    h_lengths = [len(e["history"]) for e in filtered]
    f_lengths = [len(e["future"]) for e in filtered]

    if history_len is None:
        history_len = int(np.median(h_lengths))
    if future_len is None:
        future_len = int(np.median(f_lengths))

    # -- resample + normalize ----------------------------------
    _core_keys = {"sample_idx", "history", "future"}
    prepared: list[dict[str, Any]] = []
    for rec in filtered:
        hm = resample_series(rec["history"], history_len)
        fm = resample_series(rec["future"], future_len)
        hn, fn, mu, sigma = normalize_by_history(hm, fm)

        entry: dict[str, Any] = {
            "sample_idx": rec["sample_idx"],
            "history": rec["history"],
            "future": rec["future"],
            "history_model": hm,
            "future_model": fm,
            "history_n": hn,
            "future_n": fn,
            "mu": mu,
            "sigma": sigma,
        }

        # carry forward any extra keys from build_examples
        for k, v in rec.items():
            if k not in entry and k not in _core_keys:
                entry[k] = v

        # build kernel input
        if feature_fn is not None:
            entry["x_model"] = feature_fn(entry)
        else:
            entry["x_model"] = hn

        prepared.append(entry)
    return prepared
