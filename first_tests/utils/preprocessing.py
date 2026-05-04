"""
Preprocessing utilities for preparing raw time series for the kernel pipeline.

Functions:
    slice_series   — chop a single long series into history/future chunks
    slice_pretrain — slice all examples from build_examples output
    slice_domain   — load + slice every pretrain subset in a domain
"""

from __future__ import annotations

import gc
from typing import Any, Sequence

import numpy as np
from tqdm.auto import tqdm

from utils.config import DATASETS, PRETRAIN_BY_DOMAIN
from utils.data import build_examples


# ===================================================================
# SLICE A SINGLE LONG SERIES INTO TRAINING CHUNKS
# ===================================================================

def slice_series(
    series: np.ndarray,
    *,
    future_len: int = 200,
    min_history: int = 1_000,
    max_history: int = 10_000,
    min_future: int = 200,
    max_future: int = 1_000,
    seed: int = 0,
    start_idx: int = 0,
) -> list[dict[str, Any]]:
    """
    Chop a single long series into non-overlapping history chunks,
    each followed by a future window.

    Parameters
    ----------
    series : 1-D array
        The full time series.
    future_len : int
        Fixed future length per chunk, clamped to
        min(max_future, history_len // 3).
    min_history, max_history : int
        Range for random history lengths.
    min_future, max_future : int
        Bounds on future length.
    seed : int
        Random seed for reproducible chunking.
    start_idx : int
        First sample_idx to assign (incremented for each chunk).

    Returns
    -------
    list[dict]
        Each dict has keys:
        - 'sample_idx': int (globally unique, sequential)
        - 'history': 1-D array
        - 'future': 1-D array
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    rng = np.random.default_rng(seed)

    min_chunk = min_history + min_future
    if n < min_chunk:
        return []

    chunks: list[dict[str, Any]] = []
    pos = 0
    idx = start_idx

    while pos + min_chunk <= n:
        remaining = n - pos
        h_max = min(max_history, remaining - min_future)
        h_max = max(h_max, min_history)
        if h_max < min_history:
            break

        h_len = rng.integers(min_history, h_max + 1)

        f_len = min(future_len, max_future, h_len // 3)
        if f_len < min_future:
            pos += h_len
            continue

        if pos + h_len + f_len > n:
            f_len = n - pos - h_len
            if f_len < min_future:
                break

        history = series[pos : pos + h_len].copy()
        future  = series[pos + h_len : pos + h_len + f_len].copy()

        chunks.append({
            "sample_idx": idx,
            "history": history,
            "future": future,
        })
        idx += 1
        pos += h_len

    return chunks


# ===================================================================
# SLICE MULTIPLE SERIES (from build_examples output)
# ===================================================================

def slice_pretrain(
    examples: list[dict[str, Any]],
    *,
    future_len: int = 200,
    min_history: int = 1_000,
    max_history: int = 10_000,
    min_future: int = 200,
    max_future: int = 1_000,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Slice all pretrain examples into history/future chunks.

    Parameters
    ----------
    examples : list[dict]
        Output of build_examples on a pretrain subset.
    future_len : int
        Fixed future length per chunk.
    min_history, max_history : int
        Range for random history lengths.
    min_future, max_future : int
        Bounds on future length.
    seed : int
        Random seed.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    list[dict]
        Flat list of chunks with sequential integer sample_idx (0, 1, 2, …).
    """
    all_chunks: list[dict[str, Any]] = []
    rng_master = np.random.default_rng(seed)
    next_idx = 0  # global counter across all series

    iterator = tqdm(examples, desc="slicing series", disable=not verbose)
    for ex in iterator:
        series = ex["history"]

        series_seed = int(rng_master.integers(0, 2**31))

        chunks = slice_series(
            series,
            future_len=future_len,
            min_history=min_history,
            max_history=max_history,
            min_future=min_future,
            max_future=max_future,
            seed=series_seed,
            start_idx=next_idx,
        )
        all_chunks.extend(chunks)
        next_idx += len(chunks)
        iterator.set_postfix(chunks=len(all_chunks))

    if verbose:
        total_series = len(examples)
        skipped = sum(1 for ex in examples
                      if len(ex["history"]) < min_history + min_future)
        print(f"\nSlicing done — {total_series} series → {len(all_chunks)} chunks"
              f" ({skipped} series too short, skipped)")

    return all_chunks


# ===================================================================
# SLICE AN ENTIRE DOMAIN (streaming, memory-friendly)
# ===================================================================

def slice_domain(
    domain: str,
    *,
    future_len: int = 200,
    min_history: int = 1_000,
    max_history: int = 10_000,
    min_future: int = 200,
    max_future: int = 1_000,
    max_series_per_subset: int = 1_000,
    skip_prefixes: Sequence[str] = (),
    seed: int = 0,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Load and slice every pretrain subset in a domain, one at a time.

    Each subset is downloaded, sliced, and then freed before moving to
    the next — so peak memory is only one subset at a time.

    Parameters
    ----------
    domain : str
        Key in PRETRAIN_BY_DOMAIN, e.g. "Energy", "Transport".
    future_len, min_history, max_history, min_future, max_future : int
        Passed to slice_pretrain.
    max_series_per_subset : int
        Cap on how many series to load per subset (default 1000).
        Keeps huge subsets (london_smart_meters, residential_*) manageable.
    skip_prefixes : sequence of str
        Subset name prefixes to skip, e.g. ("largest",) to skip
        largest_2017 … largest_2021.
    seed : int
        Master random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    list[dict]
        Flat list of all chunks across all subsets, ready for
        prepare_examples.

    Example
    -------
    >>> chunks = slice_domain("Energy", future_len=700,
    ...                       skip_prefixes=("largest",))
    >>> examples = prepare_examples(chunks, history_len=3000, future_len=700)
    """
    if domain not in PRETRAIN_BY_DOMAIN:
        available = ", ".join(sorted(PRETRAIN_BY_DOMAIN.keys()))
        raise ValueError(f"Unknown domain {domain!r}. Available: {available}")

    subsets = PRETRAIN_BY_DOMAIN[domain]
    rng_master = np.random.default_rng(seed)

    all_chunks: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []
    per_subset: list[tuple[str, int, int]] = []  # (name, n_series, n_chunks)

    outer = tqdm(subsets, desc=f"domain={domain}", disable=not verbose)
    for subset_name in outer:
        # skip unwanted subsets
        if any(subset_name.startswith(p) for p in skip_prefixes):
            if verbose:
                tqdm.write(f"  ⏭  {subset_name} (skipped by prefix)")
            continue

        subset_seed = int(rng_master.integers(0, 2**31))
        raw = None
        chunks = None

        try:
            raw = build_examples(
                subset_name,
                start=0,
                stop=max_series_per_subset,
                dataset_name=DATASETS["pretrain"],
            )
            n_series = len(raw)

            chunks = slice_pretrain(
                raw,
                future_len=future_len,
                min_history=min_history,
                max_history=max_history,
                min_future=min_future,
                max_future=max_future,
                seed=subset_seed,
                verbose=False,
            )
            n_chunks = len(chunks)
            all_chunks.extend(chunks)
            per_subset.append((subset_name, n_series, n_chunks))

            outer.set_postfix(
                chunks=len(all_chunks),
                last=subset_name[:20],
            )
            if verbose:
                tqdm.write(
                    f"  ✓  {subset_name}: "
                    f"{n_series} series → {n_chunks} chunks"
                )

        except Exception as e:
            msg = str(e).split("\n")[0][:120]
            failed.append((subset_name, msg))
            if verbose:
                tqdm.write(f"  ✗  {subset_name}: {msg}")
            continue

        finally:
            # free memory immediately
            del raw, chunks
            gc.collect()

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"Domain {domain!r} summary")
        print(f"{'=' * 55}")
        for name, ns, nc in per_subset:
            print(f"  {name:40s}  {ns:>4d} series → {nc:>5d} chunks")
        print(f"  {'─' * 53}")
        total_series = sum(ns for _, ns, _ in per_subset)
        print(f"  {'TOTAL':40s}  {total_series:>4d} series → {len(all_chunks):>5d} chunks")
        if failed:
            print(f"\n  Failed ({len(failed)}):")
            for name, err in failed:
                print(f"    {name}: {err}")

    return all_chunks
