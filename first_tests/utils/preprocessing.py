"""
Preprocessing utilities for preparing raw time series for the kernel pipeline.

Main function:
    slice_series  — chop a long pretrain series into history/future chunks
                    compatible with build_examples / prepare_examples output.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from tqdm.auto import tqdm


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
    series_id: int = 0,
) -> list[dict[str, Any]]:
    """
    Chop a single long series into non-overlapping history chunks,
    each followed by a future window.

    The histories do NOT overlap with each other, but the future of
    one chunk may fall within the history of the next chunk (since
    it extends beyond the history boundary).

    Parameters
    ----------
    series : 1-D array
        The full time series (e.g. from a pretrain 'target' field).
    future_len : int
        Fixed future length for every chunk.  Clamped to be at most
        min(max_future, history_len // 3) for each chunk.
    min_history : int
        Minimum history length per chunk (default 1000).
    max_history : int
        Maximum history length per chunk (default 10000).
    min_future, max_future : int
        Bounds on the future length.  The actual future for each chunk
        is min(future_len, max_future, history_len // 3), and must be
        >= min_future or the chunk is skipped.
    seed : int
        Random seed for reproducible chunking.
    series_id : int
        Identifier for this series (used in sample_idx).

    Returns
    -------
    list[dict]
        Each dict has keys: 'sample_idx', 'history', 'future',
        matching the output format of build_examples.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    rng = np.random.default_rng(seed)

    # minimum viable chunk: min_history + min_future
    min_chunk = min_history + min_future
    if n < min_chunk:
        return []

    chunks: list[dict[str, Any]] = []
    pos = 0       # current position — start of next history
    chunk_idx = 0

    while pos + min_chunk <= n:
        # -- draw a random history length -------------------------
        remaining = n - pos
        h_max = min(max_history, remaining - min_future)
        h_max = max(h_max, min_history)  # guard against edge case
        if h_max < min_history:
            break

        h_len = rng.integers(min_history, h_max + 1)

        # -- compute future length --------------------------------
        # future_len capped at max_future and history//3
        f_len = min(future_len, max_future, h_len // 3)
        if f_len < min_future:
            # history too short for a valid future — skip
            pos += h_len
            continue

        # -- check we have room for the future --------------------
        if pos + h_len + f_len > n:
            # not enough data left for the future — try shorter
            f_len = n - pos - h_len
            if f_len < min_future:
                break

        # -- extract ----------------------------------------------
        history = series[pos : pos + h_len].copy()
        future = series[pos + h_len : pos + h_len + f_len].copy()

        chunks.append({
            "sample_idx": (series_id, chunk_idx),
            "history": history,
            "future": future,
        })
        chunk_idx += 1

        # advance position by history length only
        # (future may overlap with next chunk's history)
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

    Takes the output of build_examples (where each 'history' is a full
    pretrain series and 'future' is empty) and produces training chunks
    compatible with prepare_examples.

    Parameters
    ----------
    examples : list[dict]
        Output of build_examples on a pretrain subset.  Each dict must
        have at least 'history' (the full series) and 'sample_idx'.
    future_len : int
        Fixed future length per chunk (default 200).
    min_history, max_history : int
        Range for random history lengths (default 1000–10000).
    min_future, max_future : int
        Bounds on future length (default 200–1000).  Actual future is
        min(future_len, max_future, history//3) per chunk.
    seed : int
        Random seed.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    list[dict]
        Flat list of chunks, each with 'sample_idx', 'history', 'future'.
        Ready to pass to prepare_examples.

    Example
    -------
    >>> raw = build_examples("solar_power", dataset_name=DATASETS["pretrain"])
    >>> chunks = slice_pretrain(raw, future_len=300)
    >>> examples = prepare_examples(chunks)
    >>> result = train_eval(examples, kernel=RBFKernel())
    """
    all_chunks: list[dict[str, Any]] = []
    rng_master = np.random.default_rng(seed)

    iterator = tqdm(examples, desc="slicing series", disable=not verbose)
    for ex in iterator:
        series = ex["history"]
        sid = ex.get("sample_idx", 0)

        # each series gets its own seed derived from the master
        series_seed = int(rng_master.integers(0, 2**31))

        chunks = slice_series(
            series,
            future_len=future_len,
            min_history=min_history,
            max_history=max_history,
            min_future=min_future,
            max_future=max_future,
            seed=series_seed,
            series_id=sid,
        )
        all_chunks.extend(chunks)
        iterator.set_postfix(chunks=len(all_chunks))

    if verbose:
        total_series = len(examples)
        skipped = sum(1 for ex in examples if len(ex["history"]) < min_history + min_future)
        print(f"\nSlicing done — {total_series} series → {len(all_chunks)} chunks"
              f" ({skipped} series too short, skipped)")

    return all_chunks
