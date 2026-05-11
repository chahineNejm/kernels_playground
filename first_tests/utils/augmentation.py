"""
Data augmentation and uniform-length cleaning for time-series examples.

Dataset cleaning
----------------
    uniform_length — crop or duplicate-pad every series to a fixed
                     length.  **Single entry-point** for resizing —
                     call on build_examples output before anything else.
    crop_or_pad    — low-level helper used by uniform_length.

Augmentation transforms (each returns a new array, same length)
---------------------------------------------------------------
    apply_shift    — translate by a fixed number of steps
    smooth         — centred moving-average
    jitter         — additive Gaussian noise

Batch helper
------------
    augment_train  — full Cartesian grid of (shift × smooth × jitter)
                     on the *train* slice.  Original is always kept.

Typical pipeline
----------------
    raw      = build_examples(...)
    fixed    = uniform_length(raw, target_len=3000, min_len=1500)
    prepared = prepare_examples(fixed)
    train    = [prepared[i] for i in train_idx]
    test     = [prepared[i] for i in test_idx]
    train    = augment_train(train, phase_shifts=[1,3,10], ...)
    result   = train_eval(train_examples=train, test_examples=test)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# ===================================================================
# SHIFT  (deterministic, fixed step count)
# ===================================================================

def apply_shift(
    x: np.ndarray,
    shift: int,
    fill: str = "edge",
) -> np.ndarray:
    """
    Translate a 1-D series by exactly *shift* steps.

    Positive *shift* → shift right (recent values move later).
    Negative *shift* → shift left.

    Parameters
    ----------
    x : 1-D array
    shift : int
        Number of steps to shift (signed).
    fill : str
        How to fill the vacated positions:

        - ``"edge"``    — repeat the nearest edge value (default).
        - ``"wrap"``    — circular / periodic wrap-around.
        - ``"reflect"`` — mirror the series at the boundary.

    Returns
    -------
    np.ndarray   same length as *x*.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n == 0 or shift == 0:
        return x.copy()

    # clamp so we never exceed the series length
    shift = max(-n + 1, min(shift, n - 1))

    if fill == "wrap":
        return np.roll(x, shift)

    out = np.empty_like(x)
    if shift > 0:
        out[shift:] = x[: n - shift]
        if fill == "edge":
            out[:shift] = x[0]
        elif fill == "reflect":
            pad = x[1 : shift + 1][::-1]
            out[: len(pad)] = pad
            if len(pad) < shift:
                out[: shift - len(pad)] = x[0]
    else:
        abs_s = -shift
        out[: n - abs_s] = x[abs_s:]
        if fill == "edge":
            out[n - abs_s :] = x[-1]
        elif fill == "reflect":
            pad = x[-(abs_s + 1) : -1][::-1]
            out[n - len(pad) :] = pad
            if len(pad) < abs_s:
                out[n - abs_s : n - len(pad)] = x[-1]

    return out


# ===================================================================
# SMOOTHING  (simple moving average)
# ===================================================================

def smooth(
    x: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    Smooth a 1-D series with a centred moving average.

    Parameters
    ----------
    x : 1-D array
    window : int
        Kernel width (should be odd for a symmetric window).  1 = no-op.

    Returns
    -------
    np.ndarray   same length as *x*.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if window <= 1 or len(x) <= 1:
        return x.copy()
    window = min(window, len(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# ===================================================================
# JITTER  (additive Gaussian noise)
# ===================================================================

def jitter(
    x: np.ndarray,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add i.i.d. Gaussian noise to a 1-D series.

    Parameters
    ----------
    x : 1-D array
    sigma : float
        Standard deviation of the noise **relative to the std of x**.
        E.g. ``sigma=0.05`` → noise std is 5 % of the series std.
    rng : numpy Generator, optional

    Returns
    -------
    np.ndarray   same length as *x*.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if rng is None:
        rng = np.random.default_rng()
    if len(x) == 0 or sigma <= 0:
        return x.copy()
    scale = sigma * np.std(x)
    if scale < 1e-12:
        scale = sigma
    return x + rng.normal(0.0, scale, size=len(x))


# ===================================================================
# CROP / PAD  (force uniform length)
# ===================================================================

def crop_or_pad(
    x: np.ndarray,
    target_len: int,
    min_len: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """
    Force a 1-D series to exactly ``target_len`` points.

    - ``len(x) >= target_len``: crop from the start (keep tail).
    - ``min_len <= len(x) < target_len``: duplicate a random segment
      and append it.
    - ``len(x) < min_len``: return ``None`` (too short).

    Parameters
    ----------
    x : 1-D array
    target_len : int
    min_len : int, optional
        Defaults to ``target_len // 2``.
    rng : numpy Generator, optional

    Returns
    -------
    np.ndarray of length ``target_len``, or ``None``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if rng is None:
        rng = np.random.default_rng()
    if min_len is None:
        min_len = target_len // 2

    n = len(x)
    if n >= target_len:
        return x[n - target_len :].copy()
    if n < min_len:
        return None

    deficit = target_len - n
    max_start = max(0, n - deficit)
    seg_start = int(rng.integers(0, max_start + 1))
    segment = x[seg_start : seg_start + deficit]
    if len(segment) < deficit:
        reps = (deficit // len(segment)) + 1
        segment = np.tile(segment, reps)[:deficit]

    return np.concatenate([x, segment])


# ===================================================================
# UNIFORM LENGTH — crop/pad a whole dataset in one place
# ===================================================================

def uniform_length(
    examples: list[dict[str, Any]],
    target_len: int,
    min_len: int | None = None,
    seed: int = 0,
    keys: Sequence[str] = ("history", "future"),
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Force every series in *examples* to exactly ``target_len`` points.

    This is the **single entry-point** for resizing raw series.  Call
    it on the output of ``build_examples`` before anything else
    (normalization, augmentation, train/test split).

    Rules per series (applied independently to each key):
        - ``len >= target_len`` → crop from the start (keep tail).
        - ``min_len <= len < target_len`` → duplicate a contiguous
          segment and append it until the series reaches target_len.
        - ``len < min_len`` → the whole example is **dropped**.

    Parameters
    ----------
    examples : list[dict]
        Output of ``build_examples``.
    target_len : int
        Desired length for every series.
    min_len : int, optional
        Shortest acceptable raw length.  Examples where *any* key is
        shorter than this are dropped.  Defaults to ``target_len // 2``.
    seed : int
        Random seed (only affects the segment chosen for padding).
    keys : sequence of str
        Which array keys to resize.  Default ``("history", "future")``.
    verbose : bool
        Print a one-line summary when done.

    Returns
    -------
    list[dict]
        Shallow copies with every listed key exactly ``target_len``
        long.  May be shorter than the input if some examples were
        dropped.

    Example
    -------
    >>> raw = build_examples("electricity_H_long", start=0, stop=1000)
    >>> fixed = uniform_length(raw, target_len=3000, min_len=1500)
    >>> # every fixed[i]["history"] and fixed[i]["future"] is length 3000
    """
    rng = np.random.default_rng(seed)
    if min_len is None:
        min_len = target_len // 2

    out: list[dict[str, Any]] = []
    dropped = 0

    for rec in examples:
        new_rec = dict(rec)
        skip = False

        for key in keys:
            if key not in new_rec:
                continue
            arr = np.asarray(new_rec[key], dtype=np.float64).ravel()
            result = crop_or_pad(arr, target_len, min_len=min_len, rng=rng)
            if result is None:
                skip = True
                break
            new_rec[key] = result

        if skip:
            dropped += 1
            continue
        out.append(new_rec)

    if verbose:
        print(
            f"uniform_length: {len(examples)} examples → {len(out)} kept, "
            f"{dropped} dropped (target={target_len}, min={min_len})"
        )
    return out


# ===================================================================
# BATCH HELPER — augment prepared train examples (full grid)
# ===================================================================

def augment_train(
    train_records: list[dict[str, Any]],
    *,
    phase_shifts: Sequence[int] | None = None,
    phase_fill: str = "edge",
    smooth_window: int = 1,
    jitter_sigma: float = 0.0,
    seed: int = 0,
    feature_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Augment **prepared** train examples by producing the full
    **Cartesian product** of all enabled transforms.

    Series must already be at uniform length (via ``uniform_length``
    then ``prepare_examples``) before calling this.

    For each original record, every combination of (shift × smooth ×
    jitter) is produced.  The original (0 shift, no smooth, no jitter)
    is always included.

    Grid axes
    ---------
    - **shift**:  ``[0] + phase_shifts``.  0 = no shift (always
      present).  Pass negative values explicitly for left shifts,
      e.g. ``[1, -1, 3, -3, 10, -10]``.
    - **smooth**: ``[False, True]`` if ``smooth_window > 1``,
      else ``[False]``.
    - **jitter**: ``[False, True]`` if ``jitter_sigma > 0``,
      else ``[False]``.

    Example sizes
    -------------
    ``phase_shifts=[1,3,10], smooth_window=5, jitter_sigma=0.03``
    → 4 shifts × 2 smooth × 2 jitter = **16×** per record.

    ``phase_shifts=[1,-1,3,-3,10,-10], smooth_window=5, jitter_sigma=0.03``
    → 7 × 2 × 2 = **28×** per record.

    Application order: shift → smooth → jitter.

    Parameters
    ----------
    train_records : list[dict]
        Train-only slice of ``prepare_examples`` output.
    phase_shifts : list of int, optional
        Shift amounts (signed).  ``None`` = no shift axis (only 0).
    phase_fill : str
        Fill mode (``"edge"``, ``"wrap"``, ``"reflect"``).
    smooth_window : int
        Moving-average window.  1 = smooth axis disabled.
    jitter_sigma : float
        Noise level (relative to series std).  0 = jitter axis disabled.
    seed : int
        Random seed (used by jitter).
    feature_fn : callable, optional
        Rebuilds ``x_model``.  If None, ``x_model = history_n``.

    Returns
    -------
    list[dict]
        All grid combinations for every input record.

    Example
    -------
    >>> train_aug = augment_train(
    ...     train,
    ...     phase_shifts=[1, 3, 10],
    ...     smooth_window=5,
    ...     jitter_sigma=0.03,
    ...     seed=42,
    ... )
    >>> # 16× larger than train
    """
    rng = np.random.default_rng(seed)

    # --- build grid axes -----------------------------------------
    shift_axis = [0] + (list(phase_shifts) if phase_shifts is not None else [])
    smooth_axis = [False, True] if smooth_window > 1 else [False]
    jitter_axis = [False, True] if jitter_sigma > 0 else [False]

    result: list[dict[str, Any]] = []

    for rec in train_records:

        # --- full grid -------------------------------------------
        for s in shift_axis:
            for do_smooth in smooth_axis:
                for do_jitter in jitter_axis:

                    new_rec = dict(rec)

                    for key in ("history_n", "future_n"):
                        if key not in new_rec:
                            continue
                        arr = np.asarray(new_rec[key], dtype=np.float64).ravel()

                        if s != 0:
                            arr = apply_shift(arr, s, fill=phase_fill)
                        if do_smooth:
                            arr = smooth(arr, window=smooth_window)
                        if do_jitter:
                            arr = jitter(arr, sigma=jitter_sigma, rng=rng)
                        
                        mu = float(np.mean(arr))
                        sigma = float(np.std(arr))
                        if sigma < 1e-8:
                            sigma = 1.0
                        arr = (arr - mu) / sigma
                        new_rec[key] = arr

                    # rebuild x_model
                    if feature_fn is not None:
                        new_rec["x_model"] = feature_fn(new_rec)
                    else:
                        new_rec["x_model"] = new_rec["history_n"]

                    result.append(new_rec)

    return result
