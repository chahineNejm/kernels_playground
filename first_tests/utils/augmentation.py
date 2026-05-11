"""
Data augmentation for time-series training examples.

All transforms operate on **pairs** (history, future) so that the full
series stays consistent.  Each function returns new arrays — originals
are never mutated.

Individual transforms
---------------------
    phase_shift      — circular translation (roll + edge fill)
    smooth           — simple moving-average smoothing
    jitter           — additive Gaussian noise
    crop_or_pad      — force every series to exactly `target_len`;
                       short series are extended by duplicating a tail
                       segment, long ones are cropped from the start

Batch helper
------------
    augment_examples — apply any combination of the above to a list of
                       example dicts (the format returned by
                       ``prepare_examples``).  Only the *train* set
                       should be augmented.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# ===================================================================
# PHASE SHIFT  (circular translation with edge-value fill)
# ===================================================================

def phase_shift(
    x: np.ndarray,
    shifts: Sequence[int] = (1, 3, 10),
    rng: np.random.Generator | None = None,
    fill: str = "edge",
) -> np.ndarray:
    """
    Translate a 1-D series by a random number of steps.

    A shift value is picked uniformly at random from *shifts*, and the
    direction (left or right) is also random.

    Parameters
    ----------
    x : 1-D array
    shifts : sequence of int
        Candidate shift amounts in number of time steps.
        One is chosen at random, then the sign (left / right) is
        also randomised.  Example: ``[1, 3, 10]``.
    rng : numpy Generator, optional
    fill : str
        How to fill the vacated positions after the shift:

        - ``"edge"``    — repeat the nearest edge value (default).
        - ``"wrap"``    — circular / periodic wrap-around.
        - ``"reflect"`` — mirror the series at the boundary.

    Returns
    -------
    np.ndarray   same length as *x*.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if rng is None:
        rng = np.random.default_rng()
    n = len(x)
    if n == 0:
        return x.copy()

    abs_shift = int(rng.choice(shifts))
    sign = rng.choice([-1, 1])
    shift = sign * min(abs_shift, n - 1)  # clamp so we never shift beyond length

    if shift == 0:
        return x.copy()

    if fill == "wrap":
        return np.roll(x, shift)

    out = np.empty_like(x)
    if shift > 0:
        # shift right → vacated left positions
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

    Uses ``np.convolve`` with mode ``'same'`` so that the output
    length equals the input length.  Edge values are less smoothed
    because the window is clipped there.

    Parameters
    ----------
    x : 1-D array
    window : int
        Kernel width (should be odd for a symmetric window).
        1 = no-op.

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
        Standard deviation of the noise **relative to the standard
        deviation of x**.  E.g. ``sigma=0.05`` means the noise std is
        5 % of the series std.
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
        scale = sigma  # series is nearly constant → use absolute sigma
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

    - If ``len(x) >= target_len``: crop from the **start** (keep the
      most recent ``target_len`` points — typical for time series).
    - If ``min_len <= len(x) < target_len``: extend by duplicating a
      random contiguous segment from the series and appending it.
    - If ``len(x) < min_len``: return ``None`` (series too short to
      salvage).

    Parameters
    ----------
    x : 1-D array
    target_len : int
        Desired output length.
    min_len : int, optional
        Minimum acceptable raw length.  Series shorter than this are
        rejected (returns None).  Defaults to ``target_len // 2``.
    rng : numpy Generator, optional

    Returns
    -------
    np.ndarray of length ``target_len``, or ``None`` if too short.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if rng is None:
        rng = np.random.default_rng()
    if min_len is None:
        min_len = target_len // 2

    n = len(x)

    if n >= target_len:
        # crop from the start (keep tail)
        return x[n - target_len :].copy()

    if n < min_len:
        return None

    # --- pad by duplicating a tail segment -----------------------
    deficit = target_len - n
    # pick a random starting point in x for the segment to copy
    max_start = max(0, n - deficit)
    seg_start = int(rng.integers(0, max_start + 1))
    segment = x[seg_start : seg_start + deficit]

    # if the segment is shorter than the deficit, tile it
    if len(segment) < deficit:
        reps = (deficit // len(segment)) + 1
        segment = np.tile(segment, reps)[:deficit]

    return np.concatenate([x, segment])


# ===================================================================
# BATCH HELPER — augment prepared train examples
# ===================================================================

def augment_train(
    train_records: list[dict[str, Any]],
    *,
    phase_shifts: Sequence[int] | None = None,
    phase_fill: str = "edge",
    smooth_window: int = 1,
    jitter_sigma: float = 0.0,
    crop_target_len: int | None = None,
    crop_min_len: int | None = None,
    seed: int = 0,
    feature_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Augment **prepared** train examples (output of ``prepare_examples``).

    Call this on the train split only, after ``prepare_examples`` has
    cleaned, resampled, and normalised everything.  The transforms are
    applied to the normalised arrays (``history_n`` and ``future_n``),
    and ``x_model`` is rebuilt afterwards.

    The transforms are applied in order:
        1. crop_or_pad  (if ``crop_target_len`` is set)
        2. phase_shift  (if ``phase_shifts`` is provided)
        3. smooth       (if ``smooth_window > 1``)
        4. jitter       (if ``jitter_sigma > 0``)

    Parameters
    ----------
    train_records : list[dict]
        Train-only slice of the output of ``prepare_examples``.
        Each dict must have at least ``history_n`` and ``future_n``.
    phase_shifts : list of int, optional
        Candidate shift amounts in time steps, e.g. ``[1, 3, 10]``.
        One is chosen at random per series, direction is also random.
        ``None`` = disabled.
    phase_fill : str
        Fill mode for phase_shift (``"edge"``, ``"wrap"``, ``"reflect"``).
    smooth_window : int
        Moving-average window.  1 = disabled.
    jitter_sigma : float
        Noise level relative to series std.  0 = disabled.
    crop_target_len : int, optional
        If set, every series is cropped/padded to this length.
        Series shorter than ``crop_min_len`` are dropped.
    crop_min_len : int, optional
        Minimum acceptable length for crop_or_pad.
        Defaults to ``crop_target_len // 2`` if crop_target_len is set.
    seed : int
        Master random seed.
    feature_fn : callable, optional
        Same as in ``prepare_examples``.  If provided, ``x_model`` is
        rebuilt by calling ``feature_fn(record)`` on each augmented
        record.  If None, ``x_model`` is set to the augmented
        ``history_n``.

    Returns
    -------
    list[dict]
        Augmented copies of the train records (originals untouched).
        May be shorter than the input if ``crop_or_pad`` rejects some
        series.

    Example
    -------
    >>> examples = prepare_examples(raw, history_len=3000, future_len=700)
    >>> train = [examples[i] for i in train_idx]
    >>> test  = [examples[i] for i in test_idx]
    >>> train_aug = augment_train(
    ...     train,
    ...     phase_shifts=[1, 3, 10],
    ...     smooth_window=5,
    ...     jitter_sigma=0.03,
    ...     seed=42,
    ... )
    >>> result = train_eval(train_examples=train_aug, test_examples=test)
    """
    rng = np.random.default_rng(seed)
    augmented: list[dict[str, Any]] = []

    for rec in train_records:
        new_rec = dict(rec)  # shallow copy
        skip = False

        for key in ("history_n", "future_n"):
            if key not in new_rec:
                continue
            arr = np.asarray(new_rec[key], dtype=np.float64).ravel()

            # 1. crop / pad
            if crop_target_len is not None:
                result = crop_or_pad(
                    arr, crop_target_len, min_len=crop_min_len, rng=rng,
                )
                if result is None:
                    skip = True
                    break
                arr = result

            # 2. phase shift
            if phase_shifts is not None:
                arr = phase_shift(arr, shifts=phase_shifts,
                                  rng=rng, fill=phase_fill)

            # 3. smooth
            if smooth_window > 1:
                arr = smooth(arr, window=smooth_window)

            # 4. jitter
            if jitter_sigma > 0:
                arr = jitter(arr, sigma=jitter_sigma, rng=rng)

            new_rec[key] = arr

        if skip:
            continue

        # rebuild x_model from the augmented history_n
        if feature_fn is not None:
            new_rec["x_model"] = feature_fn(new_rec)
        else:
            new_rec["x_model"] = new_rec["history_n"]

        augmented.append(new_rec)

    return augmented
