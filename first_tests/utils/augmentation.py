"""
Data augmentation for time-series training examples.

Augmentation **multiplies** the training set — the original series is
always kept, and each transform produces additional copies.

Individual transforms (each returns a new array, same length as input)
----------------------------------------------------------------------
    apply_shift    — translate by a fixed number of steps
    smooth         — centred moving-average
    jitter         — additive Gaussian noise
    crop_or_pad    — force a series to exactly ``target_len``

Batch helper
------------
    augment_train  — take the *train* slice of ``prepare_examples``
                     output, return an expanded list with the original
                     records **plus** all augmented copies.
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
# INTERNAL: build one augmented copy of a prepared record
# ===================================================================

def _augment_record(
    rec: dict[str, Any],
    transform_fn,
    feature_fn: Any | None,
) -> dict[str, Any]:
    """
    Apply *transform_fn* to history_n and future_n, rebuild x_model.
    Returns a new dict (shallow copy + replaced arrays).
    """
    new = dict(rec)
    for key in ("history_n", "future_n"):
        if key in new:
            new[key] = transform_fn(np.asarray(new[key], dtype=np.float64).ravel())
    if feature_fn is not None:
        new["x_model"] = feature_fn(new)
    else:
        new["x_model"] = new["history_n"]
    return new


# ===================================================================
# BATCH HELPER — augment prepared train examples (multiplicative)
# ===================================================================

def augment_train(
    train_records: list[dict[str, Any]],
    *,
    phase_shifts: Sequence[int] | None = None,
    phase_fill: str = "edge",
    include_random_shift: bool = False,
    smooth_window: int = 1,
    jitter_sigma: float = 0.0,
    crop_target_len: int | None = None,
    crop_min_len: int | None = None,
    seed: int = 0,
    feature_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Augment **prepared** train examples by producing additional copies.

    The **original** record is always kept.  Each enabled transform
    adds extra copies on top:

    - **phase_shifts** ``[1, 3, 10]``: for each value, two copies are
      produced (``+shift`` and ``-shift``).  With 3 values that's
      6 extra copies per original.
    - **include_random_shift**: if True, one more copy with a shift
      picked at random from ``phase_shifts`` and a random sign.
    - **smooth_window > 1**: one extra smoothed copy.
    - **jitter_sigma > 0**: one extra jittered copy.

    So with ``phase_shifts=[1,3,10]``, ``smooth_window=5``,
    ``jitter_sigma=0.03`` you get **1 original + 6 shifted + 1 smooth
    + 1 jitter = 9×** the training set.

    crop_or_pad is applied to *every* record (original and copies)
    before any other transform, if ``crop_target_len`` is set.

    Parameters
    ----------
    train_records : list[dict]
        Train-only slice of ``prepare_examples`` output.
    phase_shifts : list of int, optional
        Step counts for deterministic shifts, e.g. ``[1, 3, 10]``.
        Both +s and -s are produced for each value.
        ``None`` = no shift augmentation.
    phase_fill : str
        Fill mode (``"edge"``, ``"wrap"``, ``"reflect"``).
    include_random_shift : bool
        Add one extra copy with a randomly chosen shift from the list.
    smooth_window : int
        Moving-average window for the smoothed copy.  1 = no smooth copy.
    jitter_sigma : float
        Noise level for the jittered copy.  0 = no jitter copy.
    crop_target_len : int, optional
        Force all series (original + copies) to this length.
    crop_min_len : int, optional
        Reject series shorter than this.  Defaults to ``crop_target_len // 2``.
    seed : int
        Random seed (only used by jitter and include_random_shift).
    feature_fn : callable, optional
        Rebuilds ``x_model``.  If None, ``x_model = history_n``.

    Returns
    -------
    list[dict]
        Original records + all augmented copies, ready for
        ``train_eval(train_examples=...)``.

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
    >>> # train_aug is 9× larger than train
    >>> result = train_eval(train_examples=train_aug, test_examples=test)
    """
    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []

    for rec in train_records:
        # --- optional crop/pad on the original first ---------------
        if crop_target_len is not None:
            base = dict(rec)
            skip = False
            for key in ("history_n", "future_n"):
                if key not in base:
                    continue
                arr = np.asarray(base[key], dtype=np.float64).ravel()
                cropped = crop_or_pad(arr, crop_target_len,
                                      min_len=crop_min_len, rng=rng)
                if cropped is None:
                    skip = True
                    break
                base[key] = cropped
            if skip:
                continue
            # rebuild x_model for the cropped original
            if feature_fn is not None:
                base["x_model"] = feature_fn(base)
            else:
                base["x_model"] = base["history_n"]
        else:
            base = rec

        # 1. always keep the original
        result.append(base)

        # 2. phase-shift copies: +s and -s for every s in the list
        if phase_shifts is not None:
            for s in phase_shifts:
                for sign in (+1, -1):
                    signed_shift = sign * s
                    result.append(_augment_record(
                        base,
                        lambda x, _s=signed_shift: apply_shift(x, _s, fill=phase_fill),
                        feature_fn,
                    ))

            # 2b. optional random-shift copy
            if include_random_shift:
                rand_s = int(rng.choice(phase_shifts)) * int(rng.choice([-1, 1]))
                result.append(_augment_record(
                    base,
                    lambda x, _s=rand_s: apply_shift(x, _s, fill=phase_fill),
                    feature_fn,
                ))

        # 3. smoothed copy
        if smooth_window > 1:
            result.append(_augment_record(
                base,
                lambda x: smooth(x, window=smooth_window),
                feature_fn,
            ))

        # 4. jittered copy
        if jitter_sigma > 0:
            result.append(_augment_record(
                base,
                lambda x: jitter(x, sigma=jitter_sigma, rng=rng),
                feature_fn,
            ))

    return result
