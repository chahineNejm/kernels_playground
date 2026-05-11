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
    """Translate a 1-D series by exactly *shift* steps."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n == 0 or shift == 0:
        return x.copy()

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
    """Smooth a 1-D series with a centred moving average."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if window <= 1 or len(x) <= 1:
        return x.copy()
    
    window = min(window, len(x))
    kernel = np.ones(window) / window
    
    # FIX: Edge-pad the array before convolving to prevent the 
    # start/end of the series from artificially dipping toward zero.
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, pad_width=(pad_left, pad_right), mode='edge')
    
    return np.convolve(x_padded, kernel, mode="valid")


# ===================================================================
# JITTER  (additive Gaussian noise)
# ===================================================================

def jitter(
    x: np.ndarray,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to a 1-D series."""
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
    pad_direction: str = "left",  # FIX: Added padding direction
) -> np.ndarray | None:
    """Force a 1-D series to exactly ``target_len`` points."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if rng is None:
        rng = np.random.default_rng()
    if min_len is None:
        min_len = target_len // 2

    n = len(x)
    
    # FIX: Crop direction matches pad direction to preserve the most important boundary
    if n >= target_len:
        if pad_direction == "left":
            return x[n - target_len :].copy() # Keep tail
        else:
            return x[:target_len].copy()      # Keep head

    if n < min_len:
        return None

    deficit = target_len - n
    max_start = max(0, n - deficit)
    seg_start = int(rng.integers(0, max_start + 1))
    segment = x[seg_start : seg_start + deficit]
    if len(segment) < deficit:
        reps = (deficit // len(segment)) + 1
        segment = np.tile(segment, reps)[:deficit]

    # FIX: Left pad for history (preserves the right boundary), Right pad for future
    if pad_direction == "left":
        return np.concatenate([segment, x])
    else:
        return np.concatenate([x, segment])


# ===================================================================
# UNIFORM LENGTH 
# ===================================================================

def uniform_length(
    examples: list[dict[str, Any]],
    target_len: int,
    min_len: int | None = None,
    seed: int = 0,
    keys: Sequence[str] = ("history", "future"),
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Force every series in *examples* to exactly ``target_len`` points."""
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
            
            # FIX: Intelligently pick left or right padding based on the key name
            pad_dir = "right" if "future" in key.lower() else "left"
            
            result = crop_or_pad(arr, target_len, min_len=min_len, rng=rng, pad_direction=pad_dir)
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
# BATCH HELPER — augment prepared train examples
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
    """Augment **prepared** train examples by producing the full grid."""
    rng = np.random.default_rng(seed)

    shift_axis = [0] + (list(phase_shifts) if phase_shifts is not None else [])
    smooth_axis = [False, True] if smooth_window > 1 else [False]
    jitter_axis = [False, True] if jitter_sigma > 0 else [False]

    result: list[dict[str, Any]] = []

    for rec in train_records:
        for s in shift_axis:
            for do_smooth in smooth_axis:
                for do_jitter in jitter_axis:
                    new_rec = dict(rec)

                    has_hist = "history_n" in new_rec
                    has_fut = "future_n" in new_rec
                    
                    if not has_hist and not has_fut:
                        result.append(new_rec)
                        continue

                    # FIX: Combine into a single continuous array for shift & smooth
                    # This prevents the boundary between history and future from being torn.
                    parts = []
                    if has_hist:
                        parts.append(np.asarray(new_rec["history_n"], dtype=np.float64).ravel())
                    if has_fut:
                        parts.append(np.asarray(new_rec["future_n"], dtype=np.float64).ravel())
                    
                    arr = np.concatenate(parts)
                    hist_len = len(parts[0]) if has_hist else 0

                    if s != 0:
                        arr = apply_shift(arr, s, fill=phase_fill)
                    if do_smooth:
                        arr = smooth(arr, window=smooth_window)
                    
                    # Split back apart BEFORE applying independent noise
                    if has_hist:
                        hist_arr = arr[:hist_len]
                    if has_fut:
                        fut_arr = arr[hist_len:]

                    if do_jitter:
                        if has_hist:
                            hist_arr = jitter(hist_arr, sigma=jitter_sigma, rng=rng)
                        if has_fut:
                            fut_arr = jitter(fut_arr, sigma=jitter_sigma, rng=rng)
                    
                    # FIX: Target Leakage. We re-calculate empirical statistics strictly 
                    # from the history portion, and apply them identically to both arrays.
                    if has_hist:
                        mu = float(np.mean(hist_arr))
                        sigma = float(np.std(hist_arr))
                        if sigma < 1e-8:
                            sigma = 1.0
                        
                        new_rec["history_n"] = (hist_arr - mu) / sigma
                        if has_fut:
                            new_rec["future_n"] = (fut_arr - mu) / sigma
                    elif has_fut:
                        # Fallback if only future exists
                        new_rec["future_n"] = fut_arr

                    # Rebuild x_model
                    if feature_fn is not None:
                        new_rec["x_model"] = feature_fn(new_rec)
                    elif has_hist:
                        new_rec["x_model"] = new_rec["history_n"]

                    result.append(new_rec)

    return result