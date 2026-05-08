"""
Signal decomposition utilities for feature extraction.

    emd_decompose     - Empirical Mode Decomposition (via PyEMD)
    kmd_decompose     - Kernel Mode Decomposition (via KMD_lib)
    decompose_series  - unified interface: pick method by name

These produce IMF/mode arrays that can be used as multivariate features
in the kernel pipeline via prepare_examples(feature_fn=...).

Setup
-----
EMD:  pip install EMD-signal
KMD:  git clone https://github.com/kernel-enthusiasts/Kernel-Mode-Decomposition-1D
      and add the repo root to sys.path (or copy KMD_lib.py into utils/).
"""

from __future__ import annotations

import builtins
import contextlib
import io
import os
import re
import sys
from typing import Any, Literal

import numpy as np


# ═══════════════════════════════════════════════════════════════
# EMD — Empirical Mode Decomposition
# ═══════════════════════════════════════════════════════════════

def emd_decompose(
    signal: np.ndarray,
    max_imfs: int | None = None,
) -> np.ndarray:
    """
    Decompose a 1-D signal into Intrinsic Mode Functions via EMD.

    Parameters
    ----------
    signal : 1-D array
    max_imfs : int, optional
        Maximum number of IMFs to extract.  None = extract all.

    Returns
    -------
    imfs : ndarray, shape (n_imfs, signal_len)
        Each row is one IMF, ordered from highest to lowest frequency.
        The last row is the residual.
    """
    try:
        from PyEMD import EMD as _EMD
    except ImportError:
        raise ImportError(
            "EMD-signal package required.  Install with:\n"
            "  pip install EMD-signal"
        )

    signal = np.asarray(signal, dtype=float).ravel()
    emd = _EMD()
    if max_imfs is not None:
        emd.MAX_ITERATION = 1000
    imfs = emd.emd(signal, max_imf=max_imfs if max_imfs else -1)
    return imfs


# ═══════════════════════════════════════════════════════════════
# KMD — Kernel Mode Decomposition
# ═══════════════════════════════════════════════════════════════
###### DISCLAIMER NEEDS  alot more cleaning this is ai slop for now
def kmd_decompose(
    signal: np.ndarray,
    alpha: float = 25.0,
    wave_p: Any = None,
    thr: float = 0.005,
    thr_en: float = 0.1,
    ref_fin: bool = False,
    t_mesh: np.ndarray | None = None,
    quiet: bool = True,
) -> dict[str, np.ndarray]:
    """
    Decompose a 1-D signal into modes via Kernel Mode Decomposition.

    Wraps KMD_lib.semimanual_maxpool_peel2 in a fully non-interactive way:
    stdout spam is suppressed and interactive grouping prompts are
    auto-accepted ("Done") so it never blocks.

    Parameters
    ----------
    signal : 1-D array
    alpha : float
        Gaussian window width (larger = coarser frequency resolution).
    wave_p : list, optional
        Waveform specification for KMD:
            ["cos", 0]        cosine (default)
            ["tri", 0]        triangle wave
            ["ekg", 0]        synthetic EKG-like
            ["unk", n]        unknown waveform, learn with n overtones
            ["custom", a]     custom Fourier coefficients (ndarray (K,2))
    thr : float
        Convergence threshold (epsilon_1 in the KMD paper).
    thr_en : float
        Energy threshold for omega_low (fraction of max energy).
    ref_fin : bool
        Refine to machine precision (slow, only for clean signals).
    t_mesh : 1-D array, optional
        Evenly spaced time mesh.  Auto-generated if None.
    quiet : bool or "full"
        True  (default) — suppress progress % spam, but print energy
              and mode summary lines so you can see what was found.
        "full" — suppress all stdout (for batch / pipeline use).
        False — show everything KMD_lib prints (very verbose).

    Returns
    -------
    dict with keys:
        modes       : ndarray (n_modes, N)  — reconstructed mode signals
        amplitudes  : ndarray (n_modes, N)  — instantaneous amplitudes
        phases      : ndarray (n_modes, N)  — instantaneous phases
        frequencies : ndarray (n_modes, N)  — instantaneous frequencies
        fmodes_raw  : ndarray (n_modes, N, 4) — raw KMD output
        wp          : waveform parameters object
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import KMD_lib
    except ImportError as exc:
        raise ImportError(
            "KMD_lib not found or failed to load.  Clone the repo and add it to sys.path:\n"
            "  git clone https://github.com/kernel-enthusiasts/Kernel-Mode-Decomposition-1D\n"
            "  import sys; sys.path.insert(0, '/path/to/Kernel-Mode-Decomposition-1D')\n"
            f"Original error: {exc}"
        ) from exc

    signal = np.asarray(signal, dtype=float).ravel()
    N = len(signal)

    if wave_p is None:
        wave_p = ["cos", 0]
    if t_mesh is None:
        t_mesh = np.linspace(-1, 1, N)

    # --- run KMD non-interactively ----------------------------
    # KMD_lib.semimanual_maxpool_peel2 uses input() to ask the user
    # to group detected mode fragments.  The prompt cycle is:
    #   "Input mode segments to add to mode 0 ..." → give fragment index
    #   same prompt again for mode 0              → "Next" or "Done"
    # We auto-assign: fragment 0→mode 0, fragment 1→mode 1, etc.
    _original_input = builtins.input
    _step = [0]  # 0 = give fragment index, 1 = finalize mode

    def _auto_input(prompt=""):
        prompt_str = prompt if prompt else ""

        # "keep for next iteration" → nothing to keep
        if "keep for next iteration" in prompt_str.lower():
            _step[0] = 0
            return "Done"

        # "Input mode segments to add to mode M ..."
        m = re.search(r"to mode (\d+)", prompt_str)
        if m:
            mode_num = int(m.group(1))
            if _step[0] == 0:
                _step[0] = 1
                return str(mode_num)   # assign fragment mode_num to this mode
            else:
                _step[0] = 0
                return "Done"          # finalize this mode
        # fallback
        return "Done"

    # --- tqdm-based progress capture ----------------------------
    # KMD_lib prints lines like "5% 64 time.struct_time(...)" and
    # "Energy computation progress:".  We intercept stdout line by
    # line: progress % lines drive a tqdm bar, summary lines are
    # printed through, and everything else is swallowed.

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    class _KMDStdoutFilter(io.TextIOBase):
        """Intercept KMD_lib stdout, show tqdm bar + summaries only."""
        def __init__(self, show_summary=True):
            self._bar = None
            self._show_summary = show_summary
            self._real_stdout = sys.stdout

        def write(self, s):
            for line in s.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue

                # progress line: "5% 64 time.struct_time(...)"
                if "time.struct_time" in line or (
                    stripped and stripped[0].isdigit() and "%" in stripped
                ):
                    pct_match = re.match(r"(\d+)%", stripped)
                    if pct_match:
                        pct = int(pct_match.group(1))
                        if self._bar is None and _tqdm is not None:
                            self._bar = _tqdm(
                                total=100, desc="KMD energy",
                                file=self._real_stdout, leave=False,
                            )
                        if self._bar is not None:
                            self._bar.n = pct
                            self._bar.refresh()
                    continue

                # "Energy computation progress:" header — start new bar
                if "energy computation" in stripped.lower():
                    self._close_bar()
                    continue

                # everything else: print if showing summaries
                if self._show_summary:
                    self._real_stdout.write(line + "\n")
                    self._real_stdout.flush()

            return len(s)

        def flush(self):
            if self._real_stdout:
                self._real_stdout.flush()

        def close_bar(self):
            self._close_bar()

        def _close_bar(self):
            if self._bar is not None:
                self._bar.n = 100
                self._bar.refresh()
                self._bar.close()
                self._bar = None

    builtins.input = _auto_input
    try:
        if quiet == "full":
            with contextlib.redirect_stdout(io.StringIO()):
                fmodes, wp = KMD_lib.semimanual_maxpool_peel2(
                    signal, wave_p, alpha, t_mesh, thr, thr_en, ref_fin,
                )
        elif quiet:
            filt = _KMDStdoutFilter(show_summary=True)
            with contextlib.redirect_stdout(filt):
                fmodes, wp = KMD_lib.semimanual_maxpool_peel2(
                    signal, wave_p, alpha, t_mesh, thr, thr_en, ref_fin,
                )
            filt.close_bar()
        else:
            fmodes, wp = KMD_lib.semimanual_maxpool_peel2(
                signal, wave_p, alpha, t_mesh, thr, thr_en, ref_fin,
            )
    finally:
        builtins.input = _original_input

    n_modes = fmodes.shape[0]

    # reconstruct each mode as amplitude * waveform(phase)
    modes = np.zeros((n_modes, N))
    for i in range(n_modes):
        modes[i] = fmodes[i, :, 0] * KMD_lib.wave(wp, fmodes[i, :, 1])

    return {
        "modes": modes,                    # (n_modes, N)
        "amplitudes": fmodes[:, :, 0],     # (n_modes, N)
        "phases": fmodes[:, :, 1],         # (n_modes, N)
        "frequencies": fmodes[:, :, 2],    # (n_modes, N)
        "fmodes_raw": fmodes,              # (n_modes, N, 4)
        "wp": wp,
    }


# ═══════════════════════════════════════════════════════════════
# CWT — Continuous Wavelet Transform
# ═══════════════════════════════════════════════════════════════

def _morlet_wavelet(t: np.ndarray, w0: float = 5.0) -> np.ndarray:
    """Morlet wavelet:  exp(i*w0*t) * exp(-t^2/2)"""
    return np.exp(1j * w0 * t) * np.exp(-t ** 2 / 2.0) * np.pi ** (-0.25)


def _ricker_wavelet(t: np.ndarray) -> np.ndarray:
    """Ricker / Mexican hat wavelet:  (1 - t^2) * exp(-t^2/2)"""
    return (1.0 - t ** 2) * np.exp(-t ** 2 / 2.0) * 2.0 / (np.sqrt(3.0) * np.pi ** 0.25)


def cwt_decompose(
    signal: np.ndarray,
    wavelet: str = "morlet",
    scales: np.ndarray | None = None,
    n_scales: int = 32,
    scale_min: float = 1.0,
    scale_max: float | None = None,
    scale_spacing: str = "log",
    top_k: int | None = None,
    quiet: bool = True,
) -> dict[str, np.ndarray]:
    """
    Decompose a 1-D signal via Continuous Wavelet Transform.

    Computes CWT coefficients at each scale, then reconstructs
    approximate mode signals so the output is comparable to
    EMD/KMD modes.

    Parameters
    ----------
    signal : 1-D array
    wavelet : str
        Wavelet to use:
            "morlet"   — Morlet (default, good for oscillatory signals)
            "ricker"   — Ricker / Mexican hat
        Or pass any name accepted by pywt.ContinuousWavelet if
        PyWavelets is installed (falls back to scipy otherwise).
    scales : 1-D array, optional
        Explicit scale array.  Overrides n_scales / scale_min / scale_max.
    n_scales : int
        Number of scales to compute (default 32).
    scale_min : float
        Smallest scale (highest frequency).  Default 1.0.
    scale_max : float, optional
        Largest scale (lowest frequency).  Default = signal_len / 4.
    scale_spacing : "log" or "linear"
        How to space scales between scale_min and scale_max.
    top_k : int, optional
        Keep only the top_k scales by energy.  None = keep all.
    quiet : bool
        If True (default), show a tqdm progress bar.
        If False, silent.

    Returns
    -------
    dict with keys:
        modes       : ndarray (n_scales, N)  — reconstructed mode at each scale
        coeffs      : ndarray (n_scales, N)  — raw CWT coefficients
        scales      : ndarray (n_scales,)    — the scales used
        energies    : ndarray (n_scales,)    — energy at each scale
        frequencies : ndarray (n_scales,)    — pseudo-frequencies (1/scale)
    """
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    signal = np.asarray(signal, dtype=float).ravel()
    N = len(signal)

    # --- build scales -----------------------------------------
    if scales is None:
        if scale_max is None:
            scale_max = N / 4.0
        if scale_spacing == "log":
            scales = np.geomspace(scale_min, scale_max, n_scales)
        else:
            scales = np.linspace(scale_min, scale_max, n_scales)
    scales = np.asarray(scales, dtype=float)
    n_scales = len(scales)

    # --- resolve wavelet function -----------------------------
    _scipy_wavelets = {
        "morlet": _morlet_wavelet,
        "ricker": _ricker_wavelet,
    }

    use_pywt = False
    if wavelet in _scipy_wavelets:
        wavelet_fn = _scipy_wavelets[wavelet]
    else:
        try:
            import pywt
            use_pywt = True
        except ImportError:
            raise ValueError(
                f"Wavelet {wavelet!r} not in built-ins ({list(_scipy_wavelets)}). "
                f"Install PyWavelets for more options:  pip install PyWavelets"
            )

    # --- compute CWT ------------------------------------------
    coeffs = np.zeros((n_scales, N), dtype=complex if not use_pywt else float)

    if use_pywt:
        import pywt
        coeffs, _ = pywt.cwt(signal, scales, wavelet)
    else:
        iterator = range(n_scales)
        if quiet and _tqdm is not None:
            iterator = _tqdm(iterator, desc="CWT", total=n_scales, leave=False)

        for i in iterator:
            s = scales[i]
            width = int(min(10 * s, N))
            t_wav = np.arange(-width, width + 1) / s
            wav = wavelet_fn(t_wav)
            wav = wav / np.sqrt(s)
            coeffs[i] = np.convolve(signal, wav, mode="same")

    # --- energies and pseudo-frequencies ----------------------
    coeffs_real = np.real(coeffs)
    energies = np.sum(coeffs_real ** 2, axis=1)
    pseudo_freqs = 1.0 / scales

    # --- select top_k by energy -------------------------------
    if top_k is not None and top_k < n_scales:
        idx = np.argsort(energies)[::-1][:top_k]
        idx = np.sort(idx)  # keep scale order
        coeffs_real = coeffs_real[idx]
        coeffs = coeffs[idx]
        scales = scales[idx]
        energies = energies[idx]
        pseudo_freqs = pseudo_freqs[idx]
        n_scales = top_k

    # --- reconstruct modes ------------------------------------
    # Each mode ~ coefficients at that scale.
    # Normalize so sum of modes approximates the original signal.
    modes = coeffs_real.copy()
    mode_sum = np.sum(modes, axis=0)
    denom = np.where(np.abs(mode_sum) > 1e-12, mode_sum, 1.0)
    rescale = signal / denom
    modes = modes * rescale[None, :]

    return {
        "modes": modes,               # (n_scales, N)
        "coeffs": coeffs_real,         # (n_scales, N)
        "scales": scales,              # (n_scales,)
        "energies": energies,          # (n_scales,)
        "frequencies": pseudo_freqs,   # (n_scales,)
    }


# ═══════════════════════════════════════════════════════════════
# UNIFIED INTERFACE
# ═══════════════════════════════════════════════════════════════

def decompose_series(
    signal: np.ndarray,
    method: Literal["emd", "kmd", "cwt"] = "emd",
    **kwargs,
) -> np.ndarray:
    """
    Decompose a 1-D signal and return mode matrix.

    This is the simplest entry point — returns just the modes as a
    (n_modes, signal_len) array, regardless of method.

    Parameters
    ----------
    signal : 1-D array
    method : "emd", "kmd", or "cwt"
    **kwargs : passed to emd_decompose, kmd_decompose, or cwt_decompose

    Returns
    -------
    modes : ndarray (n_modes, signal_len)
    """
    if method == "emd":
        return emd_decompose(signal, **kwargs)
    elif method == "kmd":
        result = kmd_decompose(signal, **kwargs)
        return result["modes"]
    elif method == "cwt":
        result = cwt_decompose(signal, **kwargs)
        return result["modes"]
    else:
        raise ValueError(f"Unknown method {method!r} — use 'emd', 'kmd', or 'cwt'")


# ═══════════════════════════════════════════════════════════════
# FEATURE BUILDERS (for use with prepare_examples)
# ═══════════════════════════════════════════════════════════════

def make_decomposition_feature_fn(
    method: Literal["emd", "kmd", "cwt"] = "emd",
    max_modes: int | None = None,
    include_original: bool = True,
    target_len: int | None = None,
    **decompose_kwargs,
):
    """
    Build a feature_fn for prepare_examples that decomposes the history
    into modes and stacks them as multivariate features.

    Parameters
    ----------
    method : "emd" or "kmd"
    max_modes : int, optional
        Keep at most this many modes (highest-energy first for EMD,
        order-of-extraction for KMD).  None = keep all.
    include_original : bool
        If True, the original history_n is included as the first channel.
    target_len : int, optional
        Resample modes to this length.  If None, uses the history_n length.
    **decompose_kwargs
        Passed to emd_decompose or kmd_decompose.

    Returns
    -------
    feature_fn : callable
        f(rec) -> ndarray (seq_len, n_channels)
        Suitable for  prepare_examples(..., feature_fn=feature_fn)

    Example
    -------
    >>> from utils.decomposition import make_decomposition_feature_fn
    >>> feat_fn = make_decomposition_feature_fn("emd", max_modes=4)
    >>> examples = prepare_examples(raw, feature_fn=feat_fn)
    >>> # each example["x_model"] is now (history_len, 5):
    >>> #   column 0 = original history_n
    >>> #   columns 1-4 = first 4 IMFs
    """
    from utils.data import resample_series

    def feature_fn(rec):
        h = rec["history_n"]
        modes = decompose_series(h, method=method, **decompose_kwargs)

        if max_modes is not None and modes.shape[0] > max_modes:
            modes = modes[:max_modes]

        tgt = target_len if target_len is not None else len(h)

        # resample each mode to target length
        resampled = np.array([
            resample_series(m, tgt) for m in modes
        ])  # (n_modes, tgt)

        channels = []
        if include_original:
            channels.append(resample_series(h, tgt))
        for m in resampled:
            channels.append(m)

        return np.column_stack(channels)  # (tgt, n_channels)

    return feature_fn
