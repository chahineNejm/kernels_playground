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
    quiet : bool
        Suppress KMD_lib's internal print spam (default True).

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
    # KMD_lib uses input() to ask for mode groupings and print()
    # for verbose progress.  We patch both so it never blocks.
    _original_input = builtins.input

    def _auto_input(prompt=""):
        """Auto-respond 'Done' to grouping prompts."""
        return "Done"

    builtins.input = _auto_input
    try:
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                fmodes, wp = KMD_lib.semimanual_maxpool_peel2(
                    signal, wave_p, alpha, t_mesh, thr, thr_en, ref_fin,
                )
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
# UNIFIED INTERFACE
# ═══════════════════════════════════════════════════════════════

def decompose_series(
    signal: np.ndarray,
    method: Literal["emd", "kmd"] = "emd",
    **kwargs,
) -> np.ndarray:
    """
    Decompose a 1-D signal and return mode matrix.

    This is the simplest entry point — returns just the modes as a
    (n_modes, signal_len) array, regardless of method.

    Parameters
    ----------
    signal : 1-D array
    method : "emd" or "kmd"
    **kwargs : passed to emd_decompose or kmd_decompose

    Returns
    -------
    modes : ndarray (n_modes, signal_len)
    """
    if method == "emd":
        return emd_decompose(signal, **kwargs)
    elif method == "kmd":
        result = kmd_decompose(signal, **kwargs)
        return result["modes"]
    else:
        raise ValueError(f"Unknown method {method!r} — use 'emd' or 'kmd'")


# ═══════════════════════════════════════════════════════════════
# FEATURE BUILDERS (for use with prepare_examples)
# ═══════════════════════════════════════════════════════════════

def make_decomposition_feature_fn(
    method: Literal["emd", "kmd"] = "emd",
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
