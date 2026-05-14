"""
Kernel operator regression utilities — pluggable kernel architecture.

Kernel classes (subclass Kernel to add your own):
    RBFKernel            — squared-exponential / Gaussian
    MaskedRBFKernel      — jointly-masked RBF (missing dimensions)
    LinearKernel         — dot-product
    WaveletKernel        — CWT-based kernel (inner, energy, or coeffs mode)
    NTKKernel            — Neural Tangent Kernel (analytical, infinite-width MLP)
    RFMKernel            — Recursive Feature Machine (learned M reweights inputs)
    ConvRFMKernel        — Convolutional Recursive Feature Machine (patches)
    DTWKernel            — dynamic time warping distance

Fitting / prediction:
    fit_kernel_operator   — train with any Kernel instance
    predict_kernel_operator — infer

Metrics:
    rmse, relative_rmse, mase
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import numpy as np


# ═══════════════════════════════════════════════════════════════
# BASE KERNEL
# ═══════════════════════════════════════════════════════════════

class Kernel(ABC):
    """
    Base class for all kernels.

    Subclass contract
    -----------------
    gram(A, B) -> (n, m)   kernel / Gram matrix
    estimate_params(X, rng) -> None   auto-tune hyperparameters (optional)

    Input conventions
    -----------------
    "Flat" kernels (RBF, Linear):
        A is (n_samples, n_features), B is (m_samples, n_features).

    "Structured" kernels (DTW):
        A and B are *lists* of arrays — each element is one sample,
        shape (seq_len_i,) or (seq_len_i, n_channels).
        Sequences may have different lengths.
    """

    @abstractmethod
    def gram(self, A, B) -> np.ndarray:
        ...

    def estimate_params(self, X, rng: np.random.Generator):
        """Auto-tune kernel parameters from training data. Override as needed."""

    def __call__(self, A, B) -> np.ndarray:
        return self.gram(A, B)

    def __repr__(self):
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items()
            if not k.startswith("_")
        )
        return f"{type(self).__name__}({params})"


# ═══════════════════════════════════════════════════════════════
# DISTANCE HELPERS
# ═══════════════════════════════════════════════════════════════

def squared_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between rows of *a* and *b*."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a2 = np.sum(a ** 2, axis=1, keepdims=True)
    b2 = np.sum(b ** 2, axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)


def median_heuristic_lengthscale(
    x: np.ndarray,
    rng: np.random.Generator,
    max_points: int = 24,
) -> float:
    """Estimate a good RBF lengthscale from the median pairwise distance."""
    x = np.asarray(x, dtype=float)
    if x.shape[0] <= 1:
        return 1.0
    n = min(max_points, x.shape[0])
    idx = rng.choice(x.shape[0], size=n, replace=False)
    sub = x[idx]
    dists = np.sqrt(squared_distance_matrix(sub, sub)[np.triu_indices(n, k=1)])
    dists = dists[np.isfinite(dists) & (dists > 0.0)]
    return float(np.median(dists)) if dists.size else 1.0


# ═══════════════════════════════════════════════════════════════
# RBF KERNEL
# ═══════════════════════════════════════════════════════════════

class RBFKernel(Kernel):
    """
    Squared-exponential / Gaussian / RBF kernel.

        K(x, y) = exp( -||x - y||^2 / (2 * lengthscale^2) )

    Input: 2-D arrays (n_samples, n_features).
    If lengthscale is None, it is auto-estimated via median heuristic.
    """

    def __init__(self, lengthscale: float | None = None):
        self.lengthscale = lengthscale

    def estimate_params(self, X, rng):
        if self.lengthscale is None:
            self.lengthscale = median_heuristic_lengthscale(X, rng)

    def gram(self, A, B) -> np.ndarray:
        if self.lengthscale is None:
            raise ValueError("lengthscale not set — call estimate_params or pass it to __init__")
        return np.exp(-squared_distance_matrix(A, B) / (2.0 * self.lengthscale ** 2))


# ═══════════════════════════════════════════════════════════════
# MASKED RBF KERNEL  (jointly-masked RBF for missing dimensions)
# ═══════════════════════════════════════════════════════════════

def build_mask_matrix(X: np.ndarray) -> np.ndarray:
    """
    Build observation masks from data.

    Each sample's mask is a 1-D binary vector: 1 where the value is
    non-zero (observed), 0 where it is zero (missing).

    Parameters
    ----------
    X : (n_samples, d) array

    Returns
    -------
    P : (d, n_samples) array
        Column j is the mask vector for sample j.
        Compact storage: one column per sample, easy to slice.

    Usage
    -----
    >>> P_train = build_mask_matrix(X_train)    # (d, n_train)
    >>> P_test  = build_mask_matrix(X_test)     # (d, n_test)
    >>> kernel  = MaskedRBFKernel()
    >>> kernel.estimate_params(X_train, rng, P_train)
    >>> G_train = kernel.gram(X_train, X_train, P_train, P_train)
    >>> G_test  = kernel.gram(X_test, X_train, P_test, P_train)
    """
    X = np.asarray(X, dtype=float)
    return (X != 0).astype(np.float64).T   # (d, n)


class MaskedRBFKernel(Kernel):
    """
    Jointly-Masked RBF kernel — only compares dimensions observed in both.

    Masks are precomputed via ``build_mask_matrix`` and passed to
    ``gram`` / ``estimate_params`` explicitly, so they are computed
    once and reused.

        K(x_i, x_j) = exp( -||P_i P_j (x_i - x_j)||^2 / (2 σ^2 · ||P_i P_j||_F) )

    where P_i, P_j are diagonal observation masks (stored as columns
    of the mask matrix P).

    Properties:
        - P_i = P_j = I (fully observed): reduces to standard RBF
        - No shared observed dims → kernel = 0
        - Frobenius norm normalises by effective dimensionality

    Convention: zero = missing.  Shift data if genuine zeros exist.

    Parameters
    ----------
    lengthscale : float, optional
        RBF bandwidth σ.  Auto-estimated via masked median heuristic if None.
    """

    def __init__(self, lengthscale: float | None = None):
        self.lengthscale = lengthscale

    # -- vectorised masked squared-distance matrix ---------------------

    @staticmethod
    def _masked_sq_dist(
        A: np.ndarray,
        B: np.ndarray,
        P_a: np.ndarray,
        P_b: np.ndarray,
    ):
        """
        Compute masked squared distances and Frobenius norms.

        Parameters
        ----------
        A   : (n, d)  data matrix
        B   : (m, d)  data matrix
        P_a : (d, n)  mask matrix for A  (columns are mask vectors)
        P_b : (d, m)  mask matrix for B

        Returns
        -------
        D2  : (n, m)  masked squared Euclidean distances
        F   : (n, m)  ||P_i P_j||_F  for each pair
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        # masks as row-major for matmul:  (n, d) and (m, d)
        ma = P_a.T   # (n, d)
        mb = P_b.T   # (m, d)

        # ||P_i P_j||_F = sqrt( Σ_d  p_i[d] · p_j[d] )
        shared_count = ma @ mb.T                      # (n, m)
        F = np.sqrt(np.maximum(shared_count, 0.0))    # (n, m)

        # ||P_i P_j (x_i - x_j)||^2
        #   = Σ_d  p_i[d]·p_j[d]·(a[d]-b[d])^2
        #   = (a^2 · p_b) + (p_a · b^2) - 2·(a · b^T)
        #
        # Identity holds because p[d]·x[d]^2 = x[d]^2 when p = (x!=0).
        D2 = (A ** 2) @ mb.T + ma @ (B ** 2).T - 2.0 * A @ B.T
        D2 = np.maximum(D2, 0.0)

        return D2, F

    # -- median heuristic on masked distances --------------------------

    def estimate_params(self, X, rng, P=None):
        """
        Auto-estimate lengthscale from masked pairwise distances.

        Parameters
        ----------
        X : (n, d) data
        rng : Generator
        P : (d, n) mask matrix, optional.  Built from X if not given.
        """
        if self.lengthscale is not None:
            return
        X = np.asarray(X, dtype=float)
        if P is None:
            P = build_mask_matrix(X)

        n = min(24, X.shape[0])
        idx = rng.choice(X.shape[0], size=n, replace=False)
        sub = X[idx]
        sub_P = P[:, idx]    # (d, n_sub)

        D2, F = self._masked_sq_dist(sub, sub, sub_P, sub_P)
        safe_F = np.where(F > 0, F, 1.0)
        eff_dist = np.sqrt(D2 / safe_F)

        upper = eff_dist[np.triu_indices(n, k=1)]
        upper = upper[(upper > 0) & np.isfinite(upper)]
        self.lengthscale = float(np.median(upper)) if upper.size else 1.0

    # -- Gram matrix ---------------------------------------------------

    def gram(self, A, B, P_a=None, P_b=None) -> np.ndarray:
        """
        Compute the masked RBF Gram matrix.

        Parameters
        ----------
        A   : (n, d)  data
        B   : (m, d)  data
        P_a : (d, n)  mask matrix for A, optional (built from A if None)
        P_b : (d, m)  mask matrix for B, optional (built from B if None)

        Returns
        -------
        G : (n, m) kernel matrix
        """
        if self.lengthscale is None:
            raise ValueError(
                "lengthscale not set — call estimate_params or pass it to __init__"
            )
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if P_a is None:
            P_a = build_mask_matrix(A)
        if P_b is None:
            P_b = build_mask_matrix(B)

        D2, F = self._masked_sq_dist(A, B, P_a, P_b)

        denom = 2.0 * self.lengthscale ** 2 * F
        safe_denom = np.where(denom > 0, denom, 1.0)
        G = np.exp(-D2 / safe_denom)
        G = np.where(F > 0, G, 0.0)

        return G


# ═══════════════════════════════════════════════════════════════
# LINEAR KERNEL
# ═══════════════════════════════════════════════════════════════

class LinearKernel(Kernel):
    """
    Dot-product kernel:  K(x, y) = x · y + bias

    Input: 2-D arrays (n_samples, n_features).
    No hyperparameters to tune (bias is fixed).
    """

    def __init__(self, bias: float = 0.0):
        self.bias = bias

    def gram(self, A, B) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        return A @ B.T + self.bias


# ═══════════════════════════════════════════════════════════════
# WAVELET KERNEL
# ═══════════════════════════════════════════════════════════════

class WaveletKernel(Kernel):
    """
    Kernel that compares signals via their CWT representations.

        K(x, y) = exp( -d(Wx, Wy)^2 / (2 * sigma^2) )

    where Wx, Wy are the CWT coefficient matrices and d is a
    distance in the wavelet domain.  Three distance modes:

        "inner"   — K = <Wx, Wy>_F  (Frobenius inner product,
                    optionally weighted by scale).  Positive semi-
                    definite by construction.  No sigma needed.

        "energy"  — Euclidean distance between energy-per-scale
                    vectors, fed through an RBF.  Very compact.

        "coeffs"  — Frobenius distance between full coefficient
                    matrices, fed through an RBF.  Richest, but
                    highest dimensional.

    Input: 2-D array (n_samples, signal_len) for flat mode, or
           list of 1-D arrays for variable-length signals.

    Parameters
    ----------
    mode : "inner", "energy", or "coeffs"
    sigma : float, optional
        RBF bandwidth (auto-estimated if None).  Ignored for "inner".
    n_scales : int
        Number of CWT scales.
    scale_min, scale_max : float
        Scale range.
    wavelet : str
        Wavelet name (passed to cwt_decompose).
    scale_weights : 1-D array, optional
        Per-scale weights for "inner" mode.  None = uniform.
    top_k : int, optional
        Keep only top_k scales by energy before computing distance.
    """

    def __init__(
        self,
        mode: str = "energy",
        sigma: float | None = None,
        n_scales: int = 32,
        scale_min: float = 1.0,
        scale_max: float | None = None,
        wavelet: str = "morlet",
        scale_weights: np.ndarray | None = None,
        top_k: int | None = None,
    ):
        self.mode = mode
        self.sigma = sigma
        self._sigma_from_user = sigma is not None
        self.n_scales = n_scales
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.wavelet = wavelet
        self.scale_weights = scale_weights
        self.top_k = top_k

    def _get_signals(self, X):
        """Extract individual signals from array or list."""
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return [X[i] for i in range(X.shape[0])]
        return list(X)

    def gram(self, A, B) -> np.ndarray:
        from utils.decomposition import cwt_decompose

        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        sigs_a = self._get_signals(A)
        sigs_b = self._get_signals(B)
        n, m = len(sigs_a), len(sigs_b)
        symmetric = A is B

        # --- compute all CWTs with progress -----------------------
        desc_a = f"WaveletKernel CWT A ({n})"
        desc_b = f"WaveletKernel CWT B ({m})"

        def _cwt_batch(signals, desc):
            results = []
            iterator = range(len(signals))
            if _tqdm is not None:
                iterator = _tqdm(iterator, desc=desc, leave=False)
            for i in iterator:
                results.append(cwt_decompose(
                    signals[i],
                    wavelet=self.wavelet,
                    n_scales=self.n_scales,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                    top_k=self.top_k,
                    quiet="full",
                ))
            return results

        cwts_a = _cwt_batch(sigs_a, desc_a)
        if symmetric:
            cwts_b = cwts_a
        else:
            cwts_b = _cwt_batch(sigs_b, desc_b)

        # --- compute gram matrix ----------------------------------
        if self.mode == "inner":
            return self._gram_inner(cwts_a, cwts_b, n, m, symmetric)
        elif self.mode == "energy":
            return self._gram_distance(
                [c["energies"] for c in cwts_a],
                [c["energies"] for c in cwts_b],
                n, m, symmetric,
            )
        elif self.mode == "coeffs":
            return self._gram_distance(
                [c["coeffs"].ravel() for c in cwts_a],
                [c["coeffs"].ravel() for c in cwts_b],
                n, m, symmetric,
            )
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")

    def _gram_inner(self, cwts_a, cwts_b, n, m, symmetric):
        """Inner-product kernel in wavelet domain."""
        G = np.zeros((n, m))
        w = self.scale_weights

        if symmetric:
            total = n * (n - 1) // 2 + n
            desc = f"Wavelet inner (sym {n}x{n})"
        else:
            total = n * m
            desc = f"Wavelet inner ({n}x{m})"

        with tqdm(total=total, desc=desc, leave=False) as pbar:
            for i in range(n):
                j_start = i if symmetric else 0
                ca = cwts_a[i]["coeffs"]       # (n_scales, T)
                for j in range(j_start, m):
                    cb = cwts_b[j]["coeffs"]   # (n_scales, T)
                    T = min(ca.shape[1], cb.shape[1])
                    if w is not None:
                        val = float(np.sum(w[:, None] * ca[:, :T] * cb[:, :T]))
                    else:
                        val = float(np.sum(ca[:, :T] * cb[:, :T]))
                    G[i, j] = val
                    if symmetric and i != j:
                        G[j, i] = val
                    pbar.update(1)

        # normalize to correlation: K_ij / sqrt(K_ii * K_jj)
        diag = np.sqrt(np.diag(G))
        diag = np.where(diag > 1e-12, diag, 1.0)
        G = G / (diag[:, None] * diag[None, :])
        return G

    def _gram_distance(self, vecs_a, vecs_b, n, m, symmetric):
        """RBF kernel on feature vectors (energy or flattened coeffs)."""
        va = np.array(vecs_a)  # (n, d)
        vb = np.array(vecs_b)  # (m, d)
        D2 = squared_distance_matrix(va, vb)

        # auto-estimate sigma from training distances
        if not self._sigma_from_user and self.sigma is None and symmetric:
            dists = np.sqrt(D2[np.triu_indices(n, k=1)])
            dists = dists[dists > 0]
            self.sigma = float(np.median(dists)) if dists.size else 1.0

        if self.sigma is None:
            raise ValueError("sigma not set — run on training data first or pass sigma")

        return np.exp(-D2 / (2.0 * self.sigma ** 2))


# ═══════════════════════════════════════════════════════════════
# DTW HELPERS
# ═══════════════════════════════════════════════════════════════

def _dtw_cost(x: np.ndarray, y: np.ndarray, window: int | None = None) -> float:
    """
    DTW distance between two sequences.

    x, y : 1-D (T,) or 2-D (T, C) arrays.
    window : Sakoe-Chiba band half-width (None = unconstrained).
    """
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    n, m = x.shape[0], y.shape[0]
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - window) if window is not None else 1
        j_hi = min(m, i + window) if window is not None else m
        for j in range(j_lo, j_hi + 1):
            d = float(np.sum((x[i - 1] - y[j - 1]) ** 2))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(np.sqrt(cost[n, m]))


def _dtw_distance_matrix(
    A: list[np.ndarray],
    B: list[np.ndarray],
    window: int | None = None,
    desc: str = "DTW",
) -> np.ndarray:
    """
    Pairwise DTW distance matrix between two lists of sequences.

    Exploits symmetry when A is B (same object) — only computes the
    upper triangle, cutting wall-clock time roughly in half for gram
    matrices.
    """
    n, m = len(A), len(B)
    D = np.zeros((n, m))
    symmetric = A is B

    if symmetric:
        total = n * (n - 1) // 2
        with tqdm(total=total, desc=f"{desc} (sym {n}x{n})") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    d = _dtw_cost(A[i], A[j], window=window)
                    D[i, j] = d
                    D[j, i] = d
                    pbar.update(1)
    else:
        total = n * m
        with tqdm(total=total, desc=f"{desc} ({n}x{m})") as pbar:
            for i in range(n):
                for j in range(m):
                    D[i, j] = _dtw_cost(A[i], B[j], window=window)
                    pbar.update(1)
    return D


# ═══════════════════════════════════════════════════════════════
# NEURAL TANGENT KERNEL (analytical, infinite-width MLP)
# ═══════════════════════════════════════════════════════════════

class NTKKernel(Kernel):
    """
    Neural Tangent Kernel for a fully-connected ReLU network.

    Computes the exact infinite-width NTK via the closed-form
    recursive formula (Jacot et al., NeurIPS 2018).  No neural
    network is instantiated — this is a pure kernel method.

    The kernel corresponds to training an infinitely wide MLP:

        f(x) = W_L · σ(W_{L-1} · σ(… σ(W_1 · x + b_1) …) + b_{L-1}) + b_L

    with L = depth hidden layers, each using ReLU activation σ.

    The NTK is built recursively, layer by layer:

        1. Σ⁰ = σ_w² (X·X') / d + σ_b²           (input covariance)
        2. For each hidden layer l:
             ρ   = Σ^{l-1}_ab / √(Σ^{l-1}_aa · Σ^{l-1}_bb)
             κ₁  = √(Σ_aa·Σ_bb)/(2π) · (√(1-ρ²) + (π-arccos ρ)·ρ)
             κ₀  = (π - arccos ρ) / (2π)
             Σ^l = σ_w² · κ₁ + σ_b²
             Σ̇^l = σ_w² · κ₀
             Θ^l = Σ^l + Σ̇^l · Θ^{l-1}

    Parameters
    ----------
    depth : int
        Number of hidden layers (each with ReLU).
    sigma_w : float
        Weight standard deviation per layer.
        Default √2 (He initialisation, keeps variance stable through
        ReLU layers).
    sigma_b : float
        Bias standard deviation per layer.  Default 0.
    normalize : bool
        If True, normalise the final NTK to a correlation matrix:
        K_ij → K_ij / √(K_ii · K_jj).
        Keeps kernel values in [−1, 1] and improves conditioning of
        the Gram matrix.  Recommended unless you want raw NTK values.

    Example
    -------
    >>> kernel = NTKKernel(depth=3)
    >>> result = train_eval(examples, kernel=kernel)
    """

    def __init__(
        self,
        depth: int = 3,
        sigma_w: float = None,
        sigma_b: float = 0.0,
        normalize: bool = True,
    ):
        self.depth = depth
        self.sigma_w = np.sqrt(2.0) if sigma_w is None else sigma_w
        self.sigma_b = sigma_b
        self.normalize = normalize

    # -- ReLU dual activation functions ----------------------------

    @staticmethod
    def _kappa1(rho: np.ndarray, s_aa: np.ndarray, s_bb: np.ndarray) -> np.ndarray:
        """E[ReLU(u) · ReLU(v)] where (u,v) ~ N(0, Λ) and ρ = Λ_ab / √(Λ_aa · Λ_bb)."""
        rho = np.clip(rho, -1.0, 1.0)
        angle = np.arccos(rho)
        sin_angle = np.sqrt(np.maximum(1.0 - rho ** 2, 0.0))
        magnitude = np.sqrt(np.maximum(s_aa * s_bb, 0.0))
        return magnitude / (2.0 * np.pi) * (sin_angle + (np.pi - angle) * rho)

    @staticmethod
    def _kappa0(rho: np.ndarray) -> np.ndarray:
        """E[ReLU'(u) · ReLU'(v)] = (π − arccos ρ) / 2π."""
        rho = np.clip(rho, -1.0, 1.0)
        return (np.pi - np.arccos(rho)) / (2.0 * np.pi)

    # -- diagonal helper (needed for cross-gram normalisation) -----

    def _ntk_diag(self, X: np.ndarray) -> np.ndarray:
        """Compute NTK diagonal entries Θ(x_i, x_i) for all rows of X."""
        sw2 = self.sigma_w ** 2
        sb2 = self.sigma_b ** 2
        d = X.shape[1]

        s_ii = sw2 * np.sum(X ** 2, axis=1) / d + sb2   # (n,)
        theta_ii = s_ii.copy()

        for _ in range(self.depth):
            # at ρ = 1:  κ₀ = 0.5,  κ₁(s,s) = s/2
            s_dot = sw2 * 0.5
            s_ii = sw2 * s_ii / 2.0 + sb2
            theta_ii = s_ii + s_dot * theta_ii

        return theta_ii

    # -- Gram matrix -----------------------------------------------

    def gram(self, A, B) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        sw2 = self.sigma_w ** 2
        sb2 = self.sigma_b ** 2
        d = A.shape[1]

        # Layer 0: input covariance
        S_ab = sw2 * (A @ B.T) / d + sb2
        S_aa = sw2 * np.sum(A ** 2, axis=1, keepdims=True) / d + sb2   # (n, 1)
        S_bb = sw2 * np.sum(B ** 2, axis=1, keepdims=True).T / d + sb2 # (1, m)

        # Θ starts as the layer-0 covariance
        Theta = S_ab.copy()

        # Hidden layers with ReLU
        for _ in range(self.depth):
            denom = np.sqrt(np.maximum(S_aa * S_bb, 1e-24))
            rho = S_ab / denom

            S_dot = sw2 * self._kappa0(rho)

            S_ab = sw2 * self._kappa1(rho, S_aa, S_bb) + sb2
            S_aa = sw2 * S_aa / 2.0 + sb2    # κ₁ at ρ=1 → S_aa/2
            S_bb = sw2 * S_bb / 2.0 + sb2

            Theta = S_ab + S_dot * Theta

        # -- optional normalisation --------------------------------
        if self.normalize:
            if A is B:
                diag = np.diag(Theta)
                diag = np.maximum(diag, 1e-12)
                Theta = Theta / np.sqrt(np.outer(diag, diag))
            else:
                diag_a = self._ntk_diag(A)
                diag_b = self._ntk_diag(B)
                Theta = Theta / np.sqrt(
                    np.maximum(diag_a[:, None], 1e-12)
                    * np.maximum(diag_b[None, :], 1e-12)
                )

        return Theta


# ═══════════════════════════════════════════════════════════════
# RECURSIVE FEATURE MACHINE (RFM)
# ═══════════════════════════════════════════════════════════════

class RFMKernel(Kernel):
    """
    Recursive Feature Machine kernel (Radhakrishnan et al., NeurIPS 2022).

    Learns a feature matrix M (d × d) that reweights input dimensions:

        K_M(x, x') = exp( -||Mx - Mx'||² / (2σ²) )

    M is learned iteratively (T rounds):
        1. Solve kernel regression:  α = (K_M(X,X) + γI)^{-1} y
        2. Update M from average gradient outer product:
           M ← (1/n) Σ_i  (∇_x f(x_i)) (∇_x f(x_i))^T

    Parameters
    ----------
    T : int
        Number of RFM iterations.  Default 3.
    lengthscale : float, optional
        RBF lengthscale.  Auto-estimated if None.
    normalize_M : bool
        Normalize M by its Frobenius norm after each update.

    Example
    -------
    >>> kernel = RFMKernel(T=3)
    >>> result = train_eval(examples, kernel=kernel, ...)
    """

    def __init__(
        self,
        T: int = 3,
        lengthscale: float | None = None,
        normalize_M: bool = True,
    ):
        self.T = T
        self.lengthscale = lengthscale
        self.normalize_M = normalize_M
        self.M: np.ndarray | None = None
        self._fitted = False

    def estimate_params(self, X, rng):
        if self.lengthscale is not None:
            return
        X = np.asarray(X, dtype=float)
        if self.M is not None:
            Xp = X @ self.M.T
        else:
            Xp = X
        self.lengthscale = median_heuristic_lengthscale(Xp, rng)

    def gram(self, A, B) -> np.ndarray:
        if self.lengthscale is None:
            raise ValueError("lengthscale not set")
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if self.M is not None:
            A = A @ self.M.T
            B = B @ self.M.T
        return np.exp(
            -squared_distance_matrix(A, B) / (2.0 * self.lengthscale ** 2)
        )

    def fit_rfm(
        self,
        X_train,
        y_train: np.ndarray,
        gamma: float,
        rng: np.random.Generator,
        verbose: bool = True,
    ):
        """
        Run the RFM iterative algorithm.

        Parameters
        ----------
        X_train : (n, d) array
        y_train : (n, output_dim) array
        gamma : float
            Tikhonov regularisation.
        rng : Generator
        verbose : bool
        """
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        n, d = X.shape

        self.M = np.eye(d, dtype=float)

        for t in range(self.T):
            # step 1: estimate lengthscale with current M
            self.lengthscale = None
            self.estimate_params(X, rng)

            # step 2: Gram matrix and solve
            G = self.gram(X, X)
            alpha = np.linalg.solve(G + gamma * np.eye(n), y)

            # step 3: gradient outer products → new M
            sigma2 = 2.0 * self.lengthscale ** 2
            MtM = self.M.T @ self.M

            M_accum = np.zeros((d, d), dtype=float)

            for i in tqdm(range(n), desc=f"RFM iter {t+1}/{self.T}",
                          leave=False, disable=not verbose):
                diffs = X - X[i]                            # (n, d)
                w = np.sum(G[:, i:i+1] * alpha, axis=1)    # (n,)
                weighted_diff = (w[:, None] * diffs).sum(axis=0)  # (d,)
                grad_i = MtM @ weighted_diff / sigma2       # (d,)
                M_accum += np.outer(grad_i, grad_i)

            self.M = M_accum / n

            if self.normalize_M:
                fnorm = np.linalg.norm(self.M, "fro")
                if fnorm > 1e-12:
                    self.M = self.M / fnorm

            if verbose:
                resid = np.linalg.norm(y - G @ alpha) / max(np.linalg.norm(y), 1e-12)
                print(f"  RFM iter {t+1}/{self.T}: "
                      f"residual={resid:.4f}  "
                      f"||M||_F={np.linalg.norm(self.M, 'fro'):.4f}  "
                      f"lengthscale={self.lengthscale:.4f}")

        self._fitted = True


# ═══════════════════════════════════════════════════════════════
# CONVOLUTIONAL RECURSIVE FEATURE MACHINE (ConvRFM)
# ═══════════════════════════════════════════════════════════════

class ConvRFMKernel(Kernel):
    """
    Convolutional Recursive Feature Machine kernel for 1-D time series.

    Learns a feature matrix M (q × q) that weights local patches of
    size q.  The kernel between two signals is:

        K_M(x, x') = Σ_u  exp( -||M·x[u:u+q] - M·x'[u:u+q]||² / (2σ²) )

    summed over all aligned patch positions u.

    M is learned iteratively (T rounds):
        1. Solve kernel regression:  α = (K_M(X,X) + γI)^{-1} y
        2. Update M from the average outer product of patch gradients:
           M ← (1/n) Σ_x Σ_u  (∇_{x[u]} f(x)) (∇_{x[u]} f(x))^T
           where f(x) = α · K_M(X, x)

    Parameters
    ----------
    q : int
        Patch size (number of time steps per window).  Default 32.
    stride : int
        Stride between patches.  Default 1.  Larger values speed up
        computation at the cost of resolution.
    T : int
        Number of RFM iterations.  Default 3.
    lengthscale : float, optional
        RBF lengthscale for patch kernel.  Auto-estimated if None.
    normalize_M : bool
        Normalize M by its Frobenius norm after each update.  Default True.

    Notes
    -----
    - Call fit_rfm(X_train, y_train, gamma) before using gram().
      If used with fit_kernel_operator, the fit is triggered automatically.
    - After fitting, M is fixed and gram() can be called on new data.
    - For long signals, increase stride to reduce the number of patches.

    Example
    -------
    >>> kernel = ConvRFMKernel(q=64, T=5, stride=4)
    >>> model = fit_kernel_operator(x_train, y_train, gamma=1e-2,
    ...                             rng=rng, kernel=kernel)
    """

    def __init__(
        self,
        q: int = 32,
        stride: int = 1,
        T: int = 3,
        lengthscale: float | None = None,
        normalize_M: bool = True,
    ):
        self.q = q
        self.stride = stride
        self.T = T
        self.lengthscale = lengthscale
        self.normalize_M = normalize_M
        self.M: np.ndarray = np.eye(q, dtype=float)
        self._fitted = False

    # -- patch extraction ------------------------------------------

    def _extract_patches(self, x: np.ndarray) -> np.ndarray:
        """Extract (n_patches, q) array of sliding patches from 1-D signal."""
        x = np.asarray(x, dtype=float).ravel()
        n_patches = (len(x) - self.q) // self.stride + 1
        if n_patches <= 0:
            raise ValueError(
                f"Signal length {len(x)} too short for patch size q={self.q}"
            )
        patches = np.array([
            x[i * self.stride : i * self.stride + self.q]
            for i in range(n_patches)
        ])
        return patches  # (n_patches, q)

    def _get_signals(self, X):
        """Convert 2-D array or list to list of 1-D signals."""
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return [X[i] for i in range(X.shape[0])]
        return list(X)

    # -- patch-based Gram matrix -----------------------------------

    def _patch_gram_pair(self, x_i: np.ndarray, x_j: np.ndarray) -> float:
        """
        K_M(x_i, x_j) = Σ_u exp(-||M·p_i[u] - M·p_j[u]||² / (2σ²))
        summed over matching patch positions.
        """
        p_i = self._extract_patches(x_i)  # (S, q)
        p_j = self._extract_patches(x_j)  # (S, q)

        # use the minimum number of patches
        S = min(len(p_i), len(p_j))
        p_i, p_j = p_i[:S], p_j[:S]

        # project through M
        Mp_i = p_i @ self.M.T  # (S, q)
        Mp_j = p_j @ self.M.T  # (S, q)

        # RBF on each patch pair, then sum
        diffs = Mp_i - Mp_j                                # (S, q)
        sq_dists = np.sum(diffs ** 2, axis=1)               # (S,)
        sigma2 = 2.0 * self.lengthscale ** 2
        return float(np.sum(np.exp(-sq_dists / sigma2)))

    def gram(self, A, B) -> np.ndarray:
        if self.lengthscale is None:
            raise ValueError(
                "lengthscale not set — call estimate_params or fit_rfm first"
            )
        sigs_a = self._get_signals(A)
        sigs_b = self._get_signals(B)
        n, m = len(sigs_a), len(sigs_b)
        symmetric = A is B

        G = np.zeros((n, m))
        if symmetric:
            for i in tqdm(range(n), desc="ConvRFM gram (sym)", leave=False):
                G[i, i] = self._patch_gram_pair(sigs_a[i], sigs_a[i])
                for j in range(i + 1, n):
                    val = self._patch_gram_pair(sigs_a[i], sigs_a[j])
                    G[i, j] = val
                    G[j, i] = val
        else:
            for i in tqdm(range(n), desc="ConvRFM gram", leave=False):
                for j in range(m):
                    G[i, j] = self._patch_gram_pair(sigs_a[i], sigs_b[j])
        return G

    # -- lengthscale estimation ------------------------------------

    def estimate_params(self, X, rng):
        """Estimate lengthscale from a few projected patch distances."""
        if self.lengthscale is not None:
            return
        sigs = self._get_signals(X)
        n_probe = min(20, len(sigs))
        idx = rng.choice(len(sigs), size=n_probe, replace=False)

        # collect projected patch norms for distance estimation
        all_dists = []
        for k in range(n_probe):
            for l in range(k + 1, n_probe):
                p_k = self._extract_patches(sigs[idx[k]])
                p_l = self._extract_patches(sigs[idx[l]])
                S = min(len(p_k), len(p_l))
                Mp_k = p_k[:S] @ self.M.T
                Mp_l = p_l[:S] @ self.M.T
                dists = np.sqrt(np.sum((Mp_k - Mp_l) ** 2, axis=1))
                all_dists.extend(dists[np.isfinite(dists) & (dists > 0)])

        self.lengthscale = float(np.median(all_dists)) if all_dists else 1.0

    # -- RFM iterative training ------------------------------------

    def fit_rfm(
        self,
        X_train,
        y_train: np.ndarray,
        gamma: float,
        rng: np.random.Generator,
        verbose: bool = True,
    ):
        """
        Run the ConvRFM iterative algorithm.

        After calling this, M is learned and gram() uses it.
        Also stores alpha and X_train internally so the kernel
        can be used with fit_kernel_operator seamlessly.

        Parameters
        ----------
        X_train : array or list
            Training signals (same format as gram() input).
        y_train : (n, d) array
            Training targets.
        gamma : float
            Tikhonov regularisation.
        rng : Generator
            Random seed for lengthscale estimation.
        verbose : bool
            Print iteration progress.
        """
        sigs = self._get_signals(X_train)
        n = len(sigs)
        y = np.asarray(y_train, dtype=float)
        q = self.q

        # reset M to identity
        self.M = np.eye(q, dtype=float)

        for t in range(self.T):
            # --- step 1: estimate lengthscale with current M ------
            self.lengthscale = None  # force re-estimation
            self.estimate_params(X_train, rng)

            # --- step 2: build Gram matrix and solve --------------
            G = self.gram(X_train, X_train)
            alpha = np.linalg.solve(G + gamma * np.eye(n), y)

            # --- step 3: compute patch gradients and update M -----
            # f(x) = Σ_j α_j K_M(x_j, x)
            # ∇_{x[u]} f(x) = Σ_j α_j ∇_{x[u]} K_M(x_j, x)
            #
            # For patch u:
            #   ∇_{x[u]} K_M(x_j, x) = k(Mp_j[u], Mp[u]) · M^T (Mp_j[u] - Mp[u]) / σ²
            #
            # We accumulate:
            #   M_new = (1/n) Σ_x Σ_u  grad_u · grad_u^T

            M_accum = np.zeros((q, q), dtype=float)
            sigma2 = 2.0 * self.lengthscale ** 2
            total_patches = 0

            # pre-compute patches and projections once per iteration
            all_patches = [self._extract_patches(sigs[i]) for i in range(n)]
            all_Mp = [all_patches[i] @ self.M.T for i in range(n)]
            # pre-sum alpha over output dims: scalar weight per training point
            alpha_weights = np.sum(alpha, axis=1)  # (n,)

            for i in tqdm(range(n), desc=f"RFM iter {t+1}/{self.T} grads",
                          leave=False, disable=not verbose):
                S_i = len(all_patches[i])
                Mp_i = all_Mp[i]                          # (S_i, q)

                for u in range(S_i):
                    grad_u = np.zeros(q, dtype=float)

                    for j in range(n):
                        if u >= len(all_patches[j]):
                            continue
                        diff = all_Mp[j][u] - Mp_i[u]         # (q,)
                        sq_dist = float(np.sum(diff ** 2))
                        k_val = np.exp(-sq_dist / sigma2)
                        grad_u += alpha_weights[j] * k_val * (self.M.T @ diff)

                    M_accum += np.outer(grad_u, grad_u)
                    total_patches += 1

            # average and update M
            if total_patches > 0:
                self.M = M_accum / total_patches

            if self.normalize_M:
                fnorm = np.linalg.norm(self.M, "fro")
                if fnorm > 1e-12:
                    self.M = self.M / fnorm

            if verbose:
                # re-evaluate with new M
                resid = np.linalg.norm(y - G @ alpha) / max(np.linalg.norm(y), 1e-12)
                print(f"  RFM iter {t+1}/{self.T}: "
                      f"residual={resid:.4f}  "
                      f"||M||_F={np.linalg.norm(self.M, 'fro'):.4f}  "
                      f"lengthscale={self.lengthscale:.4f}")

        self._fitted = True


# ═══════════════════════════════════════════════════════════════
# DTW KERNEL
# ═══════════════════════════════════════════════════════════════

class DTWKernel(Kernel):
    """
    Kernel based on Dynamic Time Warping distance.

        K(x, y) = exp( -dtw(x, y)^2 / (2 * sigma^2) )

    Input: *list* of arrays, each (seq_len,) or (seq_len, n_channels).
    Sequences may have different lengths — that is the whole point.

    Parameters
    ----------
    sigma : float, optional
        Bandwidth.  If None, estimated automatically from the median
        of the training DTW distance matrix — no extra computation,
        the matrix we need for the gram is the same one we use for sigma.
    window : int, optional
        Sakoe-Chiba band half-width.  Limits warping and speeds up
        computation from O(T^2) to O(T * window).  None = full DTW.

    Note
    ----
    Complexity is O(n^2 * T^2) for n samples of length T.
    Fine for a few hundred samples; for thousands, set *window* or
    consider an approximate DTW library (dtaidistance, tslearn).
    """

    def __init__(self, sigma: float | None = None, window: int | None = None):
        self.sigma = sigma
        self.window = window
        self._sigma_from_user = sigma is not None

    def estimate_params(self, X, rng):
        # sigma is estimated inside gram() from the actual distance matrix,
        # so nothing to do here — kept for Kernel interface compatibility.
        pass

    def _estimate_sigma(self, D: np.ndarray) -> None:
        """Set sigma from median of positive entries in a distance matrix."""
        n = D.shape[0]
        dists = D[np.triu_indices(n, k=1)]
        dists = dists[dists > 0]
        self.sigma = float(np.median(dists)) if dists.size else 1.0

    def gram(self, A, B) -> np.ndarray:
        D = _dtw_distance_matrix(A, B, window=self.window, desc="DTW gram")

        # auto-estimate sigma from the training gram (A is B)
        if not self._sigma_from_user and self.sigma is None and A is B:
            self._estimate_sigma(D)

        if self.sigma is None:
            raise ValueError(
                "sigma not set — pass sigma to __init__, or call gram "
                "on the training set first (A is B) so it auto-estimates."
            )

        return np.exp(-D ** 2 / (2.0 * self.sigma ** 2))


# ═══════════════════════════════════════════════════════════════
# BACKWARD-COMPAT STANDALONE FUNCTION
# ═══════════════════════════════════════════════════════════════

def rbf_kernel(a: np.ndarray, b: np.ndarray, lengthscale: float) -> np.ndarray:
    """RBF kernel matrix (standalone helper, kept for backward compat)."""
    return np.exp(-squared_distance_matrix(a, b) / (2.0 * lengthscale ** 2))


# ═══════════════════════════════════════════════════════════════
# FIT / PREDICT
# ═══════════════════════════════════════════════════════════════

def fit_kernel_operator(
    x_train,
    y_train: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
    kernel: Kernel | None = None,
    P_train: np.ndarray | None = None,
) -> dict:
    """
    Fit a kernel-operator regressor:  y ~ K(x, X_train) * alpha

    Parameters
    ----------
    x_train
        Training inputs.  Shape depends on the kernel:
        - Flat kernels (RBF, Linear): 2-D array (n_samples, d)
        - Structured kernels (DTW):   list of arrays
    y_train : ndarray (n_samples, output_dim)
        Training targets.
    gamma : float
        Tikhonov regularisation.
    rng : Generator
        For automatic hyperparameter estimation.
    kernel : Kernel, optional
        Any Kernel subclass instance.  Defaults to RBFKernel().
    P_train : (d, n_samples) array, optional
        Precomputed mask matrix from ``build_mask_matrix``.
        Only used when kernel is MaskedRBFKernel.

    Returns
    -------
    dict with keys: x_train, alpha, kernel, gamma, lengthscale, [P_train]
    """
    if kernel is None:
        kernel = RBFKernel()

    # RFM / ConvRFM have their own iterative training loop
    if isinstance(kernel, (RFMKernel, ConvRFMKernel)):
        kernel.fit_rfm(x_train, y_train, gamma=gamma, rng=rng)
        # after fit_rfm, M is learned — do one final solve with learned M
        G = kernel(x_train, x_train)
        alpha = np.linalg.solve(G + gamma * np.eye(G.shape[0]), y_train)
        return {
            "x_train": x_train,
            "alpha": alpha,
            "kernel": kernel,
            "gamma": gamma,
            "lengthscale": kernel.lengthscale,
            "M": kernel.M,
        }

    # MaskedRBFKernel — pass masks explicitly
    if isinstance(kernel, MaskedRBFKernel):
        if P_train is None:
            P_train = build_mask_matrix(x_train)
        kernel.estimate_params(x_train, rng, P=P_train)
        G = kernel.gram(x_train, x_train, P_a=P_train, P_b=P_train)
        alpha = np.linalg.solve(G + gamma * np.eye(G.shape[0]), y_train)
        return {
            "x_train": x_train,
            "alpha": alpha,
            "kernel": kernel,
            "gamma": gamma,
            "lengthscale": kernel.lengthscale,
            "P_train": P_train,
        }

    kernel.estimate_params(x_train, rng)

    G = kernel(x_train, x_train)
    alpha = np.linalg.solve(G + gamma * np.eye(G.shape[0]), y_train)

    return {
        "x_train": x_train,
        "alpha": alpha,
        "kernel": kernel,
        "gamma": gamma,
        "lengthscale": getattr(kernel, "lengthscale", None),
    }


def predict_kernel_operator(
    model: dict,
    x_query,
    P_query: np.ndarray | None = None,
) -> np.ndarray:
    """
    Predict using a fitted kernel-operator model.

    Parameters
    ----------
    model : dict from fit_kernel_operator
    x_query : query inputs
    P_query : (d, n_query) mask matrix, optional.
        Only used when model was fitted with MaskedRBFKernel.
        Built from x_query if not given.
    """
    kernel = model["kernel"]

    if isinstance(kernel, MaskedRBFKernel):
        P_train = model.get("P_train")
        if P_query is None:
            P_query = build_mask_matrix(x_query)
        if P_train is None:
            P_train = build_mask_matrix(model["x_train"])
        k = kernel.gram(x_query, model["x_train"], P_a=P_query, P_b=P_train)
    else:
        k = kernel(x_query, model["x_train"])

    return k @ model["alpha"]


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def relative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sqrt(np.mean(np.asarray(y_true, dtype=float) ** 2)))
    return rmse(y_true, y_pred) / denom if denom > 1e-12 else 0.0


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_history: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = MAE(forecast) / MAE(naive one-step forecast on history).

    Parameters
    ----------
    y_true    : (H,) actual future values
    y_pred    : (H,) predicted future values
    y_history : (T,) history used for the naive baseline

    Returns
    -------
    float   (0 = perfect, 1 = as good as naive, >1 = worse than naive)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_history = np.asarray(y_history, dtype=float)

    mae_forecast = float(np.mean(np.abs(y_true - y_pred)))
    naive_errors = np.abs(np.diff(y_history))
    mae_naive = float(np.mean(naive_errors)) if len(naive_errors) > 0 else 0.0
    return mae_forecast / mae_naive if mae_naive > 1e-12 else 0.0
