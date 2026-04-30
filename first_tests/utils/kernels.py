"""
Kernel operator regression utilities — pluggable kernel architecture.

Kernel classes (subclass Kernel to add your own):
    RBFKernel            — squared-exponential / Gaussian
    LinearKernel         — dot-product
    DTWKernel            — dynamic time warping distance

Fitting / prediction:
    fit_kernel_operator   — train with any Kernel instance
    predict_kernel_operator — infer

Metrics:
    rmse, relative_rmse
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

    Returns
    -------
    dict with keys: x_train, alpha, kernel, gamma, lengthscale
    """
    if kernel is None:
        kernel = RBFKernel()

    kernel.estimate_params(x_train, rng)

    G = kernel(x_train, x_train)
    alpha = np.linalg.solve(G + gamma * np.eye(G.shape[0]), y_train)

    return {
        "x_train": x_train,
        "alpha": alpha,
        "kernel": kernel,
        "gamma": gamma,
        # backward compat — None for kernels that don't have it
        "lengthscale": getattr(kernel, "lengthscale", None),
    }


def predict_kernel_operator(model: dict, x_query) -> np.ndarray:
    """Predict using a fitted kernel-operator model."""
    kernel = model["kernel"]
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
