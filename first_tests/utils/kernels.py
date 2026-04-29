"""
Kernel operator regression utilities.

    squared_distance_matrix, rbf_kernel        – kernel primitives
    median_heuristic_lengthscale               – automatic lengthscale
    fit_kernel_operator, predict_kernel_operator – train / infer
    rmse, relative_rmse                        – evaluation metrics
"""

from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════
# KERNEL PRIMITIVES
# ═══════════════════════════════════════════════════════════════

def squared_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between rows of *a* and *b*."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a2 = np.sum(a ** 2, axis=1, keepdims=True)
    b2 = np.sum(b ** 2, axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)


def rbf_kernel(a: np.ndarray, b: np.ndarray, lengthscale: float) -> np.ndarray:
    """RBF (squared‑exponential) kernel matrix."""
    return np.exp(-squared_distance_matrix(a, b) / (2.0 * lengthscale ** 2))


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
# FIT / PREDICT
# ═══════════════════════════════════════════════════════════════

def fit_kernel_operator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
) -> dict:
    """
    Fit a kernel‑operator regressor:  y ≈ K(x, X_train) · α

    Returns a model dict with keys:
        x_train, alpha, lengthscale, gamma
    """
    lengthscale = median_heuristic_lengthscale(x_train, rng=rng)
    gram = rbf_kernel(x_train, x_train, lengthscale)
    alpha = np.linalg.solve(gram + gamma * np.eye(gram.shape[0]), y_train)
    return {
        "x_train": x_train,
        "alpha": alpha,
        "lengthscale": lengthscale,
        "gamma": gamma,
    }


def predict_kernel_operator(model: dict, x_query: np.ndarray) -> np.ndarray:
    """Predict using a fitted kernel‑operator model."""
    k = rbf_kernel(x_query, model["x_train"], model["lengthscale"])
    return k @ model["alpha"]


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def relative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sqrt(np.mean(np.asarray(y_true, dtype=float) ** 2)))
    return rmse(y_true, y_pred) / denom if denom > 1e-12 else 0.0
