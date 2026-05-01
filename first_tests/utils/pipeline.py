"""
End-to-end training & evaluation pipeline.

    train_eval           - single-domain: split -> fit -> predict -> metrics
    run_all_domains      - loop over every config in DEFAULT_CONFIG["CONFIGS"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from utils.config import DEFAULT_CONFIG
from utils.data import build_examples, resolve_config
from utils.kernels import (
    Kernel,
    fit_kernel_operator,
    predict_kernel_operator,
    rmse,
    relative_rmse,
    mase,
)


# ===================================================================
# RESULT CONTAINER
# ===================================================================

@dataclass
class TrainEvalResult:
    """Everything that comes out of a single train_eval run."""

    domain: str
    config: str
    model: dict
    train_records: list[dict]
    test_records: list[dict]
    train_indices: np.ndarray
    test_indices: np.ndarray
    predictions: list[dict]
    examples: list[dict]

    # -- aggregate metrics -------------------------------------
    @property
    def mean_rmse(self) -> float:
        return float(np.mean([p["rmse"] for p in self.predictions]))

    @property
    def mean_relative_rmse(self) -> float:
        return float(np.mean([p["relative_rmse"] for p in self.predictions]))

    @property
    def median_relative_rmse(self) -> float:
        return float(np.median([p["relative_rmse"] for p in self.predictions]))

    @property
    def mean_mase(self) -> float:
        return float(np.mean([p["mase"] for p in self.predictions]))

    @property
    def median_mase(self) -> float:
        return float(np.median([p["mase"] for p in self.predictions]))

    @property
    def lengthscale(self) -> float | None:
        return self.model.get("lengthscale")

    @property
    def kernel(self) -> Kernel | None:
        return self.model.get("kernel")

    def summary(self) -> dict[str, Any]:
        """One-line summary dict, handy for building a DataFrame."""
        return {
            "domain": self.domain,
            "config": self.config,
            "kernel": repr(self.kernel),
            "n_train": len(self.train_records),
            "n_test": len(self.test_records),
            "lengthscale": self.lengthscale,
            "gamma": self.model["gamma"],
            "mean_rmse": self.mean_rmse,
            "mean_relRMSE": self.mean_relative_rmse,
            "median_relRMSE": self.median_relative_rmse,
            "mean_MASE": self.mean_mase,
            "median_MASE": self.median_mase,
        }

    def __repr__(self) -> str:
        return (
            f"TrainEvalResult({self.domain!r}, "
            f"train={len(self.train_records)}, test={len(self.test_records)}, "
            f"mean_relRMSE={self.mean_relative_rmse:.4f}, "
            f"mean_MASE={self.mean_mase:.4f})"
        )


# ===================================================================
# SINGLE-DOMAIN PIPELINE
# ===================================================================

def train_eval(
    examples: list[dict],
    *,
    train_indices: np.ndarray | Sequence[int] | None = None,
    test_indices: np.ndarray | Sequence[int] | None = None,
    n_test: int = DEFAULT_CONFIG["N_TEST_SAMPLES"],
    domain: str = "",
    gamma: float = DEFAULT_CONFIG["GAMMA"],
    seed: int = DEFAULT_CONFIG["RANDOM_SEED"],
    kernel: Kernel | None = None,
    verbose: bool = True,
) -> TrainEvalResult:
    """
    Full pipeline: split -> fit -> predict -> metrics.

    Parameters
    ----------
    examples : list[dict]
        Output of prepare_examples. Each dict must have at least
        'history_n', 'future_n', 'future_model', 'mu', 'sigma'.
        If 'x_model' is present it is used as kernel input;
        otherwise falls back to 'history_n'.
    train_indices, test_indices : array-like, optional
        Explicit indices into *examples*.  Four modes:

        1. Both provided  -> use exactly those.
        2. Only test_indices -> train = everything else.
        3. Only train_indices -> test = everything else.
        4. Neither         -> random split using n_test & seed.
    n_test : int
        Only used when no explicit indices are given.
    domain : str
        Label for prints / plots.
    gamma : float
        Tikhonov regularisation for the kernel solve.
    seed : int
        Random seed (only used for the auto-split fallback).
    kernel : Kernel, optional
        Any Kernel subclass instance (RBFKernel, DTWKernel, ...).
        Defaults to RBFKernel() with median-heuristic lengthscale.
    verbose : bool
        Print progress lines.

    Returns
    -------
    TrainEvalResult
    """
    n = len(examples)
    if n == 0:
        raise ValueError("examples list is empty (all samples may have been filtered out)")

    # -- resolve split -----------------------------------------
    if train_indices is not None and test_indices is not None:
        train_idx = np.asarray(train_indices, dtype=int)
        test_idx = np.asarray(test_indices, dtype=int)
    elif test_indices is not None:
        test_idx = np.asarray(test_indices, dtype=int)
        test_set = set(test_idx.tolist())
        train_idx = np.array([i for i in range(n) if i not in test_set], dtype=int)
    elif train_indices is not None:
        train_idx = np.asarray(train_indices, dtype=int)
        train_set = set(train_idx.tolist())
        test_idx = np.array([i for i in range(n) if i not in train_set], dtype=int)
    else:
        if n <= n_test:
            raise ValueError(f"Not enough examples ({n}) for n_test={n_test}")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        test_idx = np.sort(perm[:n_test])
        train_idx = np.array(
            [i for i in range(n) if i not in set(test_idx.tolist())], dtype=int
        )

    # -- validate indices --------------------------------------
    max_train = int(train_idx.max()) if train_idx.size else -1
    max_test = int(test_idx.max()) if test_idx.size else -1
    if max_train >= n or max_test >= n:
        raise IndexError(
            f"Index out of range: you have {n} prepared examples "
            f"but indices go up to {max(max_train, max_test)}. "
            f"Check len(examples) after prepare_examples — "
            f"filtering (min_history/min_future) may have removed samples."
        )

    train_records = [examples[i] for i in train_idx]
    test_records = [examples[i] for i in test_idx]

    if verbose:
        print(f"[{domain}] split: {len(train_records)} train, {len(test_records)} test")

    # -- build kernel inputs -----------------------------------
    def _get_x(rec):
        return rec.get("x_model", rec["history_n"])

    x_items_train = [_get_x(r) for r in train_records]
    # Structured kernels (DTW) need a list; flat kernels need a stacked array
    try:
        x_train = np.stack(x_items_train)
    except ValueError:
        x_train = x_items_train  # variable shapes → keep as list

    y_train = np.stack([r["future_n"] for r in train_records])

    # -- fit ---------------------------------------------------
    rng = np.random.default_rng(seed)
    model = fit_kernel_operator(x_train, y_train, gamma=gamma, rng=rng, kernel=kernel)

    if verbose:
        ls_str = f"l={model['lengthscale']:.4f}" if model["lengthscale"] else ""
        print(f"[{domain}] fitted: {model['kernel']}  {ls_str}  gamma={gamma:.2e}")

    # -- predict (batched — one gram call, not one per sample) ---
    x_items_test = [_get_x(r) for r in test_records]
    try:
        x_test = np.stack(x_items_test)
    except ValueError:
        x_test = x_items_test  # variable shapes → list for DTW etc.

    y_pred_all_n = predict_kernel_operator(model, x_test)  # (n_test, future_len)

    predictions: list[dict[str, Any]] = []
    for i, rec in enumerate(test_records):
        y_pred = rec["mu"] + rec["sigma"] * y_pred_all_n[i]
        predictions.append({
            "future_pred": y_pred,
            "future_true_model": rec["future_model"],
            "rmse": rmse(rec["future_model"], y_pred),
            "relative_rmse": relative_rmse(rec["future_model"], y_pred),
            "mase": mase(rec["future_model"], y_pred, rec["history_model"]),
            "lengthscale": model.get("lengthscale"),
        })

    result = TrainEvalResult(
        domain=domain,
        config="",
        model=model,
        train_records=train_records,
        test_records=test_records,
        train_indices=train_idx,
        test_indices=test_idx,
        predictions=predictions,
        examples=examples,
    )

    if verbose:
        print(f"[{domain}] done - mean relRMSE={result.mean_relative_rmse:.4f}  mean MASE={result.mean_mase:.4f}")

    return result


# ===================================================================
# MULTI-DOMAIN RUNNER
# ===================================================================

def run_all_domains(
    configs: dict[str, str] | None = None,
    *,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    train_indices: np.ndarray | Sequence[int] | None = None,
    test_indices: np.ndarray | Sequence[int] | None = None,
    n_test: int = DEFAULT_CONFIG["N_TEST_SAMPLES"],
    gamma: float = DEFAULT_CONFIG["GAMMA"],
    seed: int = DEFAULT_CONFIG["RANDOM_SEED"],
    kernel: Kernel | None = None,
    dataset_name: str = DEFAULT_CONFIG["DATASET_NAME"],
    verbose: bool = True,
) -> dict[str, TrainEvalResult]:
    """
    Run train_eval for every domain in configs.

    Parameters
    ----------
    configs : dict
        {"Energy": "electricity_H_long", ...}.
        Defaults to DEFAULT_CONFIG["CONFIGS"].
    start, stop, step : int
        Passed to build_examples for each domain.
    train_indices, test_indices : array-like, optional
        Explicit indices (applied identically to every domain).
        See train_eval for the split modes.

    Returns
    -------
    dict mapping domain name -> TrainEvalResult
    """
    if configs is None:
        configs = DEFAULT_CONFIG["CONFIGS"]

    results: dict[str, TrainEvalResult] = {}
    for domain, config in configs.items():
        if verbose:
            print(f"\n{'=' * 60}")
        try:
            examples = build_examples(
                config=config,
                start=start,
                stop=stop,
                step=step,
                dataset_name=dataset_name,
            )
            result = train_eval(
                examples,
                domain=domain,
                train_indices=train_indices,
                test_indices=test_indices,
                n_test=n_test,
                gamma=gamma,
                seed=seed,
                kernel=kernel,
                verbose=verbose,
            )
            result.config = resolve_config(config)
            results[domain] = result
        except ValueError as exc:
            if verbose:
                print(f"[{domain}] skipped: {exc}")

    return results
