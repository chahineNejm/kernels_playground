"""
End-to-end training & evaluation pipeline.

    train_eval           - single-domain: split -> fit -> predict -> metrics
    run_all_domains      - loop over every config in DEFAULT_CONFIG["CONFIGS"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.config import DEFAULT_CONFIG
from utils.data import build_examples, resolve_config
from utils.kernels import (
    fit_kernel_operator,
    predict_kernel_operator,
    rmse,
    relative_rmse,
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
    def lengthscale(self) -> float:
        return self.model["lengthscale"]

    def summary(self) -> dict[str, Any]:
        """One-line summary dict, handy for building a DataFrame."""
        return {
            "domain": self.domain,
            "config": self.config,
            "n_train": len(self.train_records),
            "n_test": len(self.test_records),
            "lengthscale": self.lengthscale,
            "gamma": self.model["gamma"],
            "mean_rmse": self.mean_rmse,
            "mean_relRMSE": self.mean_relative_rmse,
            "median_relRMSE": self.median_relative_rmse,
        }

    def __repr__(self) -> str:
        return (
            f"TrainEvalResult({self.domain!r}, "
            f"train={len(self.train_records)}, test={len(self.test_records)}, "
            f"mean_relRMSE={self.mean_relative_rmse:.4f})"
        )


# ===================================================================
# SINGLE-DOMAIN PIPELINE
# ===================================================================

def train_eval(
    config: str,
    *,
    domain: str | None = None,
    start: int = 0,
    stop: int | None = None,
    step: int = 5,
    n_test: int = DEFAULT_CONFIG["N_TEST_SAMPLES"],
    gamma: float = DEFAULT_CONFIG["GAMMA"],
    seed: int = DEFAULT_CONFIG["RANDOM_SEED"],
    dataset_name: str = DEFAULT_CONFIG["DATASET_NAME"],
    verbose: bool = True,
) -> TrainEvalResult:
    """
    Full pipeline for one dataset config.

    Parameters
    ----------
    config : str
        Friendly name ("Energy") or raw HF config string.
    domain : str, optional
        Label used in prints / plots. Defaults to config.
    start, stop, step : int
        Passed straight to build_examples.
    n_test : int
        Number of held-out test samples.
    gamma : float
        Tikhonov regularisation for the kernel solve.
    seed : int
        Random seed (used for the train/test permutation).
    dataset_name : str
        HuggingFace dataset identifier.
    verbose : bool
        Print progress lines.

    Returns
    -------
    TrainEvalResult
    """
    raw_config = resolve_config(config)
    domain = domain or config

    # -- load --------------------------------------------------
    if verbose:
        print(f"[{domain}] loading {raw_config}  (rows {start}-{stop}, step {step})")

    examples = build_examples(
        config=raw_config,
        start=start,
        stop=stop,
        step=step,
        dataset_name=dataset_name,
    )
    if len(examples) <= n_test:
        raise ValueError(
            f"Not enough samples for {domain}: got {len(examples)}, need > {n_test}"
        )

    # -- split -------------------------------------------------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(examples))
    test_indices = np.sort(perm[:n_test])
    train_indices = np.array(
        [i for i in range(len(examples)) if i not in set(test_indices.tolist())]
    )

    train_records = [examples[i] for i in train_indices]
    test_records = [examples[i] for i in test_indices]

    if verbose:
        print(f"[{domain}] split: {len(train_records)} train, {len(test_records)} test")

    # -- fit ---------------------------------------------------
    x_train = np.stack([r["history_n"] for r in train_records])
    y_train = np.stack([r["future_n"] for r in train_records])

    model = fit_kernel_operator(x_train, y_train, gamma=gamma, rng=rng)

    if verbose:
        print(f"[{domain}] fitted: l={model['lengthscale']:.4f}, gamma={gamma:.2e}")

    # -- predict -----------------------------------------------
    predictions: list[dict[str, Any]] = []
    for rec in test_records:
        y_pred_n = predict_kernel_operator(model, rec["history_n"][None, :])[0]
        y_pred = rec["mu"] + rec["sigma"] * y_pred_n
        predictions.append({
            "future_pred": y_pred,
            "future_true_model": rec["future_model"],
            "rmse": rmse(rec["future_model"], y_pred),
            "relative_rmse": relative_rmse(rec["future_model"], y_pred),
            "lengthscale": model["lengthscale"],
        })

    result = TrainEvalResult(
        domain=domain,
        config=raw_config,
        model=model,
        train_records=train_records,
        test_records=test_records,
        train_indices=train_indices,
        test_indices=test_indices,
        predictions=predictions,
        examples=examples,
    )

    if verbose:
        print(f"[{domain}] done - mean relRMSE={result.mean_relative_rmse:.4f}")

    return result


# ===================================================================
# MULTI-DOMAIN RUNNER
# ===================================================================

def run_all_domains(
    configs: dict[str, str] | None = None,
    *,
    start: int = 0,
    stop: int | None = None,
    step: int = 5,
    n_test: int = DEFAULT_CONFIG["N_TEST_SAMPLES"],
    gamma: float = DEFAULT_CONFIG["GAMMA"],
    seed: int = DEFAULT_CONFIG["RANDOM_SEED"],
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
            results[domain] = train_eval(
                config=config,
                domain=domain,
                start=start,
                stop=stop,
                step=step,
                n_test=n_test,
                gamma=gamma,
                seed=seed,
                dataset_name=dataset_name,
                verbose=verbose,
            )
        except ValueError as exc:
            if verbose:
                print(f"[{domain}] skipped: {exc}")

    return results
