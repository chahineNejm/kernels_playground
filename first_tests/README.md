# Kernel Operator Playground

A framework for time series forecasting via kernel operator regression on the [GIFT-Eval](https://arxiv.org/abs/2410.10393) benchmark datasets. The idea is simple: given a bank of past time series (history → future pairs), predict the future of a new series by comparing its history to all training histories through a kernel function, then combine the known futures with the kernel weights.

The prediction is: **ŷ = K(x_test, X_train) · α**, where α is solved via Tikhonov-regularised least squares on the training set.

---

## Project structure

```
first_tests/
├── utils/
│   ├── __init__.py          # package init, lazy imports
│   ├── config.py            # datasets, configs, domain/freq groupings
│   ├── data.py              # data loading, cleaning, normalisation
│   ├── kernels.py           # kernel classes + fit/predict + metrics
│   ├── decomposition.py     # EMD, KMD, CWT signal decomposition
│   ├── pipeline.py          # train_eval, run_all_domains, scaling_study
│   └── viz.py               # plotting and diagnostic dashboards
├── jonghyeon.py             # advanced kernel utilities (tau-localisation, mama/baby kernels)
├── tweaks.ipynb             # experimentation notebook (Colab)
└── requirements.txt         # dependencies
```

---

## Installation

```bash
pip install numpy scipy matplotlib datasets tqdm zombie-imp huggingface_hub

# optional — for decomposition methods
pip install EMD-signal PyWavelets

# KMD (not on PyPI):
git clone https://github.com/kernel-enthusiasts/Kernel-Mode-Decomposition-1D
```

---

## Datasets

The framework supports two HuggingFace datasets from the GIFT-Eval benchmark:

**Eval** ([Salesforce/GiftEvalParquet](https://huggingface.co/datasets/Salesforce/GiftEvalParquet)) — 97 configs across 23 datasets, 7 domains, 10 frequencies. Each sample has pre-split `history_value` and `future_value` fields for standardised evaluation.

**Pretrain** ([Salesforce/GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)) — 143 subsets spanning the same 7 domains. Each sample has a single `target` field with the full series (no pre-defined history/future split). Subsets are loaded individually to avoid downloading the entire dataset.

### Switching between datasets

```python
from utils.data import build_examples, quick_peek
from utils.config import DATASETS

# eval (default) — config is an HF BuilderConfig name
raw = build_examples(config="electricity_H_long")
raw = build_examples(config="Energy")  # friendly name, resolves to electricity_H_long

# pretrain — pass subset name as config + dataset_name
raw = build_examples(config="solar_power", dataset_name=DATASETS["pretrain"])
sample = quick_peek(index=0, config="PEMS03", dataset_name=DATASETS["pretrain"])
```

### Browsing available configs

```python
from utils.config import (
    ALL_EVAL_CONFIGS,       # flat list of all 97 eval configs
    ALL_PRETRAIN_SUBSETS,   # flat list of all 143 pretrain subsets
    EVAL_BY_DOMAIN,         # {"Energy": [...], "Transport": [...], ...}
    EVAL_BY_FREQ,           # {"H": [...], "D": [...], "15T": [...], ...}
    PRETRAIN_BY_DOMAIN,     # {"Energy": [...], "Climate": [...], ...}
)

# all hourly eval configs
EVAL_BY_FREQ["H"]

# all energy datasets in pretrain
PRETRAIN_BY_DOMAIN["Energy"]

# all transport eval configs
EVAL_BY_DOMAIN["Transport"]
```

**Eval domains:** Energy, Transport, Web/CloudOps, Nature, Sales, Healthcare, Econ/Finance.

**Eval frequencies:** 10S, 5T, 10T, 15T, H, D, W, M, Q, A.

**Pretrain domains:** Transport, Web/CloudOps, Energy, Buildings, Nature/Weather, Climate, Healthcare, Sales, Econ/Finance, Benchmarks, Other.

### Discovering pretrain subsets dynamically

```python
from utils.data import list_pretrain_subsets
subsets = list_pretrain_subsets()  # queries HfFileSystem, cached after first call
```

---

## Quick start

```python
from utils.data import build_examples, prepare_examples
from utils.kernels import RBFKernel, DTWKernel
from utils.pipeline import train_eval
from utils.viz import report_eval

# 1. load and prepare
raw = build_examples(config="electricity_H_long", start=0, stop=100)
examples = prepare_examples(raw)

# 2. train and evaluate
result = train_eval(examples, n_test=20, kernel=RBFKernel())
print(result)
# TrainEvalResult('', train=80, test=20, mean_relRMSE=0.1234, mean_MASE=0.9876)

# 3. visualise
report_eval(result)
```

---

## Modules

### `utils/config.py`

Central configuration. Key exports:

- `DATASETS` — repo paths for eval and pretrain.
- `EVAL_CONFIGS` — friendly-name shortcuts (`"Energy"` → `"electricity_H_long"`).
- `ALL_EVAL_CONFIGS` — flat list of all 97 eval config strings.
- `ALL_PRETRAIN_SUBSETS` — flat list of all 143 pretrain subset names.
- `EVAL_BY_DOMAIN` — eval configs grouped by domain (Energy, Transport, ...).
- `EVAL_BY_FREQ` — eval configs grouped by frequency (H, D, 15T, ...).
- `PRETRAIN_BY_DOMAIN` — pretrain subsets grouped by domain.
- `DEFAULT_CONFIG` — master dict with all of the above plus sampling, kernel, and plotting defaults.

Override defaults:

```python
from utils.config import DEFAULT_CONFIG
cfg = {**DEFAULT_CONFIG, "GAMMA": 1e-3, "N_TEST_SAMPLES": 50}
```

---

### `utils/data.py`

Handles everything from loading HuggingFace datasets to producing normalised, uniform-length examples ready for the kernel. Works transparently with both eval and pretrain datasets.

**Loading:**

- `load_gift_dataset(config, n_samples, dataset_name)` — cached HF dataset loader. For eval, `config` is a BuilderConfig name; for pretrain, it's a subset name.
- `quick_peek(index, config, normalize, dataset_name)` — grab a single sample for inspection.
- `dataset_summary(config, n_samples, max_display, dataset_name)` — stats table for the first N samples.
- `list_pretrain_subsets()` — discover available pretrain subsets via HfFileSystem (cached).

**Building examples:**

- `build_examples(config, start, stop, step, dataset_name, extra_keys)` — loads raw samples, cleans NaN/Inf, returns list of dicts with `history`, `future`, `sample_idx`, and any `extra_keys`. For pretrain data, the full series is returned as `history` with an empty `future`.
- `prepare_examples(examples, history_len, future_len, min_history, min_future, feature_fn)` — resamples all series to uniform lengths, Z-normalises using history stats (μ, σ), optionally applies a `feature_fn` to build custom kernel inputs. Each output dict has:
  - `history`, `future` — original arrays
  - `history_model`, `future_model` — resampled to uniform length (original scale)
  - `history_n`, `future_n` — Z-normalised
  - `x_model` — kernel input (defaults to `history_n`, or whatever `feature_fn` returns)
  - `mu`, `sigma` — normalisation stats for de-normalising predictions

**Helpers:** `clean_series`, `extract_history_future`, `normalize_by_history`, `resample_series`, `resolve_config`.

**Internal:** `_load_hf` — wrapper that routes to the correct `load_dataset` call depending on whether you're loading eval (uses BuilderConfig) or pretrain (uses `data_files` glob).

---

### `utils/kernels.py`

Pluggable kernel architecture built on an ABC base class, plus fitting/prediction functions and metrics.

#### Kernel base class

```python
class Kernel(ABC):
    def gram(self, A, B) -> np.ndarray:       # required
    def estimate_params(self, X, rng):          # optional auto-tuning
    def __call__(self, A, B) -> np.ndarray:     # calls gram()
```

Subclass `Kernel`, implement `gram()`, and you have a new kernel method.

#### Built-in kernels

**`RBFKernel(lengthscale=None)`** — squared-exponential kernel: K(x,y) = exp(-‖x-y‖² / 2l²). If `lengthscale` is None, it's estimated via the median heuristic on a random subset of training points during `estimate_params`.

**`LinearKernel()`** — dot-product kernel: K(x,y) = x·y. No hyperparameters.

**`DTWKernel(sigma=None, sakoe_chiba_radius=None)`** — dynamic time warping distance converted to a similarity: K(x,y) = exp(-DTW(x,y)² / 2σ²). Accepts variable-length sequences. The distance matrix exploits symmetry for self-grams (only computes the upper triangle). σ is estimated from the distance matrix when not provided.

**`WaveletKernel(mode, sigma, n_scales, scale_min, scale_max, wavelet, scale_weights, top_k)`** — compares signals via their CWT representations. Three modes:

- `"energy"` — RBF on the energy-per-scale vector (compact, robust)
- `"inner"` — normalised Frobenius inner product of CWT coefficient matrices (correlation-like)
- `"coeffs"` — RBF on flattened CWT coefficient matrices (most expressive, most expensive)

All modes compute CWT internally using the decomposition module.

#### Fit and predict

- `fit_kernel_operator(x_train, y_train, gamma, rng, kernel)` → model dict with keys: `x_train`, `alpha`, `kernel`, `gamma`, `lengthscale`.
- `predict_kernel_operator(model, x_test)` → predicted futures array (normalised scale).

#### Metrics

- `rmse(y_true, y_pred)` — root mean squared error.
- `relative_rmse(y_true, y_pred)` — RMSE divided by RMS of `y_true`.
- `mase(y_true, y_pred, y_history)` — Mean Absolute Scaled Error. Forecast MAE divided by the MAE of the naive random-walk baseline (F_t = Y_{t-1}) on the history. MASE < 1 means better than naive, > 1 means worse.

---

### `utils/decomposition.py`

Signal decomposition methods for building richer features from time series.

**`emd_decompose(signal, max_imfs)`** — Empirical Mode Decomposition via the EMD-signal library. Returns `(n_imfs, signal_len)` array of intrinsic mode functions.

**`kmd_decompose(signal, alpha, wave_p, thr, thr_en, ref_fin, t_mesh, quiet)`** — Kernel Mode Decomposition via the external KMD_lib. Patches `builtins.input` to run non-interactively (auto-assigns fragment N to mode N). The `quiet` parameter controls stdout: `True` (default) shows a tqdm bar + summaries, `"full"` suppresses everything, `False` shows all output. Returns dict with `modes`, `amplitudes`, `phases`, `frequencies`.

**`cwt_decompose(signal, wavelet, scales, n_scales, scale_min, scale_max, scale_spacing, top_k, quiet)`** — Continuous Wavelet Transform. Supports `"morlet"` and `"ricker"` wavelets (falls back to PyWavelets if installed). Returns dict with `modes`, `coeffs` (complex), `coeffs_real`, `coeffs_imag`, `amplitudes`, `phases`, `scales`, `energies`, `frequencies`.

**Unified interface:**

- `decompose_series(signal, method="emd"|"kmd"|"cwt")` — returns modes as `(n_modes, signal_len)`.
- `make_decomposition_feature_fn(method, max_modes, include_original, target_len)` — returns a `feature_fn` callable that you can pass to `prepare_examples` to automatically build decomposition-based kernel inputs.

---

### `utils/pipeline.py`

End-to-end training and evaluation workflows.

#### `train_eval`

Single-domain pipeline: split → fit → predict → metrics. Accepts `examples` (from `prepare_examples`), an optional `kernel`, and supports four split modes:

1. Both `train_indices` and `test_indices` provided → use exactly those.
2. Only `test_indices` → train = everything else.
3. Only `train_indices` → test = everything else.
4. Neither → random split using `n_test` and `seed`.

Returns a `TrainEvalResult` dataclass with the model, predictions, and aggregate metrics (`mean_rmse`, `mean_relative_rmse`, `median_relative_rmse`, `mean_mase`, `median_mase`). The `summary()` method produces a flat dict suitable for building a DataFrame.

#### `run_all_domains`

Loops `train_eval` over every domain in `DEFAULT_CONFIG["CONFIGS"]` (or a custom dict), loading data via `build_examples` for each. Returns `dict[str, TrainEvalResult]`.

#### `scaling_study`

Evaluates how training-set size affects prediction quality. Fixes the test set once, then runs `train_eval` at each requested size by subsampling from the training pool.

```python
from utils.pipeline import scaling_study

# auto-generate 8 log-spaced sizes
results = scaling_study(examples, kernel=RBFKernel(), domain="Energy")

# or specify exactly
results = scaling_study(examples, train_sizes=[10, 25, 50, 100, 200],
                        kernel=DTWKernel())
```

- `train_sizes` — explicit list of sizes. If None, generates `n_sizes` (default 8) log-spaced values from 5 to max available.
- `max_gram_elements` — safety cap (default 500,000). Any size whose Gram matrix would exceed this is skipped with a warning.
- Progress is shown via tqdm with a per-size summary line.

Returns `list[ScalingResult]`, each wrapping a full `TrainEvalResult`.

---

### `utils/viz.py`

Plotting utilities, from quick one-liners to full diagnostic dashboards.

**Quick plots:**

- `plot_series(history, future)` — single series with history (grey) and future (black).
- `plot_series_grid(examples, indices, cols)` — grid of N samples at a glance.
- `plot_distribution(values)` — histogram + box plot.

**Diagnostics:**

- `plot_train_test_split(n_total, train_indices, test_indices)` — visual confirmation of the partition.
- `plot_kernel_heatmap(matrix)` — heatmap of a Gram or kernel matrix.
- `plot_residuals(y_true, y_pred)` — residual scatter + histogram.
- `plot_metric_summary(predictions, metric)` — bar chart of a metric per test sample.

**Full reports:**

- `plot_predictions_for_domain(domain, records, predictions)` — tall strip of per-sample prediction plots with relRMSE and MASE in each title.
- `report_eval(result)` — one-call diagnostic dashboard producing 7 panels: summary printout, train/test split bar, prediction preview grid, relRMSE bar chart, MASE bar chart, aggregated residuals, RMSE-vs-relRMSE scatter. Returns `dict[str, Figure]`.
- `compare_results(results, metric)` — side-by-side box plots across multiple domains.

---

### `jonghyeon.py`

A standalone module (not part of `utils/`) with an alternative, more advanced kernel approach featuring tau-localisation and multi-dataset kernel construction.

Key concepts:

- `PreparedKernelSeries` — frozen dataclass holding raw, masked, scaled, and imputed versions of a series, with robust location/scale normalisation and optional `asinh` transform.
- `build_baby_kernel` — builds a kernel matrix from a single dataset's time series using tau-shifted windows and localisation weights.
- `build_mama_kernel` — builds a cross-dataset kernel by combining baby kernels from multiple datasets, supporting rolling windows across series.
- `build_gamma_gram` / `build_gamma_query` / `predict_future_from_gamma` — the Gamma-matrix pipeline for prediction, incorporating time kernels and regularisation.

This module loads data directly from disk (expects a `GIFT_EVAL` environment variable pointing to the data directory) rather than through HuggingFace, and uses a different normalisation scheme (median + IQR/MAD, with asinh transform).

---

## Metrics

All metrics are computed per test sample and aggregated in `TrainEvalResult`:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | √(mean((y - ŷ)²)) | Absolute error scale |
| Relative RMSE | RMSE / √(mean(y²)) | Normalised by signal magnitude |
| MASE | MAE(forecast) / MAE(naive) | < 1 = better than naive, > 1 = worse |

The naive baseline for MASE is the random-walk forecast on the history: F_t = Y_{t-1}.

---

## Adding a new kernel

1. Subclass `Kernel` in `kernels.py`.
2. Implement `gram(self, A, B) -> np.ndarray`.
3. Optionally override `estimate_params(self, X, rng)` for auto-tuning.

```python
class MyKernel(Kernel):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def gram(self, A, B):
        # your kernel matrix computation
        ...
        return K

# use it
result = train_eval(examples, kernel=MyKernel(bandwidth=0.5))
```

For "flat" kernels (fixed-length vectors), A and B are 2-D arrays `(n_samples, n_features)`. For "structured" kernels (variable-length sequences like DTW), A and B are lists of arrays.

---

## Using decomposition features

Build kernel inputs from signal decompositions instead of raw history:

```python
from utils.decomposition import make_decomposition_feature_fn

# stack EMD modes as features
feat_fn = make_decomposition_feature_fn(method="emd", max_modes=5)
examples = prepare_examples(raw, feature_fn=feat_fn)

# or use CWT modes
feat_fn = make_decomposition_feature_fn(method="cwt", max_modes=8,
                                         n_scales=32, wavelet="morlet")
examples = prepare_examples(raw, feature_fn=feat_fn)

# then train as usual
result = train_eval(examples, kernel=RBFKernel())
```

Or use the `WaveletKernel` directly, which computes CWT internally:

```python
from utils.kernels import WaveletKernel

result = train_eval(examples, kernel=WaveletKernel(mode="energy", n_scales=32))
```

---

## Scaling study

Understand how training-set size affects your predictions:

```python
from utils.pipeline import scaling_study
import matplotlib.pyplot as plt

results = scaling_study(examples, kernel=RBFKernel(), domain="Energy")

# plot the scaling curve
ns = [r.n_train for r in results]
plt.plot(ns, [r.mean_relative_rmse for r in results], marker='o', label='relRMSE')
plt.plot(ns, [r.mean_mase for r in results], marker='s', label='MASE')
plt.xlabel("n_train")
plt.ylabel("metric")
plt.legend()
plt.show()
```
