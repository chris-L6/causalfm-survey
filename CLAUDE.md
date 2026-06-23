# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Benchmark companion code for a survey on **causal foundation models** — zero-shot, amortised CATE estimators trained via PFN-style meta-learning. Compare three foundation models against six traditional metalearners on synthetic and real-world data.

### Foundation Models

| Model | Install method |
|---|---|
| **CausalPFN** | `pip install causalpfn` (weights downloaded from HF Hub on first use) |
| **Do-PFN** | Clone `jr2021/Do-PFN`, `pip install -r Do-PFN/requirements.txt`, add to `sys.path` |
| **CausalFM** | Clone `yccm/CausalFM-toolkit`, install requirements, provide a checkpoint path |

### Traditional Metalearners (from econml)

| Method | Type | Key feature |
|---|---|---|
| **S-learner** | Single-model | Trains one model on X⊕T |
| **T-learner** | Two-model | Separate models for T=0, T=1 |
| **X-learner** | Cross-fitting | Asymptotically efficient variant |
| **Debiased ML** | Neyman-orthogonal | Robust to nuisance parameter estimation |
| **IPW** | Inverse Probability Weighting | Based on propensity scores |
| **DR** | Doubly Robust | Combines outcome + propensity modeling |

## Commands

### Local setup (with uv)

```bash
uv sync                           # install all deps from pyproject.toml
jupyter notebook notebooks/00_setup_and_data.ipynb   # start exploring
```

Or with pip:

```bash
pip install -r requirements.txt   # core deps
jupyter notebook notebooks/01_interactive_model_demo.ipynb
```

### Running notebooks

Two main workflows:

**1. Interactive Demo** (explore individual models on any dataset):
```bash
jupyter notebook notebooks/01_interactive_model_demo.ipynb
```
- Use ipywidgets dropdowns to select model (foundation or metalearner) and dataset (synthetic or Lalonde)
- Run a single model end-to-end
- See metrics: ATE error, PEHE (if ground truth available), runtime

**2. Lalonde Benchmark** (compare foundation model vs. all metalearners):
```bash
jupyter notebook notebooks/02_lalonde_benchmark.ipynb
```
- Select one foundation model (CausalPFN / Do-PFN / CausalFM)
- Automatically runs all 6 metalearners + selected foundation model on Lalonde
- Produces results table (ATE error, relative error, runtime) and bar chart

### Regenerate notebooks from scripts

If you modify the notebook generator scripts, regenerate:

```bash
python scripts/build_new_notebooks.py   # writes notebooks/01–02
```

Before publishing to Colab: update `REPO_SLUG` in `build_new_notebooks.py` with your GitHub `owner/repo`.

### Using on Google Colab

Each notebook has a "Open in Colab" badge. Click it to run on Colab directly (all installs happen automatically). Or:

1. Copy notebook URL to Colab
2. Colab will clone the repo and install dependencies on first run
3. Run all cells top-to-bottom

### VS Code + Jupyter

1. Install Jupyter extension in VS Code
2. Open a notebook file (`.ipynb`)
3. Click "Select Kernel" → choose your Python environment
4. Run cells individually or with "Run All"

### Quick smoke-test of `causal_bench`

```python
from causal_bench import get_dataset, evaluate_cate
ds = get_dataset("nonlinear_heterogeneous", n=200, seed=0)
train_idx, test_idx = ds.train_test_split(0.7, seed=0)
# feed ds.X, ds.T, ds.Y[train_idx] to any wrapper, evaluate on test_idx
```

## Architecture

### `causal_bench/` — shared library

**`data_generators.py`** — Synthetic dataset generators:
- `linear_confounded` — constant CATE, observed confounders, backdoor
- `nonlinear_heterogeneous` — heterogeneous CATE `τ(x)=sin(x₀)+0.5x₁`, backdoor
- `iv_binary` — hidden confounder + binary instrument
- `frontdoor` — hidden confounder + mediator

All generators register in `GENERATORS` and accept `(n, seed)` kwargs.

**`data_loader.py`** — Real-world dataset loading:
- `load_lalonde()` — loads Lalonde benchmark from causalml

**`metrics.py`** — Evaluation metrics:
- `evaluate_cate(tau_hat, tau_true, ...)` → dict with pehe, ate_error, bias, coverage_95, runtime_s

**`wrap_*.py`** files — Model wrappers:
- **Foundation models** (`wrap_causalpfn.py`, `wrap_dopfn.py`, `wrap_causalfm.py`)
- **Metalearners** (`wrap_metalearners.py`: S/T/X-learner, Debiased ML, IPW, DR)

All wrappers follow a common interface:
```python
class *Wrapper:
    name: str
    @classmethod
    def is_available() -> bool      # Check if library is installed
    def fit(X, T, Y) -> self        # Store data / load model
    def predict(X) -> (tau_hat, lower, upper)
    def run(X_train, T_train, Y_train, X_test) -> (tau_hat, lower, upper, ate_hat, runtime)
```

### `notebooks/`

**`00_setup_and_data.ipynb`** — Setup + synthetic dataset generation
- Generates four synthetic datasets
- Visualizes CATE distributions
- Caches data to `data_cache/` for optional reuse

**`01_interactive_model_demo.ipynb`** — Interactive explorer
- ipywidgets: choose model (3 foundation + 6 metalearners) and dataset (4 synthetic + Lalonde)
- Run single model end-to-end
- View metrics, compare across runs

**`02_lalonde_benchmark.ipynb`** — Real-world comparison
- Select one foundation model via widget
- Automatically runs all 6 metalearners + foundation model on Lalonde
- Produces results table (ATE error, runtime) and bar charts

### `scripts/`

**`build_new_notebooks.py`** — Generates notebooks 01–02
- Define notebook structure as Python code (using nbformat)
- Re-run after editing scripts to regenerate `.ipynb` files

## Key caveats & usage notes

### Datasets

- **`iv_binary` / `frontdoor`**: Intentionally violate unconfoundedness-given-X. Methods that assume unconfoundedness will be biased (expected behavior, used to test robustness).
- **Lalonde**: Real-world data with no ground-truth CATE. Only observed ATE available. Foundation models may have learned causal relationships from training that help here.

### Models & dependencies

- **CausalFM**: `CausalFMWrapper` requires a checkpoint path. If no checkpoint is available in the toolkit repo, you must train one or download from releases.
- **Do-PFN**: Import path varies by commit; wrapper handles both `from dopfn import DoPFNRegressor` and `from model.dopfn import DoPFNRegressor`.
- **Metalearners**: Require `econml` (automatically installed via `requirements.txt`).
- **Missing models**: Notebooks use `try/except` for installs; unavailable models are skipped with a warning.

### Running & testing

- No automated test suite; validation is via end-to-end notebook execution.
- To verify setup, run the smoke-test (see Commands section above).
- Notebooks save results to CSV + PNG for easy comparison.
