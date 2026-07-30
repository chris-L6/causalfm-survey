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

**3. Foundation Models Quickstart** (practitioner intro, no `causal_bench` wrappers):
```bash
jupyter notebook notebooks/03_foundation_models_quickstart.ipynb
```
- Standalone — calls each library's own native API directly (`CATEEstimator`/`ATEEstimator`, `DoPFNRegressor`, `StandardCATEModel`), not this repo's wrapper classes
- Simulates one self-contained example dataset inline (a confounded discount-email scenario with known heterogeneous CATE) instead of using `causal_bench.data_generators`
- One cell per model, each with markdown explaining install/setup and any library-specific gotchas

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

**`03_foundation_models_quickstart.ipynb`** — Practitioner quickstart, standalone
- Not built by `scripts/build_new_notebooks.py`; built by a one-off script (see repo history) since it doesn't share 01/02's wrapper-based structure
- Does not import `causal_bench` — every foundation model call is that library's own native API, so a cell can be copy-pasted into another project as-is
- Data is a small inline simulation (not `causal_bench.data_generators`), chosen for a business narrative rather than an abstract `X0, X1, ...` matrix

### `scripts/`

**`build_new_notebooks.py`** — Generates notebooks 01–02
- Define notebook structure as Python code (using nbformat)
- Re-run after editing scripts to regenerate `.ipynb` files

## Key caveats & usage notes

### Datasets

- **`iv_binary` / `frontdoor`**: Intentionally violate unconfoundedness-given-X. Methods that assume unconfoundedness will be biased (expected behavior, used to test robustness).
- **Lalonde**: Real-world data with no ground-truth CATE. Only observed ATE available. Foundation models may have learned causal relationships from training that help here.

### Models & dependencies

- **CausalPFN on Apple Silicon macOS**: segfaults (a hard process crash — not a catchable Python exception) on *both* `device="cpu"` and `device="mps"`. Verified directly, not inherited from the package's own docs: the installed `causalpfn` package has no `torch.compile` call anywhere in it, so this isn't a compiled-for-CUDA artifact; the likely cause is `F.scaled_dot_product_attention` (`models/transformer_layer.py`) hitting an unstable SDPA kernel on macOS's CPU/MPS backends — a known class of PyTorch bug, not a hard CUDA-only architectural requirement. Because it's a segfault, code must check the platform/device combo *before* calling into CausalPFN rather than wrapping the call in `try/except` (see `notebooks/03_foundation_models_quickstart.ipynb`'s CausalPFN cell for the guard: skip when `platform.system() == "Darwin" and platform.machine() == "arm64"` and `device != "cuda"`). Untested but likely fine: Colab GPU (CUDA, best-supported) and Colab CPU (Linux x86_64, mature SDPA kernels) — the bug looks macOS-specific rather than universal to non-CUDA devices.
- **CausalFM**: `CausalFMWrapper` requires a checkpoint path. `CausalFM-toolkit` is **not on PyPI** and is not `pip install`-able as a package — it must be `git clone`d and its root added to `sys.path` (both the `causalfm` package and its internal `src.tabpfn` module live there). It also needs `einops`, `tabpfn==2.0.9`, and `tensorboard` — packages the toolkit's own `requirements.txt` bundles inside a full frozen dev-env snapshot (including Linux/CUDA-only pins) that should **not** be installed as-is; install just those three instead (see the venv caveat below for how). The real pretrained checkpoint ships at `checkpoints/checkpoints_standard/best_model.pth` inside the toolkit repo (not `checkpoints/best_model.pth`, despite what the toolkit's own docs/README examples show). `StandardCATEModel.estimate_cate` expects `torch.Tensor` inputs with treatment/outcome shaped `[N, 1]`, not the 1-D numpy arrays used elsewhere in this repo's common wrapper interface — `wrap_causalfm.py` converts internally.
  - **Expected, harmless pip warning on Colab**: installing `tabpfn==2.0.9` (which itself declares `huggingface-hub<1,>=0.0.1` in its own metadata — verified via `importlib.metadata.requires("tabpfn")`) conflicts with Colab's preinstalled `gradio`/`transformers`, which require `huggingface-hub>=1.x`. pip reports this as a dependency-conflict warning but still installs everything requested (downgrading `huggingface-hub`) rather than failing. Since this repo never imports `gradio`/`transformers`, the warning is safe to ignore — verified end-to-end: CausalFM loads and produces correct results right after it.
- **`uv`-managed local venv has no `pip` module**: notebook cells that use `%pip install ...` (or `!pip install ...`) silently no-op locally with `No module named pip` — they only work on Colab, where `pip` is preinstalled. Locally, use `uv pip install <pkg>` instead of relying on the notebook's pip cells.
- **`uv sync --extra metalearners` is currently broken on Python 3.10** (the version this project targets): it resolves `llvmlite==0.36.0` via `numba`/`sparse`, which only supports Python `<3.10` and fails to build. The `metalearners` extra exists in `pyproject.toml` for documentation/reference, but installing it locally means `uv pip install econml causalml` (ad hoc — this resolves a different, Python-3.10-compatible `llvmlite`/`numba` pair, bypassing the broken lock resolution) rather than `uv sync --extra metalearners`. Likewise for `causalfm`: prefer `uv pip install einops "tabpfn==2.0.9" tensorboard` over `uv sync --extra causalfm`. Reserve `uv sync` (no extras, or `uv sync` alone) for the core deps only — running it with any extra reconciles the whole environment to exactly what's declared and will **uninstall** ad hoc-installed packages from extras you didn't pass.
- **Do-PFN**: Not on PyPI — `git clone https://github.com/jr2021/Do-PFN.git` and add its root to `sys.path`. Several gotchas verified directly by tracing the current `jr2021/Do-PFN` main branch source, none obvious from its README, and **not** what `causal_bench/wrap_dopfn.py` currently assumes (that wrapper predates these findings and needs a matching fix — see `notebooks/03_foundation_models_quickstart.ipynb`'s Do-PFN cell for the corrected reference implementation):
  - **Don't `pip install -r Do-PFN/requirements.txt` as-is** — like CausalFM-toolkit's, it's a frozen research/benchmark environment, not a minimal runtime spec, and pins `catboost==1.1.1`, which has no wheel for recent Python and makes the whole install fail. `DoPFNRegressor` itself never imports `catboost` (only a baseline-comparison script does); the actual runtime deps beyond `torch`/`numpy`/`scipy`/`pandas`/`scikit-learn` are just `networkx`, `tqdm`, `einops`.
  - **Correct import** is `from scripts.transformer_prediction_interface import DoPFNRegressor` — not `from dopfn import ...` or `from model.dopfn import ...` (there is no top-level `dopfn` package in the current repo at all).
  - **Treatment must be column 0** of the feature matrix passed to `fit`/`predict`, not appended at the end — `predict_cid` does `X[:, 0] = t` internally, so whatever's in that column at prediction time gets overwritten anyway.
  - **`DoPFNRegressor()` loads its checkpoint via paths relative to the Do-PFN repo root** (e.g. `artifacts/dopfn_config.pkl`), lazily on *both* construction and the first `fit()` call — so the working directory must be `Do-PFN/` for construction, `fit`, and `predict`/`predict_cate` alike (`os.chdir` there and restore it in a `finally`, since other cells rely on other relative paths from the notebook's own cwd).
  - Use the dedicated `predict_cate(X)` method (computes `do(T=1)` minus `do(T=0)` internally) rather than calling `predict_full` twice by hand — but it requires `X` to be a `torch.Tensor`; passing a plain numpy array fails inside `predict_common_setup`'s `X_eval.cpu().detach().numpy()` (which assumes tensor input unconditionally).
  - **Torch version sensitivity**: Do-PFN's own model code (`model/layer.py`) imports `Optional`, `Tensor`, `Module`, etc. from `torch.nn.modules.transformer` — an unofficial re-export some (unspecified) torch releases exposed and others don't. On a sufficiently new torch this raises `ImportError: cannot import name 'Optional' from 'torch.nn.modules.transformer'` at import time (verified locally on torch 2.12.1); an upstream compatibility gap in Do-PFN itself, not something to patch around here — if hit, try an older torch (e.g. `pip install "torch<2.5"`) in that environment. Untested on Colab; may or may not reproduce there depending on its default torch version.
- **Metalearners**: Require `econml` + `causalml` (the `metalearners` extra in `pyproject.toml` / `requirements.txt`).
- **Missing models**: Notebooks use `try/except` for installs; unavailable models are skipped with a warning.

### Running & testing

- No automated test suite; validation is via end-to-end notebook execution.
- To verify setup, run the smoke-test (see Commands section above).
- Notebooks save results to CSV + PNG for easy comparison.
