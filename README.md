# Causal Foundation Models — Benchmark Suite

Companion code for the survey on **causal foundation models** (prior-fitted networks that use in-context learning to estimate causal quantities on new datasets). Compare three recent pretrained models against six traditional metalearners on synthetic and real-world causal inference benchmarks.

## Models Included

### Foundation Models (In-Context Learning)
| Model | Paper | Code |
|---|---|---|
| **CausalPFN** | Balazadeh et al., *CausalPFN: Amortized Causal Effect Estimation via In-Context Learning*, arXiv:2506.07918 | [vdblm/CausalPFN](https://github.com/vdblm/CausalPFN) |
| **Do-PFN** | Robertson, Reuter et al., *Do-PFN: In-Context Learning for Causal Effect Estimation*, arXiv:2506.06039 | [jr2021/Do-PFN](https://github.com/jr2021/Do-PFN) |
| **CausalFM** | Ma, Frauen, Javurek & Feuerriegel, *Foundation Models for Causal Inference via Prior-Data Fitted Networks*, arXiv:2506.10914 (ICLR 2026) | [yccm/CausalFM-toolkit](https://github.com/yccm/CausalFM-toolkit) |

### Metalearners (from econml)
| Method | Description |
|---|---|
| **S-learner** | Single-model learner: trains one model on X + T |
| **T-learner** | Two-model learner: separate models for each treatment group |
| **X-learner** | Cross-fit learner: asymptotically efficient variant |
| **Debiased ML** | Neyman-orthogonal approach, robust to nuisance parameter estimation |
| **IPW** | Inverse Probability Weighting based on propensity scores |
| **DR** | Doubly Robust: combines outcome and propensity modeling |

## Quick Start

### 1. Install

```bash
# With uv (recommended):
uv sync

# Or with pip:
pip install -r requirements.txt
```

### 2. Interactive Demo (recommended first step)

```bash
jupyter notebook notebooks/01_interactive_model_demo.ipynb
```

Choose any model + dataset (synthetic or Lalonde) via widgets and run it end-to-end. See PEHE, ATE error, runtime.

### 3. Lalonde Benchmark

```bash
jupyter notebook notebooks/02_lalonde_benchmark.ipynb
```

Compare one foundation model against all six metalearners on the Lalonde real-world benchmark.

### 4. Setup & Explore Datasets

```bash
jupyter notebook notebooks/00_setup_and_data.ipynb
```

Generate and visualize the four synthetic datasets used across all models.

## Repository Structure

```
.
├── causal_bench/                           # Shared evaluation library
│   ├── __init__.py
│   ├── data_generators.py                  # 4 synthetic datasets (linear, nonlinear, IV, frontdoor)
│   ├── data_loader.py                      # Load Lalonde real-world benchmark
│   ├── metrics.py                          # PEHE, ATE error, bias, coverage, etc.
│   ├── wrap_causalpfn.py                   # CausalPFN wrapper
│   ├── wrap_dopfn.py                       # Do-PFN wrapper
│   ├── wrap_causalfm.py                    # CausalFM wrapper
│   └── wrap_metalearners.py                # S/T/X-learner, Debiased ML, IPW, DR wrappers
├── notebooks/
│   ├── 00_setup_and_data.ipynb             # Setup + synthetic dataset generation
│   ├── 01_interactive_model_demo.ipynb     # Choose model & dataset, run interactively
│   └── 02_lalonde_benchmark.ipynb          # Compare foundation model vs. metalearners on Lalonde
├── scripts/
│   └── build_new_notebooks.py              # Regenerate notebooks from template
├── requirements.txt                        # Dependencies (numpy, pandas, torch, econml, causalml, etc.)
├── pyproject.toml                          # uv configuration
├── CLAUDE.md                               # Development guide
└── README.md                               # This file
```

## Datasets

### Synthetic (from `causal_bench.data_generators`)

| Name | Identification | CATE | Notes |
|---|---|---|---|
| `linear_confounded` | Backdoor | Homogeneous (constant) | Textbook setting: linear responses, observed confounders |
| `nonlinear_heterogeneous` | Backdoor | Heterogeneous `τ(x)=sin(x₀)+0.5x₁` | Matches CausalPFN's README example; tests nonlinearity |
| `iv_binary` | Instrumental Variable | Homogeneous | Hidden confounder + binary instrument; tests IV robustness |
| `frontdoor` | Front-Door Adjustment | Homogeneous | Hidden confounder + mediator; tests mediation |

### Real-World

| Name | Source | Notes |
|---|---|---|
| `lalonde_nsw_psid` | NSW vs. PSID | Standard causal ML benchmark; no ground truth CATE |

## Metrics

All models evaluated on:
- **PEHE**: `√E[(τ̂ - τ)²]` — precision in estimating heterogeneous effects
- **ATE error**: `|ATE_hat - ATE_true|` — absolute error on average effect
- **ATE relative error**: `|ATE_hat - ATE_true| / |ATE_true|`
- **Bias**: `mean(τ̂ - τ)` — systematic over/under-estimation
- **Coverage @95%**: Fraction of true τ inside model's 95% confidence interval (when available)
- **Runtime**: Seconds (fit + predict on test set)

## Usage Examples

### Run one model on one dataset (Python)

```python
from causal_bench import get_dataset, CausalPFNWrapper, evaluate_cate
import numpy as np

# Load synthetic dataset
ds = get_dataset("nonlinear_heterogeneous", n=2000, seed=0)
train_idx, test_idx = ds.train_test_split(0.7, seed=0)

# Create and fit model (use "mps" on Apple Silicon, "cuda" on GPU, "cpu" otherwise)
model = CausalPFNWrapper(device="cpu")
model.fit(ds.X[train_idx], ds.T[train_idx], ds.Y[train_idx])

# Predict and evaluate
tau_hat, lower, upper = model.predict(ds.X[test_idx])
results = evaluate_cate(tau_hat, ds.tau[test_idx], lower=lower, upper=upper)
print(f"PEHE: {results['pehe']:.4f}")
```

### Compare models programmatically

```python
from causal_bench import (
    get_dataset, evaluate_cate,
    CausalPFNWrapper, SLearnerWrapper, TLearnerWrapper
)

ds = get_dataset("nonlinear_heterogeneous", n=2000, seed=0)
train_idx, test_idx = ds.train_test_split(0.7, seed=0)

models = [
    CausalPFNWrapper(),
    SLearnerWrapper(),
    TLearnerWrapper(),
]

for model_cls in models:
    tau_hat, lower, upper, ate_hat, runtime = model_cls().run(
        ds.X[train_idx], ds.T[train_idx], ds.Y[train_idx],
        ds.X[test_idx]
    )
    result = evaluate_cate(tau_hat, ds.tau[test_idx], ate_hat=ate_hat,
                          ate_true=ds.ate, lower=lower, upper=upper, runtime_s=runtime)
    print(f"{model_cls.name}: PEHE={result['pehe']:.4f}, runtime={runtime:.2f}s")
```

## On Google Colab

Each notebook includes an "Open in Colab" badge. Click it to run directly on Colab (all installs happen automatically). Alternatively:

1. Open Colab: https://colab.research.google.com
2. File → Open notebook → GitHub
3. Paste this repo URL and select a notebook
4. Run all cells top-to-bottom

**Note**: Foundation models that require checkpoints (CausalFM) or external repos (Do-PFN) are installed on first use in the notebook.

## Important Notes

- **Synthetic datasets**: Ground-truth CATE available; full metrics (PEHE, bias, coverage) computed.
- **Lalonde**: Real-world data; no ground-truth CATE. Only observed ATE available for comparison.
- **Foundation models**: Fast, zero-shot (no per-dataset training). Do not re-train; just condition on data.
- **Metalearners**: Traditional ML methods trained from scratch on each dataset.
- **Missing dependencies**: Notebooks gracefully skip unavailable models (with warnings) rather than crashing.

## Citation

If you use this code in research, please cite the relevant papers:
- CausalPFN: Balazadeh et al., arXiv:2506.07918
- Do-PFN: Robertson et al., arXiv:2506.06039
- CausalFM: Ma et al., arXiv:2506.10914

## License

MIT
