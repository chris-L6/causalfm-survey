"""
Builds the refactored benchmark notebooks using nbformat.
Run: python3 scripts/build_new_notebooks.py

Generates:
- 01_interactive_model_demo.ipynb (model + dataset selection with widgets)
- 02_lalonde_benchmark.ipynb (all models on Lalonde dataset)
"""
import nbformat as nbf
import os

OUT_DIR = "notebooks"
os.makedirs(OUT_DIR, exist_ok=True)

REPO_SLUG = "chris-L6/causalfm-survey"


def colab_badge(notebook_path):
    url = f"https://colab.research.google.com/github/{REPO_SLUG}/blob/main/{notebook_path}"
    return f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})'


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def save(nb, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


# ============================================================================
# 00 - Setup & Shared Data (KEEP EXISTING)
# ============================================================================
# (This notebook already exists and doesn't need regeneration)

# ============================================================================
# 01 - Interactive Model Demo
# ============================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    md(f"""# 01 — Interactive Model Demo

{colab_badge('notebooks/01_interactive_model_demo.ipynb')}

**Choose a model and dataset, then run it!**

This notebook lets you interactively select from:
- **Models**: Foundation models (CausalPFN, Do-PFN, CausalFM) or traditional metalearners (S/T/X-learner, Debiased ML, IPW, DR)
- **Datasets**: Synthetic (linear, nonlinear, IV, frontdoor) or real-world (Lalonde)

All models follow a unified interface, so swapping between them is seamless."""),

    md("""## 1. Environment Setup

If running on **Colab**, this clones the repo. If **local**, it imports from parent directory."""),

    code("""import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/chris-L6/causalfm-survey.git"
REPO_DIR = "causalfm-survey"

if IN_COLAB:
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", REPO_URL], check=True)
    sys.path.insert(0, REPO_DIR)
else:
    sys.path.insert(0, os.path.abspath(".."))

import causal_bench
print("causal_bench imported from:", causal_bench.__file__)"""),

    code("""!pip install -q econml causalml ipywidgets
import ipywidgets as widgets
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")"""),

    md("## 2. Dataset Loader"),

    code("""from causal_bench import get_dataset, load_lalonde, list_available_datasets

# Load all datasets
DATASETS = {}
print("Loading synthetic datasets...")
for name in ["linear_confounded", "nonlinear_heterogeneous", "iv_binary", "frontdoor"]:
    DATASETS[name] = get_dataset(name, n=2000, seed=0)

print("Loading Lalonde dataset...")
try:
    DATASETS["lalonde_nsw_psid"] = load_lalonde()
    print("  ✓ Lalonde loaded")
except Exception as e:
    print(f"  ✗ Lalonde unavailable: {e}")

print(f"\\nAvailable datasets: {list(DATASETS.keys())}")
for name, ds in DATASETS.items():
    print(f"  {name}: n={len(ds.Y)}, X.shape={ds.X.shape}, ATE={ds.ate:.3f}")"""),

    md("## 3. Model Selection Widgets"),

    code("""from causal_bench import (
    CausalPFNWrapper, DoPFNWrapper, CausalFMWrapper,
    SLearnerWrapper, TLearnerWrapper, XLearnerWrapper,
    DebiasedMLWrapper, IPWWrapper, DRWrapper
)

MODELS = {
    "CausalPFN": CausalPFNWrapper,
    "Do-PFN": DoPFNWrapper,
    "CausalFM": CausalFMWrapper,
    "S-learner": SLearnerWrapper,
    "T-learner": TLearnerWrapper,
    "X-learner": XLearnerWrapper,
    "Debiased ML": DebiasedMLWrapper,
    "IPW": IPWWrapper,
    "DR (Doubly Robust)": DRWrapper,
}

# Check availability
available_models = {name: cls for name, cls in MODELS.items() if cls.is_available()}
print(f"Available models ({len(available_models)}/{len(MODELS)}):")
for name in available_models:
    print(f"  ✓ {name}")
for name in set(MODELS.keys()) - set(available_models.keys()):
    print(f"  ✗ {name}")

# Widgets for selection
model_dropdown = widgets.Dropdown(options=available_models.keys(), description="Model:")
dataset_dropdown = widgets.Dropdown(options=DATASETS.keys(), description="Dataset:")

display(widgets.VBox([model_dropdown, dataset_dropdown]))"""),

    md("## 4. Run Selected Model on Selected Dataset"),

    code("""from causal_bench import evaluate_cate

def run_demo():
    model_name = model_dropdown.value
    dataset_name = dataset_dropdown.value

    print(f"\\n{'='*60}")
    print(f"Running {model_name} on {dataset_name}")
    print(f"{'='*60}\\n")

    ds = DATASETS[dataset_name]
    train_idx, test_idx = ds.train_test_split(0.7, seed=0)

    X_train, X_test = ds.X[train_idx], ds.X[test_idx]
    T_train, Y_train = ds.T[train_idx], ds.Y[train_idx]

    # For synthetic datasets, we have ground truth tau
    if hasattr(ds, 'tau'):
        tau_test = ds.tau[test_idx]
    else:
        tau_test = None

    try:
        t0 = time.time()
        model_cls = MODELS[model_name]

        # Special handling for CausalFM (needs checkpoint path)
        if model_name == "CausalFM":
            checkpoint_path = "CausalFM-toolkit/checkpoints/best_model.pth"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"CausalFM checkpoint not found at {checkpoint_path}")
            model = model_cls(checkpoint_path=checkpoint_path, device=device)
        else:
            model = model_cls(device=device) if "Wrapper" in model_cls.__name__ and "Foundation" not in model_name else model_cls()

        model.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = model.predict(X_test)
        runtime = time.time() - t0

        ate_hat = float(np.mean(tau_hat))

        if tau_test is not None:
            result = evaluate_cate(tau_hat, tau_test, ate_hat=ate_hat, ate_true=ds.ate,
                                   lower=lower, upper=upper, runtime_s=runtime)
            print("Results:")
            for key, val in result.items():
                if val is not None:
                    print(f"  {key:20s}: {val:10.4f}")
        else:
            print(f"  ATE_hat:          {ate_hat:.4f}")
            print(f"  ATE_true (observed): {ds.ate:.4f}")
            print(f"  Runtime:          {runtime:.2f}s")
            print("  (Ground truth CATE unavailable for real data)")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# Create run button
run_button = widgets.Button(description="▶ Run Demo")
output = widgets.Output()

def on_click(b):
    with output:
        output.clear_output(wait=False)
        run_demo()

run_button.on_click(on_click)

display(widgets.VBox([run_button, output]))"""),

    md("""## Notes

- **Foundation models** (CausalPFN, Do-PFN, CausalFM) use in-context learning: they don't re-train on your data, just condition on it.
- **Metalearners** (S/T/X, Debiased ML, etc.) are traditional ML methods trained separately for control and treatment.
- **Synthetic datasets** have ground-truth CATE (`tau`), so full metrics (PEHE, bias, coverage) are computed.
- **Lalonde** is real-world data: no ground truth CATE, only observed ATE is available.
- **Missing models** (shown with ✗) require additional installs; check the notebook setup cells."""),
]

save(nb, "01_interactive_model_demo.ipynb")


# ============================================================================
# 02 - Lalonde Benchmark (All Models)
# ============================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    md(f"""# 02 — Lalonde Benchmark: Foundation Models vs. Metalearners

{colab_badge('notebooks/02_lalonde_benchmark.ipynb')}

**Compare one foundation model against traditional metalearners on the Lalonde dataset.**

This notebook runs:
- 1 foundation model (choose: CausalPFN / Do-PFN / CausalFM)
- 6 metalearners (S-learner, T-learner, X-learner, Debiased ML, IPW, DR)

on the Lalonde real-world causal inference benchmark and produces a comparison table."""),

    md("## 1. Setup"),

    code("""import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/chris-L6/causalfm-survey.git"
REPO_DIR = "causalfm-survey"

if IN_COLAB:
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", REPO_URL], check=True)
    sys.path.insert(0, REPO_DIR)
else:
    sys.path.insert(0, os.path.abspath(".."))

import causal_bench
print("causal_bench imported from:", causal_bench.__file__)"""),

    code("""!pip install -q econml causalml torch
import ipywidgets as widgets
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")"""),

    md("## 2. Load Lalonde Dataset"),

    code("""from causal_bench import load_lalonde, evaluate_cate

print("Loading Lalonde dataset...")
ds = load_lalonde()
print(f"  n={len(ds.Y)}, X.shape={ds.X.shape}, observed ATE={ds.ate:.3f}")

train_idx, test_idx = ds.train_test_split(0.7, seed=0)
X_train, X_test = ds.X[train_idx], ds.X[test_idx]
T_train, Y_train = ds.T[train_idx], ds.Y[train_idx]

print(f"  train: n={len(train_idx)}, test: n={len(test_idx)}")"""),

    md("## 3. Select Foundation Model"),

    code("""from causal_bench import CausalPFNWrapper, DoPFNWrapper, CausalFMWrapper

FOUNDATION_MODELS = {
    "CausalPFN": CausalPFNWrapper,
    "Do-PFN": DoPFNWrapper,
    "CausalFM": CausalFMWrapper,
}

available_foundation = {name: cls for name, cls in FOUNDATION_MODELS.items() if cls.is_available()}
print(f"Available foundation models: {list(available_foundation.keys())}")

foundation_dropdown = widgets.Dropdown(options=available_foundation.keys(), description="Model:")
display(foundation_dropdown)"""),

    md("## 4. Run All Models"),

    code("""from causal_bench import (
    SLearnerWrapper, TLearnerWrapper, XLearnerWrapper,
    DebiasedMLWrapper, IPWWrapper, DRWrapper
)

METALEARNERS = {
    "S-learner": SLearnerWrapper,
    "T-learner": TLearnerWrapper,
    "X-learner": XLearnerWrapper,
    "Debiased ML": DebiasedMLWrapper,
    "IPW": IPWWrapper,
    "DR (Doubly Robust)": DRWrapper,
}

results = []

# Run metalearners
print("="*70)
print("METALEARNERS")
print("="*70)
for name, model_cls in METALEARNERS.items():
    if not model_cls.is_available():
        print(f"  {name:25s}: SKIPPED (not available)")
        continue

    try:
        t0 = time.time()
        model = model_cls()
        model.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = model.predict(X_test)
        runtime = time.time() - t0
        ate_hat = float(np.mean(tau_hat))

        result = {
            "model": name,
            "ate_hat": ate_hat,
            "ate_true": ds.ate,
            "ate_abs_error": abs(ate_hat - ds.ate),
            "ate_rel_error": abs(ate_hat - ds.ate) / (abs(ds.ate) + 1e-8),
            "runtime_s": runtime,
        }
        results.append(result)
        print(f"  {name:25s}: ATE_error={result['ate_abs_error']:.4f}, runtime={runtime:.2f}s")
    except Exception as e:
        print(f"  {name:25s}: ERROR: {e}")

# Run foundation model
print("\\n" + "="*70)
print("FOUNDATION MODEL")
print("="*70)
fm_name = foundation_dropdown.value
print(f"  {fm_name}...")

try:
    t0 = time.time()
    model_cls = FOUNDATION_MODELS[fm_name]

    if fm_name == "CausalFM":
        checkpoint_path = "CausalFM-toolkit/checkpoints/best_model.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"CausalFM checkpoint not found at {checkpoint_path}")
        model = model_cls(checkpoint_path=checkpoint_path, device=device)
    else:
        model = model_cls(device=device)

    model.fit(X_train, T_train, Y_train)
    tau_hat, lower, upper = model.predict(X_test)
    runtime = time.time() - t0
    ate_hat = float(np.mean(tau_hat))

    result = {
        "model": fm_name + " (Foundation)",
        "ate_hat": ate_hat,
        "ate_true": ds.ate,
        "ate_abs_error": abs(ate_hat - ds.ate),
        "ate_rel_error": abs(ate_hat - ds.ate) / (abs(ds.ate) + 1e-8),
        "runtime_s": runtime,
    }
    results.append(result)
    print(f"  {fm_name:25s}: ATE_error={result['ate_abs_error']:.4f}, runtime={runtime:.2f}s")
except Exception as e:
    print(f"  {fm_name:25s}: ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\\n" + "="*70)"""),

    md("## 5. Results Table"),

    code("""df = pd.DataFrame(results)
df_sorted = df.sort_values("ate_abs_error")

print("\\nResults (sorted by ATE absolute error):")
print(df_sorted[["model", "ate_hat", "ate_true", "ate_abs_error", "ate_rel_error", "runtime_s"]].to_string(index=False))

df_sorted.to_csv("lalonde_benchmark.csv", index=False)
print("\\nSaved to lalonde_benchmark.csv")"""),

    md("## 6. Visualization"),

    code("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ATE error comparison
ax = axes[0]
df_plot = df_sorted.copy()
colors = ['#1f77b4' if 'Foundation' in name else '#ff7f0e' for name in df_plot['model']]
ax.barh(range(len(df_plot)), df_plot['ate_abs_error'], color=colors)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['model'])
ax.set_xlabel('Absolute ATE Error')
ax.set_title('ATE Error: Foundation vs. Metalearners')
ax.axvline(ds.ate, color='red', linestyle='--', alpha=0.5, label='True ATE')

# Runtime comparison
ax = axes[1]
ax.barh(range(len(df_plot)), df_plot['runtime_s'], color=colors)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['model'])
ax.set_xlabel('Runtime (seconds)')
ax.set_title('Runtime: Fit + Predict on Test Set')

plt.legend(['Foundation', 'Metalearner'], loc='lower right')
plt.tight_layout()
plt.savefig("lalonde_benchmark.png", dpi=150)
plt.show()
print("Saved to lalonde_benchmark.png")"""),

    md("""## Interpretation

**ATE Error** (lower is better):
- Measures how well each model estimates the average treatment effect
- Foundation models learn from large training priors; metalearners fit to this specific data

**Runtime** (lower is better):
- Foundation models: fast (forward pass only, no retraining)
- Metalearners: slower (train separate models for each group)

**Real data**: No ground-truth CATE available, only observed ATE (simple difference in means).
Foundation models may have learned causal relationships from their training priors that help here."""),
]

save(nb, "02_lalonde_benchmark.ipynb")

print("\\nAll notebooks built successfully!")
print(f"Generated: {os.path.join(OUT_DIR, '01_interactive_model_demo.ipynb')}")
print(f"Generated: {os.path.join(OUT_DIR, '02_lalonde_benchmark.ipynb')}")
