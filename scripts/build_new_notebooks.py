"""
Builds the Lalonde benchmark notebook using nbformat.
Run: python3 scripts/build_new_notebooks.py

Generates:
- Lalonde_benchmark.ipynb (all 6 metalearners + all 3 foundation models on Lalonde)
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
# Lalonde Benchmark (All Models)
# ============================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    md(f"""# Lalonde Benchmark: Foundation Models vs. Metalearners

{colab_badge('notebooks/Lalonde_benchmark.ipynb')}

**Compare all foundation models against all traditional metalearners on the Lalonde dataset.**

This notebook runs:
- 3 foundation models (CausalPFN, Do-PFN, CausalFM)
- 6 metalearners (S-learner, T-learner, X-learner, Debiased ML, IPW, DR)

on the Lalonde real-world causal inference benchmark and produces a comparison table. Each
model runs independently and failures don't block the rest — unavailable or erroring
models are reported and skipped."""),

    md("## 1. Setup"),

    code("""import os, sys, subprocess

IN_COLAB = "google.colab" in sys.modules

# ── FOR COLAB ONLY: set your GitHub token if the repo is private ──────────────
# Create one at: github.com/settings/tokens  (scope: repo → read)
# Leave as "" if the repo is public.
GITHUB_TOKEN = ""
# ──────────────────────────────────────────────────────────────────────────────

REPO_SLUG = "chris-L6/causalfm-survey"
REPO_DIR  = "causalfm-survey"

if IN_COLAB:
    if not os.path.exists(REPO_DIR):
        if GITHUB_TOKEN:
            clone_url = f"https://{GITHUB_TOKEN}@github.com/{REPO_SLUG}.git"
        else:
            clone_url = f"https://github.com/{REPO_SLUG}.git"
        result = subprocess.run(["git", "clone", clone_url], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "git clone failed — the repo is likely private.\\n"
                "Fix: set GITHUB_TOKEN above (github.com/settings/tokens, scope: repo→read).\\n"
                f"Error: {result.stderr.strip()}"
            )
    sys.path.insert(0, REPO_DIR)
else:
    sys.path.insert(0, os.path.abspath(".."))

import causal_bench
print("causal_bench imported from:", causal_bench.__file__)"""),

    md("""### One-time environment check — needed for Do-PFN

Do-PFN's model code depends on an internal PyTorch name removed in
`torch>=2.10`. This must run *before* `torch` is imported anywhere else in
this notebook (see next cell).

- **On Colab**: installs `torch<2.10` and restarts the runtime automatically
  — re-run this cell once after it reconnects, then continue from the top.
- **Locally (this repo's `uv` venv)**: `pip` isn't available inside the
  notebook, so this only detects the problem. Fix in a terminal:
  `uv pip install "torch<2.10"`, then restart the kernel.
- Not planning to run Do-PFN? Skip — CausalPFN and CausalFM work fine on any
  recent torch."""),

    code("""def _torch_pre_2_10():
    try:
        import torch
    except ImportError:
        return True  # not installed yet -- nothing to fix here
    major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    return (major, minor) < (2, 10)

if _torch_pre_2_10():
    print("OK -- torch version is compatible with Do-PFN (or not installed yet).")
elif IN_COLAB:
    print("torch >= 2.10 detected -- installing torch<2.10 and restarting the runtime...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch<2.10"], check=True)
    print("Restarting now. After it reconnects, re-run THIS cell, then continue from the top.")
    os.kill(os.getpid(), 9)  # Colab reconnects automatically with a fresh process
else:
    import torch
    print(f"torch {torch.__version__} is >= 2.10 -- Do-PFN will fail to import.")
    print('Fix, in a terminal (not this notebook -- local uv venv has no pip):')
    print('    uv pip install "torch<2.10"')
    print("then restart this notebook's kernel and re-run from the top.")"""),

    code("""# "pandas<2.4" pin: econml has no pandas upper bound, so pip's resolver
# otherwise grabs the newest pandas (3.x) -- which conflicts with Colab's
# preinstalled google-colab/cudf-cu12/dask-cudf-cu12 (all require pandas<2.4).
%pip install -q econml causalpfn "pandas<2.4"
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')

import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")"""),

    md("""### Do-PFN and CausalFM setup

Neither is on PyPI, so clone if missing and install just the extra deps each
actually needs — **not** their bundled `requirements.txt` files, which are
frozen dev/CUDA snapshots that fail to install as-is (`catboost==1.1.1` has
no wheel for recent Python; CausalFM's snapshot has Linux/CUDA-only pins).
CausalFM additionally needs its pretrained checkpoint, which ships inside the
cloned repo at `checkpoints/checkpoints_standard/best_model.pth`."""),

    code("""DOPFN_DIR = "Do-PFN"
CAUSALFM_DIR = "CausalFM-toolkit"

if not os.path.exists(DOPFN_DIR):
    print(f"Cloning Do-PFN...")
    subprocess.run(["git", "clone", "https://github.com/jr2021/Do-PFN.git"], check=True)
sys.path.insert(0, os.path.abspath(DOPFN_DIR))

if not os.path.exists(CAUSALFM_DIR):
    print(f"Cloning CausalFM-toolkit...")
    subprocess.run(["git", "clone", "https://github.com/yccm/CausalFM-toolkit.git"], check=True)
sys.path.insert(0, os.path.abspath(CAUSALFM_DIR))

if IN_COLAB:
    get_ipython().system('pip install -q networkx tqdm einops "tabpfn==2.0.9" tensorboard')
# Locally: uv pip install networkx tqdm einops "tabpfn==2.0.9" tensorboard"""),

    md("## 2. Load Lalonde Dataset"),

    code("""from causal_bench import load_lalonde, evaluate_cate

print("Loading Lalonde dataset...")
ds = load_lalonde()
print(f"  n={len(ds.Y)}, X.shape={ds.X.shape}, observed ATE={ds.ate:.3f}")

train_idx, test_idx = ds.train_test_split(0.7, seed=0)
X_train, X_test = ds.X[train_idx], ds.X[test_idx]
T_train, Y_train = ds.T[train_idx], ds.Y[train_idx]

print(f"  train: n={len(train_idx)}, test: n={len(test_idx)}")"""),

    md("## 3. Foundation Model Availability"),

    code("""from causal_bench import CausalPFNWrapper, DoPFNWrapper, CausalFMWrapper

FOUNDATION_MODELS = {
    "CausalPFN": CausalPFNWrapper,
    "Do-PFN":    DoPFNWrapper,
    "CausalFM":  CausalFMWrapper,
}

print("Foundation model availability:")
for name, cls in FOUNDATION_MODELS.items():
    print(f"  {'✓' if cls.is_available() else '✗'}  {name}")
print(f"\\ndevice: {device}")"""),

    md("## 4. Run All Models"),

    code("""from causal_bench import (
    SLearnerWrapper, TLearnerWrapper, XLearnerWrapper,
    DebiasedMLWrapper, IPWWrapper, DRWrapper,
)

METALEARNERS = {
    "S-learner":          SLearnerWrapper,
    "T-learner":          TLearnerWrapper,
    "X-learner":          XLearnerWrapper,
    "Debiased ML":        DebiasedMLWrapper,
    "IPW":                IPWWrapper,
    "DR (Doubly Robust)": DRWrapper,
}

results = []

# ── Run all metalearners ──────────────────────────────────────────────────────
print("=" * 70)
print("METALEARNERS")
print("=" * 70)
for name, model_cls in METALEARNERS.items():
    if not model_cls.is_available():
        print(f"  {name:25s}: SKIPPED (not installed)")
        continue
    try:
        t0 = time.time()
        model = model_cls()
        model.fit(X_train, T_train, Y_train)
        tau_hat, _, _ = model.predict(X_test)
        runtime = time.time() - t0
        ate_hat = float(np.mean(tau_hat))
        results.append({
            "model": name,
            "ate_hat": ate_hat, "ate_true": ds.ate,
            "ate_abs_error": abs(ate_hat - ds.ate),
            "ate_rel_error": abs(ate_hat - ds.ate) / (abs(ds.ate) + 1e-8),
            "runtime_s": runtime,
        })
        print(f"  {name:25s}: ATE_error={abs(ate_hat-ds.ate):.1f}  runtime={runtime:.2f}s")
    except Exception as e:
        print(f"  {name:25s}: ERROR: {e}")

# ── Run all foundation models ─────────────────────────────────────────────────
print("\\n" + "=" * 70)
print("FOUNDATION MODELS")
print("=" * 70)
for fm_name, model_cls in FOUNDATION_MODELS.items():
    if not model_cls.is_available():
        print(f"  {fm_name:25s}: SKIPPED (not available in this environment)")
        continue
    try:
        t0 = time.time()

        if fm_name == "CausalFM":
            checkpoint_path = "CausalFM-toolkit/checkpoints/checkpoints_standard/best_model.pth"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            model = model_cls(checkpoint_path=checkpoint_path, device=device)
        else:
            model = model_cls(device=device)

        model.fit(X_train, T_train, Y_train)
        tau_hat, _, _ = model.predict(X_test)
        runtime = time.time() - t0
        ate_hat = float(np.mean(tau_hat))

        results.append({
            "model": fm_name + " (Foundation)",
            "ate_hat": ate_hat, "ate_true": ds.ate,
            "ate_abs_error": abs(ate_hat - ds.ate),
            "ate_rel_error": abs(ate_hat - ds.ate) / (abs(ds.ate) + 1e-8),
            "runtime_s": runtime,
        })
        print(f"  {fm_name:25s}: ATE_error={abs(ate_hat-ds.ate):.1f}  runtime={runtime:.2f}s")
    except Exception as e:
        print(f"  {fm_name:25s}: ERROR: {e}")

print("\\n" + "=" * 70)"""),

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

save(nb, "Lalonde_benchmark.ipynb")

print("\nAll notebooks built successfully!")
print(f"Generated: {os.path.join(OUT_DIR, 'Lalonde_benchmark.ipynb')}")
