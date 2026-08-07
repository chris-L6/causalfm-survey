# Quick Reference Cheat Sheet

Every model below shares one interface via `causal_bench`:

```python
model.fit(X_train, T_train, Y_train)        # X: (n,d) float32, T: (n,) binary, Y: (n,) float32
tau_hat, lower, upper = model.predict(X_test)
# or, in one call:
tau_hat, lower, upper, ate_hat, runtime = model.run(X_train, T_train, Y_train, X_test)
```

`lower`/`upper` are `None` for any model that doesn't expose intervals (all of
them, currently). All wrappers standardize inputs internally where it
matters — no need to scale data yourself before calling `fit`/`predict`.

## Model Overview

| Model | Install | Constructor | Notes |
|---|---|---|---|
| **CausalPFN** | `pip install causalpfn` | `CausalPFNWrapper(device=...)` | Skips itself (`is_available()` → `False`) on Apple Silicon macOS — segfaults there |
| **Do-PFN** | `git clone jr2021/Do-PFN` | `DoPFNWrapper(repo_dir="Do-PFN", device=...)` | Needs `torch<2.10` |
| **CausalFM** | `git clone yccm/CausalFM-toolkit` | `CausalFMWrapper(checkpoint_path=..., device=...)` — **`checkpoint_path` is required** | Checkpoint at `checkpoints/checkpoints_standard/best_model.pth` inside the clone |
| **S/T/X-learner, Debiased ML** | `uv pip install econml` | `SLearnerWrapper()` / `TLearnerWrapper()` / `XLearnerWrapper()` / `DebiasedMLWrapper()` | Default base model: `RandomForestRegressor` |
| **IPW, DR** | `uv pip install econml` (uses `sklearn` directly) | `IPWWrapper()` / `DRWrapper()` | |

For exact, verified raw-API usage of each foundation model **without** the
wrapper (what to import, gotchas, checkpoint paths), see
`notebooks/Foundation_models_sandbox.ipynb` — it calls each library's native
API directly and is the source of truth this cheat sheet defers to, rather
than duplicating code here that can drift out of date.

## Loop Over All Models

```python
from causal_bench import (
    CausalPFNWrapper, DoPFNWrapper, CausalFMWrapper,
    SLearnerWrapper, TLearnerWrapper, XLearnerWrapper,
    DebiasedMLWrapper, IPWWrapper, DRWrapper, evaluate_cate,
)

CKPT = "CausalFM-toolkit/checkpoints/checkpoints_standard/best_model.pth"
models = [
    CausalPFNWrapper(), DoPFNWrapper(), CausalFMWrapper(checkpoint_path=CKPT),
    SLearnerWrapper(), TLearnerWrapper(), XLearnerWrapper(),
    DebiasedMLWrapper(), IPWWrapper(), DRWrapper(),
]

results = []
for model in models:
    if not model.is_available():
        print(f"Skipping {model.name} (not available)")
        continue
    tau_hat, lower, upper, ate_hat, runtime = model.run(X_train, T_train, Y_train, X_test)
    results.append({"model": model.name, **evaluate_cate(tau_hat, tau_true, runtime_s=runtime)})
```

## On Synthetic Data (ground-truth CATE available)

```python
from causal_bench import get_dataset, evaluate_cate, SLearnerWrapper

ds = get_dataset("nonlinear_heterogeneous", n=2000, seed=0)
train_idx, test_idx = ds.train_test_split(0.7, seed=0)

model = SLearnerWrapper()
tau_hat, _, _ = model.fit(ds.X[train_idx], ds.T[train_idx], ds.Y[train_idx]).predict(ds.X[test_idx])
metrics = evaluate_cate(tau_hat, ds.tau[test_idx])
```

## On Lalonde (real data — see `docs/LALONDE_DATASET.md`)

```python
from causal_bench import load_lalonde, SLearnerWrapper, TLearnerWrapper

ds = load_lalonde("nsw_psid_trimmed")  # or "nsw_psid" for the untrimmed, harder pairing
print(f"true ATE: {ds.ate:.2f}  (naive confounded diff: {ds.ate_naive_observed:.2f})")
train_idx, test_idx = ds.train_test_split(0.7, seed=0)

for ModelClass in [SLearnerWrapper, TLearnerWrapper]:
    model = ModelClass()
    tau_hat, _, _ = model.fit(ds.X[train_idx], ds.T[train_idx], ds.Y[train_idx]).predict(ds.X[test_idx])
    print(f"{model.name} ATE: {tau_hat.mean():.2f}  (true: {ds.ate:.2f})")
```

## Key Insights

**Foundation models** (CausalPFN, Do-PFN, CausalFM): pretrained, zero-shot —
`fit()` doesn't train, just conditions on your data as context. Fast, but
sensitive to input scale (handled internally by the wrappers) and to
install/environment gotchas (see the model table above and CLAUDE.md).

**Metalearners** (S/T/X, Debiased ML, IPW, DR): train fresh on every call —
slower, but no external repo/checkpoint dependencies and well-understood
theory. DR/Debiased ML are the most robust to nuisance-model
misspecification *in principle* — but on data with poor covariate overlap
(e.g. Lalonde's NSW/PSID pairing), that doesn't guarantee they'll be closest
to the truth. See `docs/LALONDE_DATASET.md` for a worked example.
