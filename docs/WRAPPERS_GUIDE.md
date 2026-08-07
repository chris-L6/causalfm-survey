# Wrappers Guide

`causal_bench`'s `wrap_*.py` files adapt nine different libraries — each
with its own install method, constructor signature, and fit/predict
API — into one shared interface:

```python
model.fit(X_train, T_train, Y_train)
tau_hat, lower, upper = model.predict(X_test)
tau_hat, lower, upper, ate_hat, runtime = model.run(X_train, T_train, Y_train, X_test)
model.is_available()  # classmethod: is the underlying library/checkpoint usable right now?
```

Without wrappers, the same nine models require nine different calling
conventions — e.g. econml's metalearners take `.fit(Y, T, X=X)` (note the
order) and return effects via `.effect(X)`, not `.fit(X, T, Y)` /
`.predict(X)`; CausalFM's native call returns a dict (`result["cate"]`);
Do-PFN needs treatment concatenated into the feature matrix in a specific
column and expects `torch.Tensor`, not numpy. The wrapper is what lets
benchmark code loop over all nine identically instead of branching per model.

## Model-by-model notes

| Model | Underlying lib | Wrapper-specific gotcha |
|---|---|---|
| **CausalPFN** | `causalpfn` (PyPI) | `is_available()` returns `False` on Apple Silicon macOS — the library segfaults there, not a normal exception `fit()` could catch |
| **Do-PFN** | `jr2021/Do-PFN` (git clone) | Constructor takes `repo_dir` (default `"Do-PFN"`); needs `torch<2.10`; wrapper handles the treatment-in-column-0 requirement and the repo-relative checkpoint path internally |
| **CausalFM** | `yccm/CausalFM-toolkit` (git clone) | Constructor **requires** `checkpoint_path` (no default) — real path is `checkpoints/checkpoints_standard/best_model.pth` inside the clone, not `checkpoints/best_model.pth` |
| **S/T/X-learner, Debiased ML** | `econml.metalearners` / `econml.dml` | Wrapper reorders args to `.fit(X, T, Y)` and exposes `.predict(X)` in place of `.effect(X)`; default base model is `RandomForestRegressor`, override via the `model=` constructor arg |
| **IPW, DR** | `sklearn` directly (no dedicated econml class used) | Manual propensity + outcome-model implementation inside the wrapper |

For exact, verified example code calling each **foundation** model's native
API directly (no wrapper) — including every install/environment gotcha
found while getting them running — see
`notebooks/Foundation_models_sandbox.ipynb`. That notebook is the source of
truth for raw-API usage; this guide intentionally doesn't duplicate it; a
duplicated copy is exactly what went stale and wrong here before.

## Standardization (foundation models only)

All three foundation-model wrappers fit a `StandardScaler` on `X` (and `Y`)
inside `fit()` and inverse-transform predictions back to the original
scale — the underlying models are pretrained on normalized synthetic data,
so raw real-world scales (e.g. Lalonde's dollar-denominated features) put
them out of distribution. See the "Foundation-model wrappers standardize..."
note in `CLAUDE.md` for the full explanation. This is invisible to callers:
`fit`/`predict` still take/return values in your original units.

## With wrapper vs. without

Use the wrapper (via `causal_bench`) for: comparing/looping over multiple
models, benchmarking, notebooks. Use each library's native API directly
(see the sandbox notebook) for: deep-diving one model, using
library-specific features the common interface doesn't expose (e.g.
CausalPFN's calibrated quantiles), or copy-pasting a single model into
another project.
