# Wrappers Guide: Understanding the Wrapper Pattern & Model APIs

This document explains:
1. What each model needs (data format, dependencies, API)
2. How to use WITHOUT wrappers (raw API)
3. How to use WITH wrappers (unified interface)
4. Why wrappers are useful
5. Overall architecture summary

---

## Table of Contents

- [Foundation Models](#foundation-models)
  - [CausalPFN](#causalpfn)
  - [Do-PFN](#do-pfn)
  - [CausalFM](#causalfm)
- [Traditional Metalearners](#traditional-metalearners)
  - [S-learner](#s-learner)
  - [T-learner](#t-learner)
  - [X-learner](#x-learner)
  - [Debiased ML](#debiased-ml)
  - [IPW](#ipw)
  - [DR (Doubly Robust)](#dr-doubly-robust)
- [Comparison Table](#comparison-table)
- [Overall Summary](#overall-summary)

---

## Foundation Models

### CausalPFN

**What it is:** A transformer pre-trained on synthetic SCMs. Does in-context learning: conditions on (X, T, Y) to estimate CATE.

#### WITHOUT Wrapper (Raw API)

```python
import torch
import numpy as np
from causalpfn import CATEEstimator, ATEEstimator

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create instances
cate_estimator = CATEEstimator(device=device, verbose=False)
ate_estimator = ATEEstimator(device=device, verbose=False)

# Fit (stores training data internally)
X_train = np.array(..., dtype=np.float32)  # (n, d)
T_train = np.array(..., dtype=np.float32)  # (n,)
Y_train = np.array(..., dtype=np.float32)  # (n,)

cate_estimator.fit(X_train, T_train, Y_train)
ate_estimator.fit(X_train, T_train, Y_train)

# Predict CATE on test data
X_test = np.array(..., dtype=np.float32)  # (m, d)
tau_hat = cate_estimator.estimate_cate(X_test)  # shape (m,)

# Estimate ATE
ate_hat = ate_estimator.estimate_ate()  # scalar

# Data requirements
# - X: float32, any shape (n, d)
# - T: float32, binary {0, 1}
# - Y: float32, continuous
```

#### WITH Wrapper (Unified Interface)

```python
from causal_bench import CausalPFNWrapper

model = CausalPFNWrapper(device="cuda", verbose=False)

# Same data
model.fit(X_train, T_train, Y_train)

# Same predict interface (returns tuple)
tau_hat, lower_95, upper_95 = model.predict(X_test)

# Convenience method
tau_hat, lower, upper, ate_hat, runtime = model.run(
    X_train, T_train, Y_train, X_test
)
```

**Key Differences:**
- Without wrapper: separate estimators for CATE and ATE
- With wrapper: unified `.predict()` returns `(tau_hat, lower, upper)`
- Without wrapper: `.estimate_cate()` vs with wrapper: `.predict()`
- Wrapper adds `.run()` for timing and convenience

**Data Requirements:**
- X: float32, (n, d) — any features
- T: float32, (n,) — binary {0, 1}
- Y: float32, (n,) — outcome

---

### Do-PFN

**What it is:** Transformer pre-trained to predict conditional interventional distributions p(y|do(t), x). Estimates CATE as E[y|do(t=1),x] - E[y|do(t=0),x].

#### WITHOUT Wrapper (Raw API)

```python
import numpy as np
from dopfn import DoPFNRegressor  # from cloned repo

# Key insight: Do-PFN expects [X, T] concatenated as design matrix
X_train = np.array(..., dtype=np.float32)  # (n, d_x)
T_train = np.array(..., dtype=np.float32)  # (n,) or (n, 1)
Y_train = np.array(..., dtype=np.float32)  # (n,)

# Concatenate T as last column
X_train_full = np.concatenate(
    [X_train, T_train.reshape(-1, 1)], 
    axis=1
)  # shape (n, d_x + 1)

# Create and fit
dopfn = DoPFNRegressor()
dopfn.fit(X_train_full, Y_train)

# To estimate CATE, predict twice (do-calculus):
# 1. Set T=1 for all samples
X_test_t1 = np.concatenate(
    [X_test, np.ones((len(X_test), 1), dtype=np.float32)],
    axis=1
)
y_pred_t1 = dopfn.predict_full(X_test_t1)

# 2. Set T=0 for all samples
X_test_t0 = np.concatenate(
    [X_test, np.zeros((len(X_test), 1), dtype=np.float32)],
    axis=1
)
y_pred_t0 = dopfn.predict_full(X_test_t0)

# CATE = difference
tau_hat = y_pred_t1 - y_pred_t0  # shape (m,)

# Note: predict_full() may return dict/distribution in some versions
# Need to extract mean if so
```

#### WITH Wrapper (Unified Interface)

```python
from causal_bench import DoPFNWrapper

model = DoPFNWrapper(device="cpu")

# Wrapper handles concatenation internally
model.fit(X_train, T_train, Y_train)

# Wrapper handles T=0/T=1 predictions and difference
tau_hat, lower_95, upper_95 = model.predict(X_test)

# Convenience
tau_hat, lower, upper, ate_hat, runtime = model.run(
    X_train, T_train, Y_train, X_test
)
```

**Key Differences:**
- Without wrapper: must manually concatenate [X, T]
- Without wrapper: must call model twice (T=1, T=0) and diff
- Without wrapper: `.predict_full()` vs with wrapper: `.predict()`
- Wrapper hides the do-calculus details

**Data Requirements:**
- X: float32, (n, d_x) — covariates (NOT including T)
- T: float32, (n,) — binary {0, 1}
- Y: float32, (n,) — outcome

---

### CausalFM

**What it is:** Amortized model trained on synthetic SCMs. Zero-shot: doesn't re-train, just conditions on (X, T, Y) to estimate CATE.

#### WITHOUT Wrapper (Raw API)

```python
import torch
import numpy as np
from causalfm.models import StandardCATEModel

# Load pre-trained model
checkpoint_path = "CausalFM-toolkit/checkpoints/best_model.pth"
model = StandardCATEModel.from_pretrained(checkpoint_path)

# Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Prepare data
X_train = np.array(..., dtype=np.float32)  # (n, d)
T_train = np.array(..., dtype=np.float32)  # (n,)
Y_train = np.array(..., dtype=np.float32)  # (n,)
X_test = np.array(..., dtype=np.float32)   # (m, d)

# Key: pass training data AND test points together
# (amortized: context + query)
result = model.estimate_cate(X_train, T_train, Y_train, X_test)

# Result is a dict
tau_hat = result["cate"]  # shape (m,)

# Optional: intervals (if model provides them)
if "cate_lower" in result and "cate_upper" in result:
    lower = result["cate_lower"]
    upper = result["cate_upper"]
elif "ci_lower" in result and "ci_upper" in result:
    lower = result["ci_lower"]
    upper = result["ci_upper"]
else:
    lower = upper = None

ate_hat = np.mean(tau_hat)
```

#### WITH Wrapper (Unified Interface)

```python
from causal_bench import CausalFMWrapper

model = CausalFMWrapper(
    checkpoint_path="CausalFM-toolkit/checkpoints/best_model.pth",
    device="cuda"
)

# fit() just stores the data and loads checkpoint
model.fit(X_train, T_train, Y_train)

# predict() calls estimate_cate internally
tau_hat, lower_95, upper_95 = model.predict(X_test)

# Convenience
tau_hat, lower, upper, ate_hat, runtime = model.run(
    X_train, T_train, Y_train, X_test
)
```

**Key Differences:**
- Without wrapper: must handle dict result and extract keys
- Without wrapper: must handle multiple possible interval key names
- Without wrapper: pass (X_train, T_train, Y_train, X_test) all at once
- Wrapper normalizes to `.predict()` interface

**Data Requirements:**
- X: float32, (n, d) — any features
- T: float32, (n,) — binary {0, 1}
- Y: float32, (n,) — outcome
- Checkpoint: must exist at specified path

---

## Traditional Metalearners

### S-learner

**What it is:** Single-model learner. Trains ONE model on features X ⊕ T (stacked), predicting Y. CATE = predictions(X, T=1) - predictions(X, T=0).

#### WITHOUT Wrapper (Raw API)

```python
import numpy as np
from econml.metalearners import SLearner
from sklearn.ensemble import RandomForestRegressor

# Data
X_train = np.array(..., dtype=np.float32)  # (n, d)
T_train = np.array(..., dtype=np.float32)  # (n,)
Y_train = np.array(..., dtype=np.float32)  # (n,)
X_test = np.array(..., dtype=np.float32)   # (m, d)

# Create base model
base_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create S-learner
s_learner = SLearner(model_final=base_model)

# Fit (note: econml uses (Y, T, X=X) signature)
s_learner.fit(Y_train, T_train, X=X_train)

# Predict CATE
tau_hat = s_learner.effect(X_test)  # shape (m,)

ate_hat = np.mean(tau_hat)
```

#### WITH Wrapper (Unified Interface)

```python
from causal_bench import SLearnerWrapper

model = SLearnerWrapper()  # uses default RandomForestRegressor

# Wrapper normalizes fit signature
model.fit(X_train, T_train, Y_train)

# Wrapper normalizes predict return (tuple)
tau_hat, lower_95, upper_95 = model.predict(X_test)

# Convenience
tau_hat, lower, upper, ate_hat, runtime = model.run(
    X_train, T_train, Y_train, X_test
)
```

**Key Differences:**
- Without wrapper: `.fit(Y, T, X=X)` (econml order)
- With wrapper: `.fit(X, T, Y)` (standard order)
- Without wrapper: `.effect()` vs with wrapper: `.predict()`
- Without wrapper: returns array; with wrapper: returns tuple (arr, None, None)

**Data Requirements:**
- X: float32 or int, (n, d) — any features
- T: float32 or int, (n,) — binary {0, 1}
- Y: float32, (n,) — continuous outcome

**How it works internally:**
```python
# S-learner: concatenate [X, T] and train one model
X_combined = np.hstack([X_train, T_train.reshape(-1, 1)])
# model: X_combined -> Y

# Predict CATE:
# tau(x) = model(x, T=1) - model(x, T=0)
X_test_t1 = np.hstack([X_test, np.ones(...)])
X_test_t0 = np.hstack([X_test, np.zeros(...)])
tau_hat = model.predict(X_test_t1) - model.predict(X_test_t0)
```

---

### T-learner

**What it is:** Two-model learner. Train SEPARATE models for T=0 and T=1. CATE = model_1(X) - model_0(X).

#### WITHOUT Wrapper (Raw API)

```python
import numpy as np
from econml.metalearners import TLearner
from sklearn.ensemble import RandomForestRegressor

# Data
X_train = np.array(..., dtype=np.float32)
T_train = np.array(..., dtype=np.float32)
Y_train = np.array(..., dtype=np.float32)
X_test = np.array(..., dtype=np.float32)

# Create base models
base_model_0 = RandomForestRegressor(n_estimators=100, random_state=0)
base_model_1 = RandomForestRegressor(n_estimators=100, random_state=1)

# Create T-learner (pass list of models)
t_learner = TLearner(models=[base_model_0, base_model_1])

# Fit
t_learner.fit(Y_train, T_train, X=X_train)

# Predict CATE
tau_hat = t_learner.effect(X_test)  # shape (m,)

ate_hat = np.mean(tau_hat)
```

#### WITH Wrapper (Unified Interface)

```python
from causal_bench import TLearnerWrapper

model = TLearnerWrapper()  # uses default models internally

model.fit(X_train, T_train, Y_train)
tau_hat, lower_95, upper_95 = model.predict(X_test)
tau_hat, lower, upper, ate_hat, runtime = model.run(
    X_train, T_train, Y_train, X_test
)
```

**Key Differences:**
- Without wrapper: pass list of models
- With wrapper: wrapper creates them internally
- Otherwise same as S-learner

**Data Requirements:** Same as S-learner

**How it works internally:**
```python
# T-learner: train TWO separate models
model_0 = RandomForest().fit(X_train[T_train == 0], Y_train[T_train == 0])
model_1 = RandomForest().fit(X_train[T_train == 1], Y_train[T_train == 1])

# Predict CATE:
# tau(x) = model_1(x) - model_0(x)
tau_hat = model_1.predict(X_test) - model_0.predict(X_test)
```

---

### X-learner

**What it is:** Cross-fit learner. More complex: trains model_0, computes residuals for T=1, trains model_1 on residuals. Asymptotically efficient.

#### WITHOUT Wrapper (Raw API)

```python
import numpy as np
from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor

# Data
X_train = np.array(..., dtype=np.float32)
T_train = np.array(..., dtype=np.float32)
Y_train = np.array(..., dtype=np.float32)
X_test = np.array(..., dtype=np.float32)

# Create base models
base_model_0 = RandomForestRegressor(n_estimators=100, random_state=0)
base_model_1 = RandomForestRegressor(n_estimators=100, random_state=1)

# Create X-learner
x_learner = XLearner(models=[base_model_0, base_model_1])

# Fit
x_learner.fit(Y_train, T_train, X=X_train)

# Predict CATE
tau_hat = x_learner.effect(X_test)  # shape (m,)

ate_hat = np.mean(tau_hat)
```

#### WITH Wrapper (Unified Interface)

```python
from causal_bench import XLearnerWrapper

model = XLearnerWrapper()

model.fit(X_train, T_train, Y_train)
tau_hat, lower_95, upper_95 = model.predict(X_test)
tau_hat, lower, upper, ate_hat, runtime = model.run(
    X_train, T_train, Y_train, X_test
)
```

**Key Differences:** API identical to T-learner (both use model list internally)

**Data Requirements:** Same as S-learner and T-learner

**How it works internally:**
```python
# X-learner: cross-fit approach
# Stage 1: fit outcome models
model_0 = RandomForest().fit(X_train[T==0], Y_train[T==0])
model_1 = RandomForest().fit(X_train[T==1], Y_train[T==1])

# Stage 2: compute "pseudo-outcomes" (residuals against opposite group)
residuals_1 = Y_train[T==1] - model_0.predict(X_train[T==1])
residuals_0 = model_1.predict(X_train[T==0]) - Y_train[T==0]

# Stage 3: fit treatment models on pseudo-outcomes
tau_model_1 = RandomForest().fit(X_train[T==1], residuals_1)
tau_model_0 = RandomForest().fit(X_train[T==0], residuals_0)

# Predict: weighted combination
tau_hat = g(X_test) * tau_model_1.predict(X_test) + \
          (1 - g(X_test)) * tau_model_0.predict(X_test)
```

---

### Debiased ML

**What it is:** Neyman-orthogonal approach. Uses doubly-robust-style debiasing to be robust to nuisance parameter estimation errors.

#### WITHOUT Wrapper (Raw API)

```python
import numpy as np
from econml.dml import DML
from sklearn.ensemble import RandomForestRegressor

# Data
X_train = np.array(..., dtype=np.float32)
T_train = np.array(..., dtype=np.float32)
Y_train = np.array(..., dtype=np.float32)
X_test = np.array(..., dtype=np.float32)

# Create models for outcome and treatment
model_y = RandomForestRegressor(n_estimators=100, random_state=0)
model_t = RandomForestRegressor(n_estimators=100, random_state=1)

# Create DML estimator
dml = DML(model_y=model_y, model_t=model_t)

# Fit
dml.fit(Y_train, T_train, X=X_train)

# Predict CATE
tau_hat = dml.effect(X_test)  # shape (m,)

ate_hat = np.mean(tau_hat)
```

#### WITH Wrapper (Unified Interface)

```python
from causal_bench import DebiasedMLWrapper

model = DebiasedMLWrapper()

model.fit(X_train, T_train, Y_train)
tau_hat, lower_95, upper_95 = model.predict(X_test)
tau_hat, lower, upper, ate_hat, runtime = model.run(
    X_train, T_train, Y_train, X_test
)
```

**Key Differences:**
- Without wrapper: specify both `model_y` and `model_t`
- With wrapper: wrapper creates them internally
- Otherwise similar to S/T/X learners

**Data Requirements:** Same as other metalearners

**How it works internally:**
```python
# DML: debiased machine learning
# Stage 1: fit propensity score model
propensity = RandomForest().fit(X_train, T_train)
p = propensity.predict_proba(X_train)[:, 1]

# Stage 2: fit outcome model
outcome_model = RandomForest().fit(X_train, Y_train)
mu = outcome_model.predict(X_train)

# Stage 3: residualize
Y_residual = Y_train - mu
T_residual = T_train - p

# Stage 4: fit final model on residuals
final_model = LinearRegression().fit(T_residual.reshape(-1,1), Y_residual)

# CATE estimate from final coefficient
tau_hat = final_model.coef_[0]  # constant CATE estimate
```

---

### IPW

**What it is:** Inverse Probability Weighting. Weights observations by propensity score to create pseudo-population.

#### WITHOUT Wrapper (Raw API)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# Data
X_train = np.array(..., dtype=np.float32)
T_train = np.array(..., dtype=np.float32)
Y_train = np.array(..., dtype=np.float32)
X_test = np.array(..., dtype=np.float32)

# Fit propensity score model
propensity_model = LogisticRegression(max_iter=1000)
propensity_model.fit(X_train, T_train)

# Get propensity scores
probs = propensity_model.predict_proba(X_train)
p_t = probs[:, 1]  # P(T=1|X)
p_c = probs[:, 0]  # P(T=0|X)

# Fit outcome models separately for each group
outcome_model_1 = RandomForestRegressor(n_estimators=100, random_state=0)
outcome_model_0 = RandomForestRegressor(n_estimators=100, random_state=1)

outcome_model_1.fit(X_train[T_train == 1], Y_train[T_train == 1])
outcome_model_0.fit(X_train[T_train == 0], Y_train[T_train == 0])

# Predict CATE
y_pred_1 = outcome_model_1.predict(X_test)
y_pred_0 = outcome_model_0.predict(X_test)

tau_hat = y_pred_1 - y_pred_0

# Note: IPW typically doesn't provide confidence intervals easily
```

#### WITH Wrapper (Unified Interface)

```python
from causal_bench import IPWWrapper

model = IPWWrapper()

model.fit(X_train, T_train, Y_train)
tau_hat, lower_95, upper_95 = model.predict(X_test)
tau_hat, lower, upper, ate_hat, runtime = model.run(
    X_train, T_train, Y_train, X_test
)
```

**Key Differences:**
- Without wrapper: manual propensity fitting and weighting
- With wrapper: simplified interface
- IPW is simple to implement but wrapper provides consistency

**Data Requirements:** Same as other metalearners

**How it works internally:**
```python
# IPW: inverse probability weighting
# ATE = E[Y*T / P(T=1|X)] - E[Y*(1-T) / P(T=0|X)]

# Fit propensity
propensity = LogisticRegression().fit(X_train, T_train)
p = propensity.predict_proba(X_train)[:, 1]

# Weights
weights_t = T_train / p
weights_c = (1 - T_train) / (1 - p)

# ATE from weighted averages
ate = np.mean(Y_train * weights_t) - np.mean(Y_train * weights_c)
```

---

### DR (Doubly Robust)

**What it is:** Combines outcome and propensity modeling. Robust if EITHER outcome model OR propensity model is correctly specified (not both necessary).

#### WITHOUT Wrapper (Raw API)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# Data
X_train = np.array(..., dtype=np.float32)
T_train = np.array(..., dtype=np.float32)
Y_train = np.array(..., dtype=np.float32)
X_test = np.array(..., dtype=np.float32)

# Fit propensity score model
propensity_model = LogisticRegression(max_iter=1000)
propensity_model.fit(X_train, T_train)
probs = propensity_model.predict_proba(X_train)
p_t = probs[:, 1]  # P(T=1|X)
p_c = probs[:, 0]  # P(T=0|X)

# Fit outcome models (weighted by inverse propensity)
outcome_model_1 = RandomForestRegressor(n_estimators=100, random_state=0)
outcome_model_0 = RandomForestRegressor(n_estimators=100, random_state=1)

# Fit on treated, weighted by 1/p_t
outcome_model_1.fit(X_train[T_train==1], Y_train[T_train==1], 
                    sample_weight=1 / (p_t[T_train==1] + 1e-6))

# Fit on control, weighted by 1/(1-p_t)
outcome_model_0.fit(X_train[T_train==0], Y_train[T_train==0],
                    sample_weight=1 / (p_c[T_train==0] + 1e-6))

# Predict CATE
y_pred_1 = outcome_model_1.predict(X_test)
y_pred_0 = outcome_model_0.predict(X_test)

tau_hat = y_pred_1 - y_pred_0
```

#### WITH Wrapper (Unified Interface)

```python
from causal_bench import DRWrapper

model = DRWrapper()

model.fit(X_train, T_train, Y_train)
tau_hat, lower_95, upper_95 = model.predict(X_test)
tau_hat, lower, upper, ate_hat, runtime = model.run(
    X_train, T_train, Y_train, X_test
)
```

**Key Differences:**
- Without wrapper: manual propensity + weighted outcome fitting
- With wrapper: simplified interface
- DR more complex than IPW but more robust

**Data Requirements:** Same as other metalearners

**How it works internally:**
```python
# DR: doubly robust
# ATE = E[T*Y/p - (T-p)*mu_1(X) / p - (1-T)*Y/(1-p) + (T-p)*mu_0(X)/(1-p)]

# where:
# - p = propensity score
# - mu_1, mu_0 = outcome models for T=1, T=0

# Fit propensity
p = LogisticRegression().fit(X_train, T_train).predict_proba(X_train)[:, 1]

# Fit outcome models (with IPW weighting for robustness)
mu_1 = RandomForest().fit(X_train[T==1], Y_train[T==1], 
                          sample_weight=1/(p[T==1] + eps))
mu_0 = RandomForest().fit(X_train[T==0], Y_train[T==0],
                          sample_weight=1/(1-p[T==0] + eps))

# DR combines both:
ate = (
    np.mean(T_train * Y_train / (p + eps)) -
    np.mean((T_train - p) * mu_1(X_train) / (p + eps)) -
    np.mean((1 - T_train) * Y_train / (1 - p + eps)) +
    np.mean((T_train - p) * mu_0(X_train) / (1 - p + eps))
)
```

---

## Comparison Table

| Model | Type | Trains New Model? | Data Input | API Without Wrapper | API With Wrapper | Key Feature |
|-------|------|-------------------|------------|---------------------|------------------|-------------|
| **CausalPFN** | Foundation | ❌ No (zero-shot) | X, T, Y | `.fit()` `.estimate_cate()` | `.fit()` `.predict()` `.run()` | In-context learning, transformer |
| **Do-PFN** | Foundation | ❌ No (zero-shot) | X, T, Y | `.fit()` on [X,T] `.predict_full()` twice | `.fit()` `.predict()` `.run()` | Do-calculus, SCM prior |
| **CausalFM** | Foundation | ❌ No (zero-shot) | X, T, Y | `.from_pretrained()` `.estimate_cate()` dict | `.fit()` `.predict()` `.run()` | Amortized, dict result |
| **S-learner** | Metalearner | ✅ Yes | X, T, Y | `.fit(Y,T,X)` `.effect()` | `.fit(X,T,Y)` `.predict()` `.run()` | Single model on X⊕T |
| **T-learner** | Metalearner | ✅ Yes | X, T, Y | `.fit(Y,T,X)` `.effect()` | `.fit(X,T,Y)` `.predict()` `.run()` | Two separate models |
| **X-learner** | Metalearner | ✅ Yes | X, T, Y | `.fit(Y,T,X)` `.effect()` | `.fit(X,T,Y)` `.predict()` `.run()` | Cross-fit, asymptotically efficient |
| **Debiased ML** | Metalearner | ✅ Yes | X, T, Y | `.fit(Y,T,X)` `.effect()` | `.fit(X,T,Y)` `.predict()` `.run()` | Neyman-orthogonal |
| **IPW** | Metalearner | ✅ Yes | X, T, Y | Propensity + outcomes (manual) | `.fit(X,T,Y)` `.predict()` `.run()` | Inverse probability weighting |
| **DR** | Metalearner | ✅ Yes | X, T, Y | Propensity + weighted outcomes | `.fit(X,T,Y)` `.predict()` `.run()` | Doubly robust |

---

## Overall Summary

### What Are Wrappers?

**Wrappers are adapter objects that standardize different APIs into one unified interface.**

Each model library has its own conventions:
- CausalPFN: `.estimate_cate()`
- econml: `.effect(X)` and `.fit(Y, T, X=X)`
- CausalFM: dict result with `.estimate_cate(X_train, T_train, Y_train, X_test)`
- IPW/DR: manual code (no class)

**Wrappers normalize all of this to:**
```python
model.fit(X, T, Y)
tau_hat, lower, upper = model.predict(X_test)
```

### Benefits

1. **Consistency** — All models have same interface
2. **Swappability** — Change one line to swap models
3. **Loop-ability** — Can loop over models without conditionals
4. **Maintainability** — Model-specific logic in one place
5. **Extensibility** — Adding new models doesn't break existing code

### When to Use Without Wrappers

- Deep diving into a specific model
- Need model-specific features (e.g., CausalPFN's calibrated quantiles)
- Learning how models work internally
- Research/experimentation

### When to Use With Wrappers

- ✅ Comparing multiple models
- ✅ Benchmarking
- ✅ Production code
- ✅ Notebooks
- ✅ Teaching unified concepts

---

## Architecture Overview

```
User Code
    ↓
┌─────────────────────────────────────────┐
│       Wrapper Interface (unified)        │
│  .fit(X, T, Y)                          │
│  .predict(X) → (tau, lower, upper)      │
│  .run() → (tau, lower, upper, ate, rt)  │
└─────────────────────────────────────────┘
    ↓
    ├─ CausalPFNWrapper ──→ causalpfn lib
    ├─ DoPFNWrapper ──────→ dopfn lib
    ├─ CausalFMWrapper ───→ causalfm lib
    ├─ SLearnerWrapper ───→ econml lib
    ├─ TLearnerWrapper ───→ econml lib
    ├─ XLearnerWrapper ───→ econml lib
    ├─ DebiasedMLWrapper ─→ econml lib
    ├─ IPWWrapper ────────→ sklearn lib
    └─ DRWrapper ─────────→ sklearn lib
```

### Data Flow

```
Raw Data (X, T, Y)
    ↓
Dataset Loader (data_generators.py or data_loader.py)
    ↓
SyntheticDataset / LalondeDataset
    ↓
Train/Test Split (train_test_split())
    ↓
Model.fit(X_train, T_train, Y_train)  ← Wrapper normalizes this
    ↓
Model.predict(X_test)  ← Wrapper normalizes this
    ↓
(tau_hat, lower, upper)
    ↓
evaluate_cate()  ← Compute metrics
    ↓
Results (PEHE, ATE error, bias, coverage, runtime)
```

---

## Example: Using Without vs With Wrapper

### Without Wrapper (Messy — 3 Different APIs)

```python
import numpy as np
import time
from causalpfn import CATEEstimator
from econml.metalearners import SLearner
from sklearn.ensemble import RandomForestRegressor

X_train, T_train, Y_train, X_test = ...

# CausalPFN: unique API
results = []
t0 = time.time()
pfn = CATEEstimator(device="cuda")
pfn.fit(X_train, T_train, Y_train)
tau_pfn = pfn.estimate_cate(X_test)
time_pfn = time.time() - t0
results.append(("CausalPFN", tau_pfn, time_pfn))

# S-learner: different API
t0 = time.time()
base = RandomForestRegressor(n_estimators=100)
s = SLearner(model_final=base)
s.fit(Y_train, T_train, X=X_train)  # Different arg order!
tau_s = s.effect(X_test)  # Different method name!
time_s = time.time() - t0
results.append(("S-learner", tau_s, time_s))

# Manual IPW: completely different
t0 = time.time()
from sklearn.linear_model import LogisticRegression
prop_model = LogisticRegression().fit(X_train, T_train)
p = prop_model.predict_proba(X_train)[:, 1]
m1 = RandomForestRegressor().fit(X_train[T_train==1], Y_train[T_train==1])
m0 = RandomForestRegressor().fit(X_train[T_train==0], Y_train[T_train==0])
tau_ipw = m1.predict(X_test) - m0.predict(X_test)
time_ipw = time.time() - t0
results.append(("IPW", tau_ipw, time_ipw))

# Problem: different shapes, types, names...
```

### With Wrapper (Clean — Same for All)

```python
import time
from causal_bench import CausalPFNWrapper, SLearnerWrapper, IPWWrapper

X_train, T_train, Y_train, X_test = ...

models = [
    CausalPFNWrapper(),
    SLearnerWrapper(),
    IPWWrapper(),
]

results = []
for model in models:
    t0 = time.time()
    model.fit(X_train, T_train, Y_train)
    tau_hat, _, _ = model.predict(X_test)
    runtime = time.time() - t0
    
    results.append((model.name, tau_hat, runtime))

# All same interface!
```

---

## Key Takeaways

1. **Models have different libraries and APIs** — CausalPFN, econml, sklearn all different
2. **Wrappers provide a unified interface** — `.fit()`, `.predict()`, `.run()`
3. **Without wrappers** — need model-specific code, copy-paste, hard to compare
4. **With wrappers** — loop over models, easy to add new ones, cleaner notebooks
5. **Wrapper pattern is a design best practice** — used everywhere in ML (PyTorch, scikit-learn, etc.)

Use wrappers for production/benchmarking. Use raw APIs when diving deep into a specific model.
