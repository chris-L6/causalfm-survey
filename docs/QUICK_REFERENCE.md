# Quick Reference Cheat Sheet

## One-Liner Comparison

| Model | Install | Data | Train? | Best For |
|-------|---------|------|--------|----------|
| **CausalPFN** | `pip install causalpfn` | float32 | ❌ | Fast, general CATE |
| **Do-PFN** | `git clone + pip install` | float32 | ❌ | SCM-based, robust |
| **CausalFM** | `git clone + pip install` | float32 | ❌ | IV/frontdoor settings |
| **S-learner** | in requirements.txt | any | ✅ | Baseline, simple |
| **T-learner** | in requirements.txt | any | ✅ | Separate groups |
| **X-learner** | in requirements.txt | any | ✅ | Asymptotically efficient |
| **Debiased ML** | in requirements.txt | any | ✅ | Robust nuisance |
| **IPW** | in requirements.txt | any | ✅ | Propensity-based |
| **DR** | in requirements.txt | any | ✅ | Double robustness |

---

## Using Each Model (WITH Wrapper)

```python
from causal_bench import (
    CausalPFNWrapper, DoPFNWrapper, CausalFMWrapper,
    SLearnerWrapper, TLearnerWrapper, XLearnerWrapper,
    DebiasedMLWrapper, IPWWrapper, DRWrapper,
    evaluate_cate
)

# Data: X (n, d), T (n,), Y (n,)
X_train, T_train, Y_train, X_test, tau_true = ...

# Pick a model
model = CausalPFNWrapper()  # or any other

# Standard workflow
model.fit(X_train, T_train, Y_train)
tau_hat, lower, upper = model.predict(X_test)

# Evaluate
metrics = evaluate_cate(tau_hat, tau_true, lower=lower, upper=upper)
print(f"PEHE: {metrics['pehe']:.4f}")
```

---

## Using Each Model (WITHOUT Wrapper)

### CausalPFN
```python
from causalpfn import CATEEstimator, ATEEstimator
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
cate = CATEEstimator(device=device)
cate.fit(X_train, T_train, Y_train)
tau_hat = cate.estimate_cate(X_test)
```

### Do-PFN
```python
from dopfn import DoPFNRegressor
import numpy as np

X_full = np.concatenate([X_train, T_train.reshape(-1,1)], axis=1)
dopfn = DoPFNRegressor()
dopfn.fit(X_full, Y_train)

x1_full = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)
x0_full = np.concatenate([X_test, np.zeros((len(X_test), 1))], axis=1)
tau_hat = dopfn.predict_full(x1_full) - dopfn.predict_full(x0_full)
```

### CausalFM
```python
from causalfm.models import StandardCATEModel

model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
result = model.estimate_cate(X_train, T_train, Y_train, X_test)
tau_hat = result["cate"]
```

### S-learner
```python
from econml.metalearners import SLearner
from sklearn.ensemble import RandomForestRegressor

base = RandomForestRegressor(n_estimators=100)
s = SLearner(model_final=base)
s.fit(Y_train, T_train, X=X_train)
tau_hat = s.effect(X_test)
```

### T-learner
```python
from econml.metalearners import TLearner
from sklearn.ensemble import RandomForestRegressor

base1 = RandomForestRegressor(n_estimators=100)
base2 = RandomForestRegressor(n_estimators=100)
t = TLearner(models=[base1, base2])
t.fit(Y_train, T_train, X=X_train)
tau_hat = t.effect(X_test)
```

### X-learner
```python
from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor

base1 = RandomForestRegressor(n_estimators=100)
base2 = RandomForestRegressor(n_estimators=100)
x = XLearner(models=[base1, base2])
x.fit(Y_train, T_train, X=X_train)
tau_hat = x.effect(X_test)
```

### Debiased ML
```python
from econml.dml import DML
from sklearn.ensemble import RandomForestRegressor

model_y = RandomForestRegressor(n_estimators=100)
model_t = RandomForestRegressor(n_estimators=100)
dml = DML(model_y=model_y, model_t=model_t)
dml.fit(Y_train, T_train, X=X_train)
tau_hat = dml.effect(X_test)
```

### IPW (Manual)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

prop = LogisticRegression(max_iter=1000)
prop.fit(X_train, T_train)
p = prop.predict_proba(X_train)[:, 1]

m1 = RandomForestRegressor().fit(X_train[T_train==1], Y_train[T_train==1])
m0 = RandomForestRegressor().fit(X_train[T_train==0], Y_train[T_train==0])

tau_hat = m1.predict(X_test) - m0.predict(X_test)
```

### DR (Manual)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

prop = LogisticRegression(max_iter=1000)
prop.fit(X_train, T_train)
p = prop.predict_proba(X_train)[:, 1]

m1 = RandomForestRegressor().fit(X_train[T_train==1], Y_train[T_train==1],
                                 sample_weight=1/(p[T_train==1]+1e-6))
m0 = RandomForestRegressor().fit(X_train[T_train==0], Y_train[T_train==0],
                                 sample_weight=1/(1-p[T_train==0]+1e-6))

tau_hat = m1.predict(X_test) - m0.predict(X_test)
```

---

## Common Patterns

### Loop Over All Models
```python
from causal_bench import (
    CausalPFNWrapper, DoPFNWrapper, CausalFMWrapper,
    SLearnerWrapper, TLearnerWrapper, XLearnerWrapper,
    DebiasedMLWrapper, IPWWrapper, DRWrapper
)

models = [
    CausalPFNWrapper(),
    DoPFNWrapper(),
    CausalFMWrapper(),
    SLearnerWrapper(),
    TLearnerWrapper(),
    XLearnerWrapper(),
    DebiasedMLWrapper(),
    IPWWrapper(),
    DRWrapper(),
]

results = []
for model in models:
    if not model.is_available():
        print(f"Skipping {model.name} (not installed)")
        continue
    
    tau_hat, lower, upper, ate_hat, runtime = model.run(
        X_train, T_train, Y_train, X_test
    )
    results.append({
        "model": model.name,
        "pehe": pehe(tau_hat, tau_true),
        "runtime": runtime,
    })
```

### Compare on Synthetic Data
```python
from causal_bench import get_dataset, evaluate_cate, SLearnerWrapper

ds = get_dataset("nonlinear_heterogeneous", n=2000, seed=0)
train_idx, test_idx = ds.train_test_split(0.7, seed=0)

model = SLearnerWrapper()
tau_hat, _, _ = model.predict(model.fit(
    ds.X[train_idx], ds.T[train_idx], ds.Y[train_idx]
).predict(ds.X[test_idx]))

metrics = evaluate_cate(tau_hat, ds.tau[test_idx])
```

### Compare on Real Data (Lalonde)
```python
from causal_bench import load_lalonde, SLearnerWrapper, TLearnerWrapper

ds = load_lalonde()
train_idx, test_idx = ds.train_test_split(0.7, seed=0)

for ModelClass in [SLearnerWrapper, TLearnerWrapper]:
    model = ModelClass()
    tau_hat, _, _ = model.predict(model.fit(
        ds.X[train_idx], ds.T[train_idx], ds.Y[train_idx]
    ).predict(ds.X[test_idx]))
    
    ate_hat = tau_hat.mean()
    print(f"{model.name} ATE: {ate_hat:.4f}")
```

---

## Key Insights

**Foundation Models (CausalPFN, Do-PFN, CausalFM):**
- ❌ Don't train: just condition on data
- ✅ Fast inference
- ✅ Zero-shot
- ✅ May have learned causal priors from training

**Metalearners (S/T/X, Debiased ML, IPW, DR):**
- ✅ Train on your data
- ❌ Slower (fit new model each time)
- ✅ Well-understood theory
- ✅ No external dependencies

**When to Use:**
- Foundation models: quick comparisons, real-world data
- Metalearners: baselines, theory validation, small data
- DR/Debiased ML: robustness to model misspecification
- X-learner: efficiency (asymptotically)
