"""
Synthetic data generators for benchmarking causal foundation models.

Each generator returns a dict with:
    X        : (n, d) covariates
    T        : (n,)   treatment (binary, 0/1)
    Y        : (n,)   observed outcome
    Y0, Y1   : (n,)   potential outcomes (for ground truth)
    tau      : (n,)   true CATE = Y1 - Y0
    ate      : float, true ATE = mean(tau)
    meta     : dict, description of the DGP / identification setting

All generators are seeded for reproducibility and are intentionally
simple closed-form structural causal models (SCMs), in the spirit of
the synthetic priors used by CausalPFN, Do-PFN and CausalFM. They are
NOT the exact training priors of those models (which are randomized,
large-scale, and not easily reproducible outside their repos) -- they
are a small, fixed, shared evaluation suite so that all three models
can be compared on identical data.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Any


@dataclass
class SyntheticDataset:
    name: str
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    Y0: np.ndarray
    Y1: np.ndarray
    tau: np.ndarray
    ate: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def train_test_split(self, train_frac: float = 0.7, seed: int = 0):
        n = len(self.Y)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_train = int(train_frac * n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        return train_idx, test_idx


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# 1. Linear confounding (textbook backdoor-adjustment setting)
# ---------------------------------------------------------------------------
def make_linear_confounded(n=2000, d=5, seed=0) -> SyntheticDataset:
    """
    Simple linear SCM with observed confounders (backdoor adjustment
    suffices). Constant CATE (homogeneous treatment effect = ATE).
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, size=(n, d)).astype(np.float32)

    # Propensity depends on X (confounding)
    logits = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    p = _sigmoid(logits)
    T = rng.binomial(1, p).astype(np.float32)

    tau_const = 2.0
    noise = rng.normal(0, 0.5, size=n).astype(np.float32)
    Y0 = X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2] + noise
    Y1 = Y0 + tau_const
    Y = np.where(T == 1, Y1, Y0).astype(np.float32)

    tau = (Y1 - Y0).astype(np.float32)
    return SyntheticDataset(
        name="linear_confounded",
        X=X, T=T, Y=Y, Y0=Y0, Y1=Y1, tau=tau, ate=float(np.mean(tau)),
        meta=dict(identification="backdoor", heterogeneous=False, d=d),
    )


# ---------------------------------------------------------------------------
# 2. Nonlinear heterogeneous effects (classic CATE/PEHE benchmark, IHDP-like)
# ---------------------------------------------------------------------------
def make_nonlinear_heterogeneous(n=2000, d=6, seed=0) -> SyntheticDataset:
    """
    Nonlinear response surfaces with heterogeneous treatment effect
    that depends on covariates (sine + interaction term), as used in
    e.g. CausalPFN's README example.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(1, 1, size=(n, d)).astype(np.float32)

    def true_cate(x):
        return np.sin(x[:, 0]) + 0.5 * x[:, 1]

    tau = true_cate(X).astype(np.float32)

    logits = 0.4 * (X[:, 2] - 1.0)
    p = _sigmoid(logits)
    T = rng.binomial(1, p).astype(np.float32)

    noise = rng.normal(0, 0.1, size=n).astype(np.float32)
    Y0 = X[:, 0] - X[:, 1] + 0.3 * X[:, 3] ** 2 + noise
    Y1 = Y0 + tau
    Y = np.where(T == 1, Y1, Y0).astype(np.float32)

    return SyntheticDataset(
        name="nonlinear_heterogeneous",
        X=X, T=T, Y=Y, Y0=Y0, Y1=Y1, tau=tau, ate=float(np.mean(tau)),
        meta=dict(identification="backdoor", heterogeneous=True, d=d),
    )


# ---------------------------------------------------------------------------
# 3. Instrumental variable setting (binary instrument, hidden confounder)
# ---------------------------------------------------------------------------
def make_iv_binary(n=2000, d=4, seed=0) -> SyntheticDataset:
    """
    Binary-instrument SCM with an unobserved confounder U affecting
    both T and Y. The instrument Z affects T but not Y directly.
    X are exogenous observed covariates (not required for identification
    but included so all models receive the same input shape).

    NOTE: Backdoor adjustment on X alone is NOT sufficient here because
    U is unobserved -- this dataset stress-tests methods that assume
    unconfoundedness (S/T/X-learners) vs. models designed for the IV
    regime (e.g. CausalFM's IV setting).
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    U = rng.normal(0, 1, size=n).astype(np.float32)          # unobserved
    Z = rng.binomial(1, 0.5, size=n).astype(np.float32)       # instrument

    # Treatment depends on instrument + hidden confounder
    t_logits = 1.5 * Z + 1.0 * U + 0.2 * X[:, 0]
    p = _sigmoid(t_logits)
    T = rng.binomial(1, p).astype(np.float32)

    tau_const = 1.5
    noise = rng.normal(0, 0.5, size=n).astype(np.float32)
    Y0 = 0.5 * X[:, 0] - 0.5 * X[:, 1] + 1.0 * U + noise
    Y1 = Y0 + tau_const
    Y = np.where(T == 1, Y1, Y0).astype(np.float32)

    tau = (Y1 - Y0).astype(np.float32)

    # Pack Z as an extra column for models that can use it (IV-aware models)
    X_with_Z = np.concatenate([X, Z.reshape(-1, 1)], axis=1).astype(np.float32)

    ds = SyntheticDataset(
        name="iv_binary",
        X=X_with_Z, T=T, Y=Y, Y0=Y0, Y1=Y1, tau=tau, ate=float(np.mean(tau)),
        meta=dict(identification="iv", heterogeneous=False, d=d + 1,
                  instrument_col=d, confounded=True),
    )
    return ds


# ---------------------------------------------------------------------------
# 4. Front-door adjustment setting
# ---------------------------------------------------------------------------
def make_frontdoor(n=2000, d=4, seed=0) -> SyntheticDataset:
    """
    Front-door SCM: T -> M -> Y, with an unobserved confounder U
    affecting T and Y directly but NOT the mediator M. Backdoor
    adjustment on X is insufficient; front-door adjustment via M
    identifies the effect.

    The mediator M is appended as an extra covariate column so models
    that don't explicitly support front-door adjustment can still be
    run (as a baseline / ablation), while models such as CausalFM's
    front-door setting can make use of it explicitly.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    U = rng.normal(0, 1, size=n).astype(np.float32)  # unobserved confounder

    t_logits = 0.3 * X[:, 0] + 0.8 * U
    p = _sigmoid(t_logits)
    T = rng.binomial(1, p).astype(np.float32)

    # Mediator depends on T only (plus its own noise)
    m_noise = rng.normal(0, 0.3, size=n).astype(np.float32)
    M = (1.0 * T + 0.2 * X[:, 1] + m_noise).astype(np.float32)

    tau_const = 1.0
    y_noise = rng.normal(0, 0.5, size=n).astype(np.float32)
    Y0 = 0.5 * X[:, 0] + 1.2 * M - 1.0 * U + y_noise
    # Recompute M for the counterfactual T=1 vs T=0 to get true potential outcomes
    M0 = (0.0 + 0.2 * X[:, 1] + m_noise).astype(np.float32)
    M1 = (1.0 + 0.2 * X[:, 1] + m_noise).astype(np.float32)
    Y0_true = 0.5 * X[:, 0] + 1.2 * M0 - 1.0 * U + y_noise
    Y1_true = 0.5 * X[:, 0] + 1.2 * M1 - 1.0 * U + y_noise + tau_const

    Y = np.where(T == 1, Y1_true, Y0_true).astype(np.float32)
    tau = (Y1_true - Y0_true).astype(np.float32)

    X_with_M = np.concatenate([X, M.reshape(-1, 1)], axis=1).astype(np.float32)

    return SyntheticDataset(
        name="frontdoor",
        X=X_with_M, T=T, Y=Y, Y0=Y0_true, Y1=Y1_true, tau=tau,
        ate=float(np.mean(tau)),
        meta=dict(identification="frontdoor", heterogeneous=False, d=d + 1,
                  mediator_col=d, confounded=True),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
GENERATORS: Dict[str, Callable[..., SyntheticDataset]] = {
    "linear_confounded": make_linear_confounded,
    "nonlinear_heterogeneous": make_nonlinear_heterogeneous,
    "iv_binary": make_iv_binary,
    "frontdoor": make_frontdoor,
}


def get_dataset(name: str, **kwargs) -> SyntheticDataset:
    if name not in GENERATORS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(GENERATORS)}")
    return GENERATORS[name](**kwargs)


def list_datasets():
    return list(GENERATORS.keys())
