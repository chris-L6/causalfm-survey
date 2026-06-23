"""
Evaluation metrics for causal effect estimators.

Implements the standard metrics used across the CausalPFN, Do-PFN and
CausalFM papers:

  - PEHE  : sqrt(E[(tau_hat - tau)^2])           (precision in estimating
                                                   heterogeneous effects)
  - eps_ATE : |ATE_hat - ATE_true|                (absolute ATE error)
  - ATE relative error : |ATE_hat - ATE_true| / |ATE_true|
  - Bias  : mean(tau_hat - tau)
  - Coverage @ 95% : fraction of true tau values inside a model's
                      reported 95% interval (only computed if the
                      model returns uncertainty estimates)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    tau_hat = np.asarray(tau_hat, dtype=np.float64)
    tau_true = np.asarray(tau_true, dtype=np.float64)
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def ate_abs_error(ate_hat: float, ate_true: float) -> float:
    return float(abs(ate_hat - ate_true))


def ate_rel_error(ate_hat: float, ate_true: float, eps: float = 1e-8) -> float:
    return float(abs(ate_hat - ate_true) / (abs(ate_true) + eps))


def bias(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(np.mean(np.asarray(tau_hat) - np.asarray(tau_true)))


def coverage_95(
    tau_true: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
) -> Optional[float]:
    """Fraction of true tau within [lower, upper]. Returns None if
    the model does not provide intervals."""
    if lower is None or upper is None:
        return None
    tau_true = np.asarray(tau_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    inside = (tau_true >= lower) & (tau_true <= upper)
    return float(np.mean(inside))


def evaluate_cate(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    ate_hat: Optional[float] = None,
    ate_true: Optional[float] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    runtime_s: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Compute the full metric suite for one model on one dataset."""
    tau_hat = np.asarray(tau_hat, dtype=np.float64).reshape(-1)
    tau_true = np.asarray(tau_true, dtype=np.float64).reshape(-1)

    if ate_hat is None:
        ate_hat = float(np.mean(tau_hat))
    if ate_true is None:
        ate_true = float(np.mean(tau_true))

    result = {
        "pehe": pehe(tau_hat, tau_true),
        "ate_hat": float(ate_hat),
        "ate_true": float(ate_true),
        "ate_abs_error": ate_abs_error(ate_hat, ate_true),
        "ate_rel_error": ate_rel_error(ate_hat, ate_true),
        "bias": bias(tau_hat, tau_true),
        "coverage_95": coverage_95(tau_true, lower, upper),
        "runtime_s": runtime_s,
    }
    return result
