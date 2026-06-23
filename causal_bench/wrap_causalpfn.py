"""
Thin wrapper around CausalPFN (https://github.com/vdblm/CausalPFN) exposing
a common `.fit(X, T, Y)` / `.predict(X)` interface for the benchmark.

CausalPFN: Balazadeh et al., "CausalPFN: Amortized Causal Effect Estimation
via In-Context Learning", arXiv:2506.07918.

Install:
    pip install causalpfn

The first call downloads pretrained weights from the Hugging Face Hub
(~ a few hundred MB), so an internet connection is required on first run.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple


class CausalPFNWrapper:
    name = "CausalPFN"

    def __init__(self, device: str = "cpu", verbose: bool = False):
        self.device = device
        self.verbose = verbose
        self._available = None
        self._cate_estimator = None
        self._ate_estimator = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from causalpfn import CATEEstimator, ATEEstimator  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        from causalpfn import CATEEstimator, ATEEstimator

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        self._cate_estimator = CATEEstimator(device=self.device, verbose=self.verbose)
        self._cate_estimator.fit(X, T, Y)

        self._ate_estimator = ATEEstimator(device=self.device, verbose=self.verbose)
        self._ate_estimator.fit(X, T, Y)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (tau_hat, lower_95, upper_95). Intervals are None if
        CausalPFN's installed version does not expose them through this
        call -- see its docs for `estimate_cate` with `return_quantiles`."""
        X = np.asarray(X, dtype=np.float32)
        tau_hat = self._cate_estimator.estimate_cate(X)
        return np.asarray(tau_hat).reshape(-1), None, None

    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
        # ATEEstimator.fit was already called with full data in `fit`
        ate_hat = self._ate_estimator.estimate_ate()
        return float(np.asarray(ate_hat).reshape(-1)[0])

    def run(self, X_train, T_train, Y_train, X_test):
        """Convenience: fit on train and return (tau_hat_test, ate_hat, runtime)."""
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = self.estimate_ate(X_train, T_train, Y_train)
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime
