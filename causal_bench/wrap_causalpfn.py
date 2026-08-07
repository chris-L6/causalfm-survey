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
import platform
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


class CausalPFNWrapper:
    name = "CausalPFN"

    def __init__(self, device: str = "cpu", verbose: bool = False):
        self.device = device
        self.verbose = verbose
        self._available = None
        self._cate_estimator = None
        self._ate_estimator = None
        self._x_scaler = None
        self._y_scaler = None

    @classmethod
    def is_available(cls) -> bool:
        # CausalPFN segfaults on Apple Silicon macOS -- a hard process crash,
        # not a catchable exception -- on both CPU and MPS (likely an
        # unstable scaled_dot_product_attention kernel, not a CUDA
        # requirement). Report unavailable here rather than let `fit()`
        # crash the interpreter. Fine on Colab (CPU or GPU).
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return False
        try:
            from causalpfn import CATEEstimator, ATEEstimator  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Standardizes X and Y before fitting -- CausalPFN is pretrained on
        normalized synthetic priors, and raw real-world scales (e.g. Lalonde's
        dollar-denominated features/outcome) are out of that distribution.
        Only a linear rescaling, so CATE/ATE are converted back to the
        original outcome scale in `predict`/`estimate_ate` (multiply by
        Y's std -- the mean cancels out in a difference of group means)."""
        from causalpfn import CATEEstimator, ATEEstimator

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        self._x_scaler = StandardScaler().fit(X)
        self._y_scaler = StandardScaler().fit(Y.reshape(-1, 1))
        X_s = self._x_scaler.transform(X).astype(np.float32)
        Y_s = self._y_scaler.transform(Y.reshape(-1, 1)).reshape(-1).astype(np.float32)

        self._cate_estimator = CATEEstimator(device=self.device, verbose=self.verbose)
        self._cate_estimator.fit(X_s, T, Y_s)

        self._ate_estimator = ATEEstimator(device=self.device, verbose=self.verbose)
        self._ate_estimator.fit(X_s, T, Y_s)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (tau_hat, lower_95, upper_95). Intervals are None if
        CausalPFN's installed version does not expose them through this
        call -- see its docs for `estimate_cate` with `return_quantiles`."""
        X = np.asarray(X, dtype=np.float32)
        X_s = self._x_scaler.transform(X).astype(np.float32)
        tau_hat_s = np.asarray(self._cate_estimator.estimate_cate(X_s)).reshape(-1)
        tau_hat = tau_hat_s * self._y_scaler.scale_[0]
        return tau_hat, None, None

    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
        # ATEEstimator.fit was already called with (standardized) full data in `fit`
        ate_hat_s = float(np.asarray(self._ate_estimator.estimate_ate()).reshape(-1)[0])
        return ate_hat_s * self._y_scaler.scale_[0]

    def run(self, X_train, T_train, Y_train, X_test):
        """Convenience: fit on train and return (tau_hat_test, ate_hat, runtime)."""
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = self.estimate_ate(X_train, T_train, Y_train)
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime
