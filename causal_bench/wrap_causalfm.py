"""
Thin wrapper around CausalFM-toolkit
(https://github.com/yccm/CausalFM-toolkit) exposing a common
`.fit(X, T, Y)` / `.predict(X)` interface for the benchmark.

CausalFM: Ma, Frauen, Javurek & Feuerriegel, "Foundation Models for Causal
Inference via Prior-Data Fitted Networks", arXiv:2506.10914 (ICLR 2026).

Install:
    git clone https://github.com/yccm/CausalFM-toolkit.git
    cd CausalFM-toolkit && pip install -r requirements.txt

Per the toolkit's quick-start example:

    from causalfm.data import StandardCATEGenerator
    from causalfm.models import StandardCATEModel
    from causalfm.evaluation import compute_pehe

    model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
    result = model.estimate_cate(x_train, a_train, y_train, x_test)
    cate = result['cate']

This wrapper uses `StandardCATEModel` for the standard CATE setting (our
`linear_confounded` / `nonlinear_heterogeneous` datasets). A pretrained
checkpoint path must be supplied (see notebook setup); if unavailable,
`is_available()` returns False and the benchmark skips CausalFM.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple


class CausalFMWrapper:
    name = "CausalFM"

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    @classmethod
    def is_available(cls, checkpoint_path: Optional[str] = None) -> bool:
        try:
            from causalfm.models import StandardCATEModel  # noqa: F401
        except Exception:
            return False
        if checkpoint_path is not None:
            import os
            return os.path.exists(checkpoint_path)
        return True

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """CausalFM is amortized (zero-shot): 'fit' loads the pretrained
        model and stores the training context (X, T, Y), which is passed
        to `estimate_cate` together with the test points -- this mirrors
        the toolkit's `model.estimate_cate(x_train, a_train, y_train,
        x_test)` signature."""
        from causalfm.models import StandardCATEModel

        if self._model is None:
            self._model = StandardCATEModel.from_pretrained(self.checkpoint_path)
            if hasattr(self._model, "to"):
                try:
                    self._model = self._model.to(self.device)
                except Exception:
                    pass

        self._X_train = np.asarray(X, dtype=np.float32)
        self._T_train = np.asarray(T, dtype=np.float32)
        self._Y_train = np.asarray(Y, dtype=np.float32)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        X = np.asarray(X, dtype=np.float32)
        result = self._model.estimate_cate(self._X_train, self._T_train, self._Y_train, X)

        tau_hat = np.asarray(result["cate"]).reshape(-1)
        lower = upper = None
        # Optional calibrated uncertainty intervals, if the toolkit
        # returns them (key names per docs: 'cate_lower'/'cate_upper'
        # or 'ci_lower'/'ci_upper')
        for lk, uk in (("cate_lower", "cate_upper"), ("ci_lower", "ci_upper")):
            if lk in result and uk in result:
                lower = np.asarray(result[lk]).reshape(-1)
                upper = np.asarray(result[uk]).reshape(-1)
                break
        return tau_hat, lower, upper

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime
