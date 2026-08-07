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

    model = StandardCATEModel.from_pretrained("checkpoints/checkpoints_standard/best_model.pth")
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
import torch
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


class CausalFMWrapper:
    name = "CausalFM"

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None
        self._x_scaler = None
        self._y_scaler = None

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
        x_test)` signature. X and Y are standardized here -- CausalFM is
        pretrained on normalized synthetic priors, and raw real-world
        scales (e.g. Lalonde's dollar-denominated features/outcome) are out
        of that distribution; `predict` converts CATE back to the original
        outcome scale by multiplying by Y's std (the mean cancels in a
        difference of group means)."""
        from causalfm.models import StandardCATEModel

        if self._model is None:
            self._model = StandardCATEModel.from_pretrained(self.checkpoint_path)
            if hasattr(self._model, "to"):
                try:
                    self._model = self._model.to(self.device)
                except Exception:
                    pass

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        self._x_scaler = StandardScaler().fit(X)
        self._y_scaler = StandardScaler().fit(Y.reshape(-1, 1))

        self._X_train = self._x_scaler.transform(X).astype(np.float32)
        self._T_train = np.asarray(T, dtype=np.float32)
        self._Y_train = self._y_scaler.transform(Y.reshape(-1, 1)).reshape(-1).astype(np.float32)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        # The toolkit's PerFeatureTransformerCATE expects torch.Tensor inputs,
        # with treatment/outcome shaped [N, 1] (not the 1-D arrays our common
        # wrapper interface uses) -- see `_pack_eval_io` in
        # src/tabpfn/model/causalFM.py.
        X_test_s = self._x_scaler.transform(np.asarray(X, dtype=np.float32))

        X_train_t = torch.as_tensor(self._X_train, dtype=torch.float32)
        T_train_t = torch.as_tensor(self._T_train, dtype=torch.float32).reshape(-1, 1)
        Y_train_t = torch.as_tensor(self._Y_train, dtype=torch.float32).reshape(-1, 1)
        X_test_t = torch.as_tensor(X_test_s.astype(np.float32))

        result = self._model.estimate_cate(X_train_t, T_train_t, Y_train_t, X_test_t)

        tau_hat = result["cate"].detach().cpu().numpy().reshape(-1) * self._y_scaler.scale_[0]
        lower = upper = None
        # Optional calibrated uncertainty intervals, if the toolkit
        # returns them (key names per docs: 'cate_lower'/'cate_upper'
        # or 'ci_lower'/'ci_upper')
        for lk, uk in (("cate_lower", "cate_upper"), ("ci_lower", "ci_upper")):
            if lk in result and uk in result:
                lower = result[lk].detach().cpu().numpy().reshape(-1) * self._y_scaler.scale_[0]
                upper = result[uk].detach().cpu().numpy().reshape(-1) * self._y_scaler.scale_[0]
                break
        return tau_hat, lower, upper

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime
