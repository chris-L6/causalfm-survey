"""
Thin wrapper around Do-PFN (https://github.com/jr2021/Do-PFN) exposing a
common `.fit(X, T, Y)` / `.predict(X)` interface for the benchmark.

Do-PFN: Robertson & Reuter et al., "Do-PFN: In-Context Learning for Causal
Effect Estimation", arXiv:2506.06039.

Do-PFN is NOT distributed on PyPI -- it must be cloned from GitHub and its
requirements installed:

    git clone https://github.com/jr2021/Do-PFN.git
    cd Do-PFN && pip install -r requirements.txt

This wrapper assumes the cloned `Do-PFN` directory has been added to
`sys.path` (see notebook setup cells), exposing `DoPFNRegressor` and
`load_dataset` from its top-level package, per its README:

    dopfn = DoPFNRegressor()
    dopfn.fit(train_ds.x, train_ds.y)
    y_pred = dopfn.predict_full(test_ds.x)

For CATE, Do-PFN's README estimates:
    CATE = E[y | do(t=1), x] - E[y | do(t=0), x]
by setting the treatment column to 1 and 0 respectively and taking the
difference of predicted means.
"""

from __future__ import annotations
import time
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple


class DoPFNWrapper:
    name = "Do-PFN"

    def __init__(self, treatment_col: int = 0, device: str = "cpu"):
        """
        treatment_col: index of the treatment column within the X matrix
            passed to `fit`/`predict`. Do-PFN treats the SCM input matrix
            as a single feature block including treatment; this wrapper
            concatenates [X, T] and sets treatment_col = X.shape[1] by
            default unless overridden (see `fit`).
        """
        self.treatment_col = treatment_col
        self.device = device
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from dopfn import DoPFNRegressor  # noqa: F401
            return True
        except Exception:
            try:
                # some forks expose it at top-level package import
                import DoPFN  # noqa: F401
                return True
            except Exception:
                return False

    def _get_regressor_cls(self):
        try:
            from dopfn import DoPFNRegressor
            return DoPFNRegressor
        except Exception:
            from model.dopfn import DoPFNRegressor  # fallback path used in some clones
            return DoPFNRegressor

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """X here is the covariate matrix WITHOUT treatment; this wrapper
        builds the combined design matrix [X, T] and remembers the
        treatment column index = X.shape[1]."""
        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32).reshape(-1, 1)
        Y = np.asarray(Y, dtype=np.float32)

        self.treatment_col = X.shape[1]
        X_full = np.concatenate([X, T], axis=1)

        RegCls = self._get_regressor_cls()
        self._model = RegCls()
        self._model.fit(X_full, Y)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (tau_hat, lower_95, upper_95) using Do-PFN's CATE
        formula from its README: CATE = E[y|do(t=1),x] - E[y|do(t=0),x]."""
        X = np.asarray(X, dtype=np.float32)
        x1, x0 = deepcopy(X), deepcopy(X)
        x1_full = np.concatenate([x1, np.ones((len(x1), 1), dtype=np.float32)], axis=1)
        x0_full = np.concatenate([x0, np.zeros((len(x0), 1), dtype=np.float32)], axis=1)

        y_pred_1 = self._model.predict_full(x1_full)
        y_pred_0 = self._model.predict_full(x0_full)

        # predict_full may return distributions; reduce to means if needed
        y1 = _reduce_to_mean(y_pred_1)
        y0 = _reduce_to_mean(y_pred_0)

        tau_hat = (y1 - y0).reshape(-1)
        return tau_hat, None, None

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime


def _reduce_to_mean(pred) -> np.ndarray:
    """Do-PFN's predict_full can return either a point prediction array
    or a richer object (e.g. dict with 'mean', or a distribution object
    with a `.mean()` method) depending on version. Handle common cases."""
    pred_arr = np.asarray(pred) if not isinstance(pred, dict) else None
    if isinstance(pred, dict):
        for key in ("mean", "mu", "y_pred", "prediction"):
            if key in pred:
                return np.asarray(pred[key]).reshape(-1)
        raise ValueError(f"Unrecognized Do-PFN prediction dict keys: {list(pred.keys())}")
    if hasattr(pred, "mean") and callable(getattr(pred, "mean")):
        try:
            return np.asarray(pred.mean()).reshape(-1)
        except TypeError:
            pass
    return pred_arr.reshape(-1)
