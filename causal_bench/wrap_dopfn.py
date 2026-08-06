"""
Thin wrapper around Do-PFN (https://github.com/jr2021/Do-PFN) exposing a
common `.fit(X, T, Y)` / `.predict(X)` interface for the benchmark.

Do-PFN: Robertson & Reuter et al., "Do-PFN: In-Context Learning for Causal
Effect Estimation", arXiv:2506.06039.

Do-PFN is NOT distributed on PyPI -- it must be cloned from GitHub:

    git clone https://github.com/jr2021/Do-PFN.git

Don't `pip install -r Do-PFN/requirements.txt` as-is -- it's a frozen
research/benchmark environment (pins `catboost==1.1.1`, which has no wheel
for recent Python, purely for baseline comparisons `DoPFNRegressor` never
imports). The actual runtime deps beyond `torch`/`numpy`/`scipy`/`pandas`/
`scikit-learn` are just `networkx`, `tqdm`, `einops`.

This wrapper assumes the cloned `Do-PFN` directory lives at `repo_dir`
(default `"Do-PFN"`, relative to the process's cwd). Verified directly
against the current `jr2021/Do-PFN` main branch source (its README has
drifted from the code):

    from scripts.transformer_prediction_interface import DoPFNRegressor  # not `dopfn`

    dopfn = DoPFNRegressor()
    dopfn.fit(X_full_train, Y_train)          # X_full: treatment in COLUMN 0
    tau_hat = dopfn.predict_cate(X_full_test)  # torch.Tensor input required

`DoPFNRegressor()` loads its checkpoint via a path relative to the Do-PFN
repo root, lazily on both construction and `fit()` -- so this wrapper
`chdir`s into `repo_dir` for those calls and restores the original cwd
afterward (`finally`).

Requires `torch<2.10`: Do-PFN's own `model/layer.py` imports `Optional` from
`torch.nn.modules.transformer`, an unofficial re-export PyTorch removed in
2.10. Raises `ImportError: cannot import name 'Optional' from
'torch.nn.modules.transformer'` on newer torch -- not something this wrapper
can work around.
"""

from __future__ import annotations
import os
import time
import numpy as np
import torch
from typing import Optional, Tuple


class DoPFNWrapper:
    name = "Do-PFN"

    def __init__(self, repo_dir: str = "Do-PFN", device: str = "cpu"):
        """
        repo_dir: path to the cloned Do-PFN repo root (checkpoint paths are
            relative to it, so this wrapper `chdir`s there for
            construction/fit/predict).
        device: accepted for interface consistency with the other wrappers;
            DoPFNRegressor takes no device argument of its own.
        """
        self.repo_dir = repo_dir
        self.device = device
        self._model = None

    @classmethod
    def is_available(cls, repo_dir: str = "Do-PFN") -> bool:
        if not os.path.isdir(repo_dir):
            return False
        import sys
        sys.path.insert(0, os.path.abspath(repo_dir))
        try:
            from scripts.transformer_prediction_interface import DoPFNRegressor  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Builds the combined design matrix [T, X] -- treatment in column
        0, since `predict_cid` overwrites `X[:, 0]` internally regardless of
        what's there at prediction time."""
        import sys
        sys.path.insert(0, os.path.abspath(self.repo_dir))
        from scripts.transformer_prediction_interface import DoPFNRegressor

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32).reshape(-1, 1)
        Y = np.asarray(Y, dtype=np.float32)
        X_full = np.concatenate([T, X], axis=1)

        _cwd = os.getcwd()
        os.chdir(self.repo_dir)
        try:
            self._model = DoPFNRegressor()
            self._model.show_progress = False
            self._model.fit(X_full, Y)
        finally:
            os.chdir(_cwd)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (tau_hat, lower_95, upper_95) via Do-PFN's own
        `predict_cate`, which computes do(T=1) minus do(T=0) internally."""
        X = np.asarray(X, dtype=np.float32)
        # Column 0 is a placeholder -- predict_cate overwrites it internally.
        X_full = np.concatenate([np.zeros((len(X), 1), dtype=np.float32), X], axis=1)

        _cwd = os.getcwd()
        os.chdir(self.repo_dir)
        try:
            tau_hat = np.asarray(
                self._model.predict_cate(torch.as_tensor(X_full))
            ).reshape(-1)
        finally:
            os.chdir(_cwd)
        return tau_hat, None, None

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime
