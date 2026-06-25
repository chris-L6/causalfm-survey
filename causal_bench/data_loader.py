"""
Load the Lalonde dataset for benchmarking.

Downloads from Rajeev Dehejia's NBER page (the original source) — no external
causal ML package required.

Features: age, educ, black, hisp, married, nodegree, re74, re75
Treatment: treat  (1 = NSW participant, 0 = PSID/CPS control)
Outcome:   re78   (earnings in 1978)
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

_NBER_BASE = "http://www.nber.org/~rdehejia/data/"

_NBER_FILES = {
    "nsw_psid": [
        _NBER_BASE + "nswre74_treated.txt",
        _NBER_BASE + "psid_controls.txt",
    ],
}

_NBER_COLS = ["treat", "age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75", "re78"]

_FEATURE_COLS = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]
_TREATMENT_COL = "treat"
_OUTCOME_COL = "re78"

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")


@dataclass
class LalondeDataset:
    name: str
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    ate: float
    meta: Dict[str, Any]

    def train_test_split(self, train_frac: float = 0.7, seed: int = 0):
        n = len(self.Y)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_train = int(train_frac * n)
        return idx[:n_train], idx[n_train:]


def load_lalonde(variant: str = "nsw_psid") -> LalondeDataset:
    """
    Load Lalonde dataset (NSW treated + PSID controls by default).

    Downloads from Dehejia's NBER page on first call and caches locally.
    """
    cache_path = os.path.join(_CACHE_DIR, f"lalonde_{variant}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        urls = _NBER_FILES.get(variant)
        if urls is None:
            raise ValueError(f"Unknown Lalonde variant '{variant}'. Choose from: {list(_NBER_FILES)}")

        frames = []
        for url in urls:
            try:
                frames.append(
                    pd.read_csv(url, sep=r"\s+", header=None, names=_NBER_COLS)
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download Lalonde data from {url}\n"
                    f"Error: {e}\n"
                    f"Check your internet connection or manually place a CSV at: {cache_path}"
                )

        df = pd.concat(frames, ignore_index=True)
        df.to_csv(cache_path, index=False)

    X = df[_FEATURE_COLS].values.astype(np.float32)
    T = df[_TREATMENT_COL].values.astype(np.float32)
    Y = df[_OUTCOME_COL].values.astype(np.float32)
    ate_observed = float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))

    return LalondeDataset(
        name=f"lalonde_{variant}",
        X=X, T=T, Y=Y,
        ate=ate_observed,
        meta=dict(
            source="Dehejia & Wahba (1999) via NBER",
            variant=variant,
            n_samples=len(Y),
            n_features=X.shape[1],
            feature_names=_FEATURE_COLS,
            notes="Real-world data: no ground truth CATE. ATE is observed mean difference.",
        ),
    )


def list_available_datasets() -> list:
    return [
        "linear_confounded",
        "nonlinear_heterogeneous",
        "iv_binary",
        "frontdoor",
        "lalonde_nsw_psid",
    ]
