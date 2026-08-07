"""
Load the Lalonde dataset for benchmarking.

Downloads from Rajeev Dehejia's NBER page (the original source) — no external
causal ML package required.

Features: age, educ, black, hisp, married, nodegree, re74, re75
Treatment: treat  (1 = NSW participant, 0 = control)
Outcome:   re78   (earnings in 1978)

Two different NSW comparisons live on that page, used here for different
purposes:
- **NSW-treated vs. PSID-controls**: the classic *observational* benchmark
  (severe selection bias — PSID controls are a much better-off population
  than the NSW experimental sample). This is the (X, T, Y) actually fed to
  every model — the confounded estimation task methods are being tested on.
- **NSW-treated vs. NSW-control**: the original *randomized experiment*.
  Because assignment was random, the simple difference in means here IS an
  unbiased ATE estimate — verified directly: **$1,794.34**, matching the
  standard literature benchmark (Dehejia & Wahba, 1999). This pair is never
  fed to any model; it only supplies the ground-truth `ate` to score against.

`ds.ate` is therefore the true experimental benchmark, not a naive
diff-in-means computed from the confounded (X, T, Y) models actually see —
that naive number (~-$15,205, driven almost entirely by selection bias, not
treatment effect) is exposed separately as `ds.ate_naive_observed` so it's
visible just how badly an unadjusted comparison is distorted here.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

_NBER_BASE = "http://www.nber.org/~rdehejia/data/"

_NBER_COLS = ["treat", "age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75", "re78"]
_FEATURE_COLS = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]
_TREATMENT_COL = "treat"
_OUTCOME_COL = "re78"

# NSW-treated is shared by every variant; only the control group differs.
_NSW_TREATED_URL = _NBER_BASE + "nswre74_treated.txt"
_NBER_FILES = {
    "nsw_psid": _NBER_BASE + "psid_controls.txt",
}
# Randomized-experiment control group -- used only to compute the true `ate`.
_NSW_EXPERIMENTAL_CONTROL_URL = _NBER_BASE + "nswre74_control.txt"

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")


@dataclass
class LalondeDataset:
    name: str
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    ate: float                 # true experimental ATE -- score against this
    ate_naive_observed: float  # naive diff-in-means on the confounded (X, T, Y) itself
    meta: Dict[str, Any]

    def train_test_split(self, train_frac: float = 0.7, seed: int = 0):
        n = len(self.Y)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_train = int(train_frac * n)
        return idx[:n_train], idx[n_train:]


def _load_group(url: str, cache_name: str) -> pd.DataFrame:
    cache_path = os.path.join(_CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    try:
        df = pd.read_csv(url, sep=r"\s+", header=None, names=_NBER_COLS)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Lalonde data from {url}\n"
            f"Error: {e}\n"
            f"Check your internet connection or manually place a CSV at: {cache_path}"
        )
    df.to_csv(cache_path, index=False)
    return df


def load_lalonde(variant: str = "nsw_psid") -> LalondeDataset:
    """
    Load Lalonde dataset: NSW-treated + PSID-controls (by default) as the
    (X, T, Y) estimation task, and separately the NSW-treated + NSW-control
    randomized comparison to compute the true `ate` to score against.
    """
    control_url = _NBER_FILES.get(variant)
    if control_url is None:
        raise ValueError(f"Unknown Lalonde variant '{variant}'. Choose from: {list(_NBER_FILES)}")

    treated = _load_group(_NSW_TREATED_URL, "lalonde_nsw_treated.csv")
    control = _load_group(control_url, f"lalonde_{variant}_control.csv")
    df = pd.concat([treated, control], ignore_index=True)

    X = df[_FEATURE_COLS].values.astype(np.float32)
    T = df[_TREATMENT_COL].values.astype(np.float32)
    Y = df[_OUTCOME_COL].values.astype(np.float32)
    ate_naive_observed = float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))

    nsw_control = _load_group(_NSW_EXPERIMENTAL_CONTROL_URL, "lalonde_nsw_control_experimental.csv")
    ate_experimental = float(treated[_OUTCOME_COL].mean() - nsw_control[_OUTCOME_COL].mean())

    return LalondeDataset(
        name=f"lalonde_{variant}",
        X=X, T=T, Y=Y,
        ate=ate_experimental,
        ate_naive_observed=ate_naive_observed,
        meta=dict(
            source="Dehejia & Wahba (1999) via NBER",
            variant=variant,
            n_samples=len(Y),
            n_features=X.shape[1],
            feature_names=_FEATURE_COLS,
            notes=(
                "X/T/Y are NSW-treated vs. PSID-controls (observational, "
                "confounded by design). `ate` is the true experimental "
                "benchmark from the separate, randomized NSW-treated vs. "
                "NSW-control comparison -- not computed from X/T/Y. "
                "`ate_naive_observed` is the naive diff-in-means on X/T/Y "
                "itself, exposed to show the scale of the selection bias a "
                "method needs to correct for."
            ),
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
