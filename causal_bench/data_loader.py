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

**Overlap**: NSW-treated and the full PSID-controls sample barely overlap at
all — verified directly (`age`, `married`, `nodegree`, `re74`, `re75` means
are wildly different between groups; PSID controls are, on average, 9 years
older, mostly married vs. mostly not, and earn ~9x more pre-treatment). This
is why practically every estimator applied to the untrimmed `nsw_psid`
variant gets the *sign* of the effect wrong, not just the magnitude — it
matches Dehejia & Wahba's own published "PSID-1" result. `variant="nsw_psid_trimmed"`
restricts PSID-controls to common propensity-score support with the treated
group (fit `p(treat=1|X)` on the pooled sample, keep only control units whose
score falls within the range observed among treated units — the standard
common-support trimming approach), giving methods an estimation task that
isn't defeated by construction.
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
# "nsw_psid_trimmed" reuses "nsw_psid"'s raw data with common-support trimming
# applied -- see _trim_to_common_support.
_TRIMMED_SUFFIX = "_trimmed"
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


def _trim_to_common_support(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict control units to the propensity-score range observed among
    the treated units (standard common-support trimming): fit p(treat=1|X)
    on the pooled sample, then drop any control unit whose score falls
    outside [min, max] of the treated units' scores -- those controls have
    no comparable treated unit to be compared against at all."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = df[_FEATURE_COLS].values.astype(np.float32)
    T = df[_TREATMENT_COL].values.astype(np.float32)
    # re74/re75 are in the thousands vs. age/educ in the tens -- without
    # standardizing, LogisticRegression's lbfgs solver overflows badly
    # enough to not converge at all (verified directly).
    X_s = StandardScaler().fit_transform(X)

    propensity_model = LogisticRegression(max_iter=1000)
    # NSW-treated and PSID-controls are separable enough on some covariates
    # (e.g. re74/re75) that exp() overflows internally for the most extreme
    # units -- harmless: it still saturates to the correct ~0/~1 probability
    # (verified: no NaNs in the output), which is exactly what should happen
    # for units with no comparable match. Silence the resulting noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        propensity_model.fit(X_s, T)
        p = propensity_model.predict_proba(X_s)[:, 1]

    treated_mask = T == 1
    lo, hi = p[treated_mask].min(), p[treated_mask].max()
    keep = treated_mask | ((p >= lo) & (p <= hi))
    return df[keep].reset_index(drop=True)


def load_lalonde(variant: str = "nsw_psid") -> LalondeDataset:
    """
    Load Lalonde dataset: NSW-treated + PSID-controls (by default) as the
    (X, T, Y) estimation task, and separately the NSW-treated + NSW-control
    randomized comparison to compute the true `ate` to score against.

    variant="nsw_psid_trimmed" applies common-support trimming to the PSID
    controls first (see _trim_to_common_support) -- use this if you want an
    estimation task that isn't defeated by near-zero covariate overlap.
    """
    trimmed = variant.endswith(_TRIMMED_SUFFIX)
    base_variant = variant[: -len(_TRIMMED_SUFFIX)] if trimmed else variant
    control_url = _NBER_FILES.get(base_variant)
    if control_url is None:
        available = list(_NBER_FILES) + [v + _TRIMMED_SUFFIX for v in _NBER_FILES]
        raise ValueError(f"Unknown Lalonde variant '{variant}'. Choose from: {available}")

    treated = _load_group(_NSW_TREATED_URL, "lalonde_nsw_treated.csv")
    control = _load_group(control_url, f"lalonde_{base_variant}_control.csv")
    df = pd.concat([treated, control], ignore_index=True)

    n_before_trim = len(df)
    if trimmed:
        df = _trim_to_common_support(df)
    n_dropped_by_trimming = n_before_trim - len(df)

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
            n_dropped_by_trimming=n_dropped_by_trimming,
            notes=(
                "X/T/Y are NSW-treated vs. PSID-controls (observational, "
                "confounded by design). `ate` is the true experimental "
                "benchmark from the separate, randomized NSW-treated vs. "
                "NSW-control comparison -- not computed from X/T/Y. "
                "`ate_naive_observed` is the naive diff-in-means on X/T/Y "
                "itself, exposed to show the scale of the selection bias a "
                "method needs to correct for."
                + (
                    f" Common-support trimming dropped {n_dropped_by_trimming} "
                    f"PSID-control units with no propensity-score overlap "
                    f"with any treated unit."
                    if trimmed else ""
                )
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
        "lalonde_nsw_psid_trimmed",
    ]
