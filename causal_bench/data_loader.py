"""
Load the Lalonde dataset for benchmarking.

The Lalonde dataset is a real-world causal inference benchmark combining:
- NSW (National Supported Work) program participants (treatment group)
- PSID (Panel Study of Income Dynamics) or CPS controls (control group)

Source: causalml.datasets.load_lalonde()
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class LalondeDataset:
    """Simple data holder for Lalonde (compatible with metrics evaluation)."""
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
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        return train_idx, test_idx


def load_lalonde(treatment_name: str = "NSW", control_name: str = "PSID") -> LalondeDataset:
    """
    Load Lalonde dataset from causalml.

    Parameters
    ----------
    treatment_name : str
        Treatment group source ("NSW" is the only standard treatment)
    control_name : str
        Control group source ("PSID" or "CPS")

    Returns
    -------
    LalondeDataset
        Dataset with X, T, Y, ate (observed ATE), metadata
    """
    try:
        from causalml.datasets import load_lalonde as causalml_load_lalonde
    except ImportError:
        raise ImportError(
            "causalml is required to load Lalonde dataset. "
            "Install it with: pip install causalml"
        )

    data = causalml_load_lalonde()
    X = data["X"].astype(np.float32)
    T = data["T"].astype(np.float32)
    Y = data["Y"].astype(np.float32)

    ate_observed = float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))

    dataset_name = f"lalonde_{treatment_name}_{control_name}".lower()

    return LalondeDataset(
        name=dataset_name,
        X=X,
        T=T,
        Y=Y,
        ate=ate_observed,
        meta=dict(
            source="Lalonde",
            treatment=treatment_name,
            control=control_name,
            n_samples=len(Y),
            n_features=X.shape[1],
            notes="Real-world data: no ground truth CATE available. ATE is observed mean difference.",
        ),
    )


def list_available_datasets() -> list[str]:
    """List available dataset names (for interactive demo)."""
    return [
        "linear_confounded",
        "nonlinear_heterogeneous",
        "iv_binary",
        "frontdoor",
        "lalonde_nsw_psid",
        "lalonde_nsw_cps",
    ]
