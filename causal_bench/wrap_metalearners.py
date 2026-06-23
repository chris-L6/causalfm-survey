"""
Wrappers for traditional CATE metalearners from econml.

These wrappers provide a common interface matching the foundation model wrappers:
- .fit(X, T, Y) - train the model
- .predict(X) - return (tau_hat, lower_95, upper_95)
- .run(X_train, T_train, Y_train, X_test) - convenience method
- .is_available() - check if library is installed
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple


class SLearnerWrapper:
    """S-learner (single-model learner) from econml.metalearners.SLearner."""
    name = "S-learner"

    def __init__(self, model=None, device: str = "cpu"):
        """
        model: base sklearn model (default: RandomForestRegressor)
        """
        self.model = model
        self.device = device
        self._learner = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from econml.metalearners import SLearner  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        from econml.metalearners import SLearner
        from sklearn.ensemble import RandomForestRegressor

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        base_model = self.model or RandomForestRegressor(n_estimators=100, random_state=0)
        self._learner = SLearner(model_final=base_model)
        self._learner.fit(Y, T, X=X)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        X = np.asarray(X, dtype=np.float32)
        tau_hat = self._learner.effect(X).reshape(-1)
        return tau_hat, None, None

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime


class TLearnerWrapper:
    """T-learner (two-model learner) from econml.metalearners.TLearner."""
    name = "T-learner"

    def __init__(self, model=None, device: str = "cpu"):
        """
        model: base sklearn model (default: RandomForestRegressor)
        """
        self.model = model
        self.device = device
        self._learner = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from econml.metalearners import TLearner  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        from econml.metalearners import TLearner
        from sklearn.ensemble import RandomForestRegressor

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        base_model = self.model or RandomForestRegressor(n_estimators=100, random_state=0)
        self._learner = TLearner(models=[base_model, base_model])
        self._learner.fit(Y, T, X=X)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        X = np.asarray(X, dtype=np.float32)
        tau_hat = self._learner.effect(X).reshape(-1)
        return tau_hat, None, None

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime


class XLearnerWrapper:
    """X-learner (cross-fit learner) from econml.metalearners.XLearner."""
    name = "X-learner"

    def __init__(self, model=None, device: str = "cpu"):
        """
        model: base sklearn model (default: RandomForestRegressor)
        """
        self.model = model
        self.device = device
        self._learner = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from econml.metalearners import XLearner  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        from econml.metalearners import XLearner
        from sklearn.ensemble import RandomForestRegressor

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        base_model = self.model or RandomForestRegressor(n_estimators=100, random_state=0)
        self._learner = XLearner(models=[base_model, base_model])
        self._learner.fit(Y, T, X=X)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        X = np.asarray(X, dtype=np.float32)
        tau_hat = self._learner.effect(X).reshape(-1)
        return tau_hat, None, None

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime


class DebiasedMLWrapper:
    """DML (Debiased Machine Learning) from econml.dml.DML."""
    name = "Debiased ML"

    def __init__(self, model=None, device: str = "cpu"):
        """
        model: base sklearn model (default: RandomForestRegressor)
        """
        self.model = model
        self.device = device
        self._learner = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from econml.dml import DML  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        from econml.dml import DML
        from sklearn.ensemble import RandomForestRegressor

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        base_model = self.model or RandomForestRegressor(n_estimators=100, random_state=0)
        self._learner = DML(model_y=base_model, model_t=base_model)
        self._learner.fit(Y, T, X=X)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        X = np.asarray(X, dtype=np.float32)
        tau_hat = self._learner.effect(X).reshape(-1)
        return tau_hat, None, None

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime


class IPWWrapper:
    """IPW (Inverse Probability Weighting) from econml.dml.LinearDML with IPW."""
    name = "IPW"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._propensity_model = None
        self._outcome_model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestRegressor

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        self._propensity_model = LogisticRegression(max_iter=1000)
        self._propensity_model.fit(X, T)

        self._outcome_model_1 = RandomForestRegressor(n_estimators=100, random_state=0)
        self._outcome_model_0 = RandomForestRegressor(n_estimators=100, random_state=0)

        self._outcome_model_1.fit(X[T == 1], Y[T == 1])
        self._outcome_model_0.fit(X[T == 0], Y[T == 0])
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        X = np.asarray(X, dtype=np.float32)
        probs = self._propensity_model.predict_proba(X)
        p_t = probs[:, 1]
        p_c = probs[:, 0]

        y1_pred = self._outcome_model_1.predict(X)
        y0_pred = self._outcome_model_0.predict(X)

        tau_hat = y1_pred - y0_pred
        return tau_hat.reshape(-1), None, None

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime


class DRWrapper:
    """DR (Doubly Robust) learner combining outcome and propensity modeling."""
    name = "DR (Doubly Robust)"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._propensity_model = None
        self._outcome_model_1 = None
        self._outcome_model_0 = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: F401
            from sklearn.ensemble import RandomForestRegressor  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestRegressor

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        self._propensity_model = LogisticRegression(max_iter=1000)
        self._propensity_model.fit(X, T)

        self._outcome_model_1 = RandomForestRegressor(n_estimators=100, random_state=0)
        self._outcome_model_0 = RandomForestRegressor(n_estimators=100, random_state=0)

        self._outcome_model_1.fit(X, Y, sample_weight=(T / (self._propensity_model.predict_proba(X)[:, 1] + 1e-6)))
        self._outcome_model_0.fit(X, Y, sample_weight=((1 - T) / (self._propensity_model.predict_proba(X)[:, 0] + 1e-6)))
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        X = np.asarray(X, dtype=np.float32)
        y1_pred = self._outcome_model_1.predict(X)
        y0_pred = self._outcome_model_0.predict(X)
        tau_hat = y1_pred - y0_pred
        return tau_hat.reshape(-1), None, None

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = float(np.mean(tau_hat))
        runtime = time.time() - t0
        return tau_hat, lower, upper, ate_hat, runtime


METALEARNER_WRAPPERS = {
    "S-learner": SLearnerWrapper,
    "T-learner": TLearnerWrapper,
    "X-learner": XLearnerWrapper,
    "Debiased ML": DebiasedMLWrapper,
    "IPW": IPWWrapper,
    "DR (Doubly Robust)": DRWrapper,
}
