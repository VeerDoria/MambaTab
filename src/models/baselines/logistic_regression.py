"""
Logistic Regression Baseline Model.

Simple linear classifier serving as the baseline floor in the paper.
"""

from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from src.models.base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Linear / Logistic Regression baseline model.

    Serves as the linear baseline floor (Logistic Regression for classification,
    Ridge Regression for regression).
    """

    name: str = "logistic_regression"

    def __init__(
        self,
        num_features: int,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 42,
        task_type: str = "classification",
    ):
        self.num_features = num_features
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.task_type = task_type.lower().strip()
        self.model = None
        self._is_fitted = False

    @property
    def num_parameters(self) -> int:
        """Parameters = weights (n_features) + bias (1)."""
        return self.num_features + 1

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> "LogisticRegressionModel":
        """Fit linear / logistic regression on training data."""
        lr_cfg = config.get("logistic_regression", {})
        C = lr_cfg.get("C", self.C)
        max_iter = lr_cfg.get("max_iter", self.max_iter)
        solver = lr_cfg.get("solver", self.solver)

        if self.task_type == "regression":
            # Ridge regression with L2 regularization corresponding to 1 / C
            alpha = 1.0 / max(C, 1e-5)
            self.model = Ridge(alpha=alpha, random_state=self.random_state)
            self.model.fit(X_train, y_train.astype(float))
        else:
            self.model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver=solver,
                random_state=self.random_state,
            )
            self.model.fit(X_train, y_train.astype(int))

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values (for regression)."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        preds = self.model.predict(X)
        return preds.ravel().astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities (for classification)."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1].astype(np.float32)
        elif probs.shape[1] == 1:
            return np.ones(len(X), dtype=np.float32) * float(self.model.classes_[0])
        return probs.ravel().astype(np.float32)
