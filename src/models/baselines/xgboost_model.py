"""
XGBoost Baseline Model.

Wraps XGBoost classifier following the paper's baseline setup.
Includes graceful fallback to scikit-learn's HistGradientBoostingClassifier
if the xgboost binary is not yet compiled or installed.
"""

from typing import Any, Dict, Optional
import warnings

import numpy as np

from src.models.base import BaseModel
from src.utils.logging_utils import setup_logger

logger = setup_logger("models.xgboost")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier


class XGBoostModel(BaseModel):
    """XGBoost tabular classifier baseline."""

    name: str = "xgboost"

    def __init__(
        self,
        num_features: int,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: Optional[int] = 10,
        random_state: int = 42,
        device: Optional[str] = None,
        task_type: str = "classification",
    ):
        self.num_features = num_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.device = device
        self.task_type = task_type.lower().strip()
        self.model = None
        self._is_fitted = False

    @property
    def num_parameters(self) -> int:
        """Estimate total parameters (split nodes and leaf values in trees).

        Approximate: n_estimators * (2^(max_depth + 1) - 1) * 2
        """
        trees = getattr(self.model, "best_iteration", self.n_estimators)
        if trees is None:
            trees = self.n_estimators
        nodes_per_tree = (2 ** (min(self.max_depth, 6) + 1)) - 1
        return int(trees * nodes_per_tree)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> "XGBoostModel":
        """Fit the gradient boosting model with early stopping on validation data."""
        xgb_cfg = config.get(self.name, config.get("xgboost", {}))
        n_estimators = xgb_cfg.get("n_estimators", self.n_estimators)
        max_depth = xgb_cfg.get("max_depth", self.max_depth)
        lr = xgb_cfg.get("learning_rate", self.learning_rate)
        subsample = xgb_cfg.get("subsample", self.subsample)
        colsample = xgb_cfg.get("colsample_bytree", self.colsample_bytree)
        early_stopping = xgb_cfg.get("early_stopping_rounds", self.early_stopping_rounds)

        global XGBOOST_AVAILABLE, xgb
        if not XGBOOST_AVAILABLE:
            try:
                import xgboost as xgb
                XGBOOST_AVAILABLE = True
            except ImportError:
                pass

        is_regression = self.task_type == "regression"
        eval_metric = "rmse" if is_regression else "logloss"

        if XGBOOST_AVAILABLE:
            import torch
            xgb_device = self.device or xgb_cfg.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
            xgb_kwargs = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": lr,
                "subsample": subsample,
                "colsample_bytree": colsample,
                "eval_metric": eval_metric,
                "random_state": self.random_state,
                "verbosity": 0,
                "device": xgb_device,
            }
            if early_stopping is not None and early_stopping > 0:
                xgb_kwargs["early_stopping_rounds"] = early_stopping

            if is_regression:
                self.model = xgb.XGBRegressor(**xgb_kwargs)
                if early_stopping is not None and early_stopping > 0:
                    self.model.fit(
                        X_train,
                        y_train.astype(float),
                        eval_set=[(X_val, y_val.astype(float))],
                        verbose=False,
                    )
                else:
                    self.model.fit(
                        X_train,
                        y_train.astype(float),
                        verbose=False,
                    )
            else:
                self.model = xgb.XGBClassifier(**xgb_kwargs)
                if early_stopping is not None and early_stopping > 0:
                    self.model.fit(
                        X_train,
                        y_train.astype(int),
                        eval_set=[(X_val, y_val.astype(int))],
                        verbose=False,
                    )
                else:
                    self.model.fit(
                        X_train,
                        y_train.astype(int),
                        verbose=False,
                    )
        else:
            logger.warning(
                "xgboost package not found; using HistGradientBoosting as drop-in fallback."
            )
            if is_regression:
                from sklearn.ensemble import HistGradientBoostingRegressor
                self.model = HistGradientBoostingRegressor(
                    max_iter=n_estimators,
                    max_depth=max_depth,
                    learning_rate=lr,
                    early_stopping=bool(early_stopping and early_stopping > 0),
                    random_state=self.random_state,
                )
                self.model.fit(X_train, y_train.astype(float))
            else:
                self.model = HistGradientBoostingClassifier(
                    max_iter=n_estimators,
                    max_depth=max_depth,
                    learning_rate=lr,
                    early_stopping=bool(early_stopping and early_stopping > 0),
                    random_state=self.random_state,
                )
                self.model.fit(X_train, y_train.astype(int))

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target (for regression)."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        preds = self.model.predict(X)
        return preds.ravel().astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1].astype(np.float32)
        return probs.ravel().astype(np.float32)


class XGBoostPaperModel(XGBoostModel):
    """XGBoost baseline strictly matching the unregularized paper setup (TransTab 2022).

    Key properties:
      - Fixed 100 estimators (NO early stopping)
      - CPU execution (device='cpu')
      - Evaluated on initial feature subset s1 in Feature-Incremental Learning (FIL)
      - Perfectly reproduces the baseline protocol of TransTab / MambaTab Table II & III
    """

    name: str = "xgboost_paper"

    def __init__(
        self,
        num_features: int,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: Optional[int] = None,
        random_state: int = 42,
        device: str = "cpu",
        task_type: str = "classification",
    ):
        super().__init__(
            num_features=num_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            device=device,
            task_type=task_type,
        )

