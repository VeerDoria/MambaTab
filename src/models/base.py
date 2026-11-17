"""
Base classes for all models in the MambaTab replication study.

Provides:
- BaseModel: Abstract base class defining the required interface for all models
- TorchBaseModel: Base class for PyTorch-based models providing a unified,
  robust training loop with early stopping, learning rate scheduling,
  validation tracking, and parameter counting.
"""

from abc import ABC, abstractmethod
import copy
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.data.dataset import create_dataloader
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import get_device

logger = setup_logger("models.base")


class BaseModel(ABC):
    """Abstract base class defining the model interface."""

    name: str = "base_model"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> "BaseModel":
        """Train the model.

        Args:
            X_train: Training features, shape (n_train, n_features).
            y_train: Training targets, shape (n_train,).
            X_val: Validation features, shape (n_val, n_features).
            y_val: Validation targets, shape (n_val,).
            config: Full experiment configuration dictionary.

        Returns:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities for input samples.

        Args:
            X: Feature array, shape (n_samples, n_features).

        Returns:
            1D numpy array of probabilities, shape (n_samples,), with values in [0, 1].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Return the total number of learnable parameters."""
        raise NotImplementedError


class TorchBaseModel(BaseModel, nn.Module):
    """Base class for PyTorch neural network tabular models.

    Implements a standardized, reusable training procedure following
    the paper's protocol:
    - Adam optimizer
    - Cosine annealing learning rate schedule
    - Early stopping with configurable patience (default: 5)
    - BCE with logits loss
    - In-memory checkpointing of the best validation model
    """

    def __init__(self, task_type: str = "classification"):
        BaseModel.__init__(self)
        nn.Module.__init__(self)
        self.device = get_device()
        self.task_type = task_type.lower().strip()
        self.training_history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
        }

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _train_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> "TorchBaseModel":
        """Unified training loop for PyTorch tabular architectures."""
        self.to(self.device)

        training_cfg = config.get("training", {})
        max_epochs = training_cfg.get("max_epochs", 1000)
        patience = training_cfg.get("early_stopping_patience", 5)
        min_epochs = training_cfg.get("min_epochs", 10)
        early_stopping_metric = training_cfg.get("early_stopping_metric", "val_loss").lower()
        batch_size = training_cfg.get("batch_size", 64)
        lr = training_cfg.get("learning_rate", 1e-4)

        train_loader = create_dataloader(
            X_train, y_train, batch_size=batch_size, shuffle=True
        )
        val_loader = create_dataloader(
            X_val, y_val, batch_size=batch_size, shuffle=False
        )

        is_regression = self.task_type == "regression"
        criterion = nn.MSELoss() if is_regression else nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )

        best_val_loss = float("inf")
        best_val_auroc = 0.0
        best_weights = copy.deepcopy(self.state_dict())
        patience_counter = 0
        best_epoch = 0

        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
        }

        for epoch in range(1, max_epochs + 1):
            # Training phase
            self.train()
            epoch_loss = 0.0
            num_batches = 0

            for X_b, y_b in train_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                logits = self(X_b).squeeze(-1)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(1, num_batches)

            # Validation phase
            self.eval()
            val_loss = 0.0
            val_batches = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(self.device)
                    y_b = y_b.to(self.device)

                    logits = self(X_b).squeeze(-1)
                    loss = criterion(logits, y_b)

                    val_loss += loss.item()
                    val_batches += 1

                    if is_regression:
                        preds = logits.cpu().numpy().reshape(-1)
                        val_preds.extend(preds.tolist())
                    else:
                        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                        val_preds.extend(probs.tolist())
                    val_targets.extend(y_b.cpu().numpy().reshape(-1).tolist())

            avg_val_loss = val_loss / max(1, val_batches)

            # Calculate validation AUROC (classification only)
            if not is_regression:
                val_targets_arr = np.array(val_targets)
                val_preds_arr = np.array(val_preds)
                if len(np.unique(val_targets_arr)) > 1:
                    val_auroc = roc_auc_score(val_targets_arr, val_preds_arr)
                else:
                    val_auroc = 0.5
            else:
                val_auroc = 0.0

            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(avg_val_loss)
            self.training_history["val_auroc"].append(val_auroc)

            # Check early stopping criterion on validation loss (or AUROC)
            is_better = False
            if early_stopping_metric == "auroc" and not is_regression:
                if val_auroc > best_val_auroc + 1e-4:
                    is_better = True
            else:
                if avg_val_loss < best_val_loss - 1e-5:
                    is_better = True

            if is_better:
                best_val_loss = avg_val_loss
                best_val_auroc = val_auroc
                best_weights = copy.deepcopy(self.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                if epoch >= min_epochs:
                    patience_counter += 1

            if epoch % 50 == 0 or epoch == 1:
                if is_regression:
                    logger.info(
                        f"Epoch {epoch:4d}/{max_epochs:4d} | Train MSE: {avg_train_loss:.4f} | "
                        f"Val MSE: {avg_val_loss:.4f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch:4d}/{max_epochs:4d} | Train Loss: {avg_train_loss:.4f} | "
                        f"Val Loss: {avg_val_loss:.4f} | Val AUROC: {val_auroc:.4f}"
                    )

            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})"
                )
                break

        # Restore best model checkpoint
        self.load_state_dict(best_weights)
        self.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values (for regression)."""
        self.eval()
        self.to(self.device)

        batch_size = 256
        n_samples = len(X)
        preds = np.zeros(n_samples, dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_X = torch.FloatTensor(X[i : i + batch_size]).to(self.device)
                logits = self(batch_X).squeeze(-1)
                batch_preds = logits.cpu().numpy().reshape(-1)
                preds[i : i + len(batch_preds)] = batch_preds

        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities for input features."""
        self.eval()
        self.to(self.device)

        batch_size = 256
        n_samples = len(X)
        probs = np.zeros(n_samples, dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_X = torch.FloatTensor(X[i : i + batch_size]).to(self.device)
                logits = self(batch_X).squeeze(-1)
                batch_probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                probs[i : i + len(batch_probs)] = batch_probs

        return probs
