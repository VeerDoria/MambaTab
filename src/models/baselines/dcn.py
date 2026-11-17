"""
Deep & Cross Network (DCN) Baseline Model.

Pure PyTorch implementation of Deep & Cross Network introduced by:
Wang et al., "Deep & Cross Network for Ad Click Predictions", ADKDD 2017.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.base import TorchBaseModel


class CrossNetworkLayer(nn.Module):
    """Single layer of the Cross Network.

    Formula: x_{l+1} = x_0 * (x_l^T * w_l) + b_l + x_l
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(in_features, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(in_features))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """Cross layer forward pass.

        Args:
            x0: Initial input tensor of shape (batch_size, in_features).
            xl: Current layer input tensor of shape (batch_size, in_features).

        Returns:
            Tensor of shape (batch_size, in_features).
        """
        # xl * weight: (B, F) @ (F, 1) -> (B, 1)
        scalar_proj = torch.matmul(xl, self.weight)
        # x0 * scalar_proj + bias + xl
        return x0 * scalar_proj + self.bias + xl


class DCNModel(TorchBaseModel):
    """Deep & Cross Network (DCN) architecture."""

    name: str = "dcn"

    def __init__(
        self,
        num_features: int,
        hidden_dims: Optional[List[int]] = None,
        num_cross_layers: int = 2,
        dropout: float = 0.1,
        task_type: str = "classification",
    ):
        super().__init__(task_type=task_type)
        self.num_features = num_features
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.num_cross_layers = num_cross_layers
        self.dropout_rate = dropout

        # 1. Cross Network
        self.cross_layers = nn.ModuleList([
            CrossNetworkLayer(num_features) for _ in range(num_cross_layers)
        ])

        # 2. Deep Network (MLP)
        deep_layers = []
        prev_dim = num_features
        for h_dim in self.hidden_dims:
            deep_layers.append(nn.Linear(prev_dim, h_dim))
            deep_layers.append(nn.LayerNorm(h_dim))
            deep_layers.append(nn.ReLU())
            if dropout > 0:
                deep_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.deep_network = nn.Sequential(*deep_layers)

        # 3. Combination Layer
        # Concatenate Cross output (num_features) and Deep output (last hidden_dim)
        combined_dim = num_features + self.hidden_dims[-1]
        self.head = nn.Linear(combined_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor of shape (batch, num_features).

        Returns:
            Logits of shape (batch, 1).
        """
        # Cross network forward
        x0 = x
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        x_cross = xl

        # Deep network forward
        x_deep = self.deep_network(x)

        # Combine
        combined = torch.cat([x_cross, x_deep], dim=-1)
        logits = self.head(combined)
        return logits

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> "DCNModel":
        """Train DCN with the standardized training loop."""
        return self._train_loop(X_train, y_train, X_val, y_val, config)
