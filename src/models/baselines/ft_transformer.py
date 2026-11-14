"""
FT-Transformer Baseline Model.

Pure PyTorch implementation of Feature Tokenizer + Transformer (FT-Transformer)
introduced by Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
NeurIPS 2021.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.base import TorchBaseModel


class FeatureTokenizer(nn.Module):
    """Linear feature tokenizer mapping continuous features to token embeddings."""

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        # Per-feature linear projection weights and biases
        self.weight = nn.Parameter(torch.randn(num_features, d_model))
        self.bias = nn.Parameter(torch.randn(num_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize continuous features.

        Args:
            x: Input feature tensor of shape (batch_size, num_features).

        Returns:
            Tokens of shape (batch_size, num_features, d_model).
        """
        # x: (B, F) -> (B, F, 1) * (1, F, D) + (1, F, D) -> (B, F, D)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerModel(TorchBaseModel):
    """FT-Transformer architecture for tabular classification."""

    name: str = "ft_transformer"

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ffn: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ffn = d_ffn
        self.dropout_rate = dropout
        self.attention_dropout = attention_dropout

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(num_features, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor (batch, num_features).

        Returns:
            Logits of shape (batch, 1).
        """
        batch_size = x.shape[0]

        # 1. Tokenize features -> (B, num_features, d_model)
        tokens = self.tokenizer(x)

        # 2. Prepend [CLS] token -> (B, num_features + 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_tokens = torch.cat([cls_tokens, tokens], dim=1)

        # 3. Transformer encoder
        encoded = self.encoder(x_tokens)

        # 4. Extract [CLS] representation (index 0)
        cls_out = self.norm(encoded[:, 0, :])

        # 5. Classification head
        logits = self.head(cls_out)
        return logits

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> "FTTransformerModel":
        """Train FT-Transformer with the standardized training loop."""
        return self._train_loop(X_train, y_train, X_val, y_val, config)
