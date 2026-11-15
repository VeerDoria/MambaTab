"""
MambaTab: A Plug-and-Play Model for Learning Tabular Data.

Pure PyTorch implementation of the MambaTab-D architecture described in:
Ahamed & Cheng, "MambaTab: A Plug-and-Play Model for Learning Tabular Data",
arXiv:2401.08867 / IEEE MIPR 2024.

Architecture components:
1. Embedding Learner: Linear(n_features, embedding_dim=32)
2. Layer Normalization
3. ReLU activation
4. Cascading Selective SSM (Mamba) Blocks (M=1 default, with residual connections)
5. Prediction Head: Linear(embedding_dim, 1) -> Logits
"""

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import TorchBaseModel


class SelectiveSSM(nn.Module):
    """Selective State-Space Model core component (pure PyTorch).

    Implements the selective scan mechanism of Mamba (Gu & Dao, 2023):
    - Input-dependent B, C, and Δ (step size)
    - Diagonal state matrix A
    - Discretization via zero-order hold / exponential
    - Recurrent scan (efficient for L=1 tabular inputs)
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 32,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, math.ceil(d_inner / 16))

        # A parameter initialized as structured 1-D state space
        # A_log stores log(-A) where A_n = -(n+1)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))  # Skip connection

        # Input projections for B, C, and delta
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # Initialize dt_proj bias to positive for sensible initial delta
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize delta bias to ~0.01 - 0.1
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Selective SSM forward pass.

        Args:
            u: Input tensor of shape (batch, seq_len, d_inner).

        Returns:
            Output tensor of shape (batch, seq_len, d_inner).
        """
        batch_size, seq_len, d_inner = u.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Compute input-dependent projections
        # x_proj maps (B, L, d_inner) -> (B, L, dt_rank + 2 * d_state)
        x_dbl = self.x_proj(u)
        delta_proj, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Delta step size: (B, L, d_inner)
        delta = F.softplus(self.dt_proj(delta_proj))

        # For tabular data in MambaTab, seq_len = 1
        if seq_len == 1:
            # Squeeze time dimension for efficiency
            u_t = u.squeeze(1)  # (B, d_inner)
            delta_t = delta.squeeze(1)  # (B, d_inner)
            B_t = B_proj.squeeze(1)  # (B, d_state)
            C_t = C_proj.squeeze(1)  # (B, d_state)

            # Discretize:
            # dA = exp(delta * A): shape (B, d_inner, d_state)
            dA = torch.exp(torch.einsum("bd,dn->bdn", delta_t, A))
            # dB = delta * B: shape (B, d_inner, d_state)
            dB = torch.einsum("bd,bn->bdn", delta_t, B_t)

            # Recurrence with h_0 = 0:
            # h_1 = dB * u_1: shape (B, d_inner, d_state)
            h = dB * u_t.unsqueeze(-1)
            # y = C * h: shape (B, d_inner)
            y = torch.einsum("bdn,bn->bd", h, C_t)
            # Add skip connection D * u
            y = y + u_t * self.D
            return y.unsqueeze(1)

        # General multi-step scan (for L > 1 sequence lengths)
        ys = []
        h = torch.zeros(batch_size, d_inner, self.d_state, device=u.device)
        for t in range(seq_len):
            u_t = u[:, t, :]
            delta_t = delta[:, t, :]
            B_t = B_proj[:, t, :]
            C_t = C_proj[:, t, :]

            dA = torch.exp(torch.einsum("bd,dn->bdn", delta_t, A))
            dB = torch.einsum("bd,bn->bdn", delta_t, B_t)

            h = dA * h + dB * u_t.unsqueeze(-1)
            y_t = torch.einsum("bdn,bn->bd", h, C_t) + u_t * self.D
            ys.append(y_t)

        return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """Single Mamba Block as formulated in Mamba (Gu & Dao 2023) and MambaTab.

    Dual-branch structure:
    Branch 1: LP1 -> Causal 1D Conv -> SiLU -> Selective SSM
    Branch 2: LP2 -> SiLU
    Modulation: Branch 1 * SiLU(Branch 2)
    Projection: Linear -> Residual Add
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv

        # Linear projection to dual branches
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D causal convolution on branch 1
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # Selective SSM
        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)

        # Final projection back to model dimension
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a Mamba block with residual connection.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        """
        residual = x
        batch_size, seq_len, _ = x.shape

        # Dual branch linear projection
        projected = self.in_proj(x)
        x_branch, z_branch = torch.chunk(projected, 2, dim=-1)

        # Branch 1: 1D causal convolution + SiLU
        # Conv1d expects (batch, channels, seq_len)
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim causal padding
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Pass through selective SSM
        ssm_out = self.ssm(x_conv)

        # Branch 2 gating: multiplicative modulation with SiLU(z)
        gated = ssm_out * F.silu(z_branch)

        # Final projection and residual connection
        out = self.out_proj(gated)
        return out + residual


class MambaTabDModel(TorchBaseModel):
    """MambaTab-D model implementation for tabular classification.

    Follows the exact paper architecture:
    1. Embedding Learner (Linear -> LayerNorm -> ReLU)
    2. Cascading Mamba Blocks (M=1 default, N=32, d_conv=4, expand=2)
    3. Output Prediction (Linear -> Logit)
    """

    name: str = "mambatab_d"

    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 32,
        ssm_state_dim: int = 32,
        conv_width: int = 4,
        num_blocks: int = 1,
        block_expansion: int = 2,
    ):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.ssm_state_dim = ssm_state_dim
        self.conv_width = conv_width
        self.num_blocks = num_blocks
        self.block_expansion = block_expansion

        # Component 1: Embedding learner
        self.embedding_learner = nn.Linear(num_features, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.relu = nn.ReLU()

        # Component 2: Cascading Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=embedding_dim,
                d_state=ssm_state_dim,
                d_conv=conv_width,
                expand=block_expansion,
            )
            for _ in range(num_blocks)
        ])

        # Component 3: Output prediction head
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor of shape (batch, num_features).

        Returns:
            Logits of shape (batch, 1).
        """
        # Step 1: Learn embedded representation
        embedded = self.embedding_learner(x)
        normalized = self.layer_norm(embedded)
        activated = self.relu(normalized)

        # Step 2: Reshape to (Batch, Length=1, Dimension)
        u = activated.unsqueeze(1)

        # Step 3: Cascading Mamba blocks
        for block in self.blocks:
            u = block(u)

        # Step 4: Output prediction from length-1 token
        # (Batch, 1, Dimension) -> (Batch, Dimension)
        token_out = u.squeeze(1)
        logits = self.classifier(token_out)
        return logits

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> "MambaTabDModel":
        """Train MambaTab-D with the standardized training loop."""
        return self._train_loop(X_train, y_train, X_val, y_val, config)

    def expand_input_features(
        self,
        new_num_features: int,
        preserve_weights: bool = True,
    ) -> None:
        """Dynamically adapt the embedding learner for Feature-Incremental Learning (FIL).

        Expands the input dimension of `embedding_learner` while preserving the weights
        of downstream LayerNorm, Cascading Mamba Blocks, and Output Classifier.

        Args:
            new_num_features: The new input feature dimension.
            preserve_weights: If True, copies overlapping feature weights from the previous stage.
        """
        if new_num_features == self.num_features:
            return

        old_learner = self.embedding_learner
        device = old_learner.weight.device
        dtype = old_learner.weight.dtype

        new_learner = nn.Linear(new_num_features, self.embedding_dim, device=device, dtype=dtype)
        if preserve_weights:
            with torch.no_grad():
                min_feat = min(old_learner.in_features, new_num_features)
                new_learner.weight[:, :min_feat] = old_learner.weight[:, :min_feat]
                if old_learner.bias is not None and new_learner.bias is not None:
                    new_learner.bias.copy_(old_learner.bias)

        self.embedding_learner = new_learner
        self.num_features = new_num_features


class MambaTabTModel(MambaTabDModel):
    """MambaTab-T (Tuned) model.

    Instantiated with the dataset-specific optimal hyperparameters
    discovered and published in Table VI of the paper.
    """

    name: str = "mambatab_t"

