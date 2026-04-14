"""S5 (Simplified State Space Sequence) model for time series forecasting.

Based on FlowState's architecture — the SSM that beat models 55x larger on GIFT-Eval.
Uses HiPPO initialization for the state matrix.
"""

import math
import torch
import torch.nn as nn
import numpy as np


def hippo_init(state_dim: int) -> np.ndarray:
    """HiPPO-LegS initialization for the A matrix.

    Constructs the A matrix such that the hidden state approximates
    Legendre polynomial coefficients of the input history.
    """
    A = np.zeros((state_dim, state_dim))
    for i in range(state_dim):
        for j in range(state_dim):
            if i > j:
                A[i, j] = -(2 * i + 1) ** 0.5 * (2 * j + 1) ** 0.5
            elif i == j:
                A[i, j] = -(i + 1)
    return A


class S5Block(nn.Module):
    """Single S5 layer with diagonal state space."""

    def __init__(self, input_dim: int, state_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # Initialize with HiPPO, then take diagonal (eigenvalues)
        A_init = hippo_init(state_dim)
        eigenvalues = np.linalg.eigvals(A_init).real
        eigenvalues = np.sort(eigenvalues)[:state_dim]

        # Learnable parameters (real-valued diagonal SSM)
        self.log_A = nn.Parameter(torch.tensor(
            np.log(-eigenvalues + 1e-6), dtype=torch.float32
        ))
        self.B = nn.Parameter(torch.randn(input_dim, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.D = nn.Parameter(torch.zeros(input_dim))
        self.log_dt = nn.Parameter(torch.tensor(0.0))

        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        residual = x
        x = self.norm(x)

        batch, seq_len, _ = x.shape
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A)

        # Discretize: A_bar = exp(A * dt)
        A_bar = torch.exp(A * dt)

        # Sequential scan
        h = torch.zeros(batch, self.state_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = A_bar * h + x[:, t] @ self.B
            y = h @ self.C + x[:, t] * self.D
            outputs.append(y)

        y = torch.stack(outputs, dim=1)
        y = y + residual
        y = y + self.mlp(self.norm(y))
        return y


class S5Forecaster(nn.Module):
    """S5-based forecaster inspired by FlowState."""

    def __init__(
        self,
        num_streams: int,
        d_model: int = 256,
        state_dim: int = 64,
        num_layers: int = 4,
        horizon: int = 90,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_streams = num_streams
        self.horizon = horizon

        self.input_proj = nn.Linear(num_streams, d_model)
        self.layers = nn.ModuleList([
            S5Block(d_model, state_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon * num_streams),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, context_len, num_streams]
        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h)
            h = self.dropout(h)

        h = h[:, -1]  # Take final hidden state
        out = self.output_proj(h)
        return out.reshape(-1, self.horizon, self.num_streams)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
