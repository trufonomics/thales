"""Mamba-2 style selective SSM for time series forecasting.

Based on the architecture from MarineMamba — selective state space with
input-dependent gating of the state transition.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Simplified Mamba-style selective SSM block.

    Key difference from S5: Delta (discretization step) is input-dependent,
    allowing the model to selectively attend to or ignore inputs.
    """

    def __init__(self, d_model: int, state_dim: int = 64, conv_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Input projections
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)

        # Conv for local context
        self.conv = nn.Conv1d(
            d_model, d_model, conv_size,
            padding=conv_size - 1, groups=d_model
        )

        # SSM parameters
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(d_model))

        # Input-dependent projections (the "selective" part)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.B_proj = nn.Linear(d_model, state_dim, bias=False)
        self.C_proj = nn.Linear(d_model, state_dim, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        residual = x
        x = self.norm(x)

        batch, seq_len, _ = x.shape

        # Project and split into main path + gate
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        # Conv (causal)
        x_conv = self.conv(x_main.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Input-dependent SSM parameters
        A = -torch.exp(self.A_log)
        dt = F.softplus(self.dt_proj(x_conv))
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        # Discretize and scan
        A_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))

        h = torch.zeros(batch, self.d_model, self.state_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + x_conv[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1) + x_conv[:, t] * self.D
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        # Gate with z
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y + residual


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, state_dim: int = 64):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, state_dim)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ssm(x)
        x = x + self.mlp(self.norm(x))
        return x


class MambaForecaster(nn.Module):
    """Mamba-2 style forecaster."""

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
            MambaBlock(d_model, state_dim) for _ in range(num_layers)
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

        h = h[:, -1]
        out = self.output_proj(h)
        return out.reshape(-1, self.horizon, self.num_streams)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
