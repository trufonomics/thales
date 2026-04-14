"""xLSTM (Extended LSTM) for time series forecasting.

Based on TiRex — the NeurIPS 2025 model that achieved best CRPS on GIFT-Eval.
Uses sLSTM variant with exponential gating for state tracking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialGating(nn.Module):
    """Exponential gating as used in sLSTM.

    Unlike sigmoid gating (0-1), exponential gating uses exp() with
    normalization for dramatically more expressive control.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.w(x))


class SLSTMCell(nn.Module):
    """Simplified sLSTM cell with exponential gating."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Standard LSTM gates but with exponential input/forget gates
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Input gate
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Forget gate
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Output gate
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Cell candidate

        # Normalization for exponential gates
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=-1)

        # Exponential gates (key difference from standard LSTM)
        i = torch.exp(self.W_i(combined))
        f = torch.exp(self.W_f(combined))
        o = torch.sigmoid(self.W_o(combined))
        c_candidate = torch.tanh(self.W_c(combined))

        # Normalizer for numerical stability
        n = f * n + i
        c = f * c + i * c_candidate
        h = o * torch.tanh(self.ln(c / (n + 1e-6)))

        return h, c, n


class SLSTMBlock(nn.Module):
    """sLSTM block with residual connection and MLP."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.cell = SLSTMCell(d_model, hidden_dim)
        self.norm = nn.RMSNorm(d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(hidden_dim, d_model)
        self.mlp = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        c = torch.zeros(batch, self.hidden_dim, device=x.device)
        n = torch.ones(batch, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            h, c, n = self.cell(x[:, t], h, c, n)
            outputs.append(h)

        y = torch.stack(outputs, dim=1)
        y = self.proj_out(y)
        y = y + residual
        y = y + self.mlp(y)
        return y


class XLSTMForecaster(nn.Module):
    """xLSTM-based forecaster inspired by TiRex."""

    def __init__(
        self,
        num_streams: int,
        d_model: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        horizon: int = 90,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_streams = num_streams
        self.horizon = horizon

        self.input_proj = nn.Linear(num_streams, d_model)
        self.layers = nn.ModuleList([
            SLSTMBlock(d_model, hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.RMSNorm(d_model),
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
