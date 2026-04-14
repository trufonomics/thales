"""Experiment 2 — xLSTM only (with numerical stability fixes).

The exponential gating in sLSTM causes NaN with default LR.
Fixes: lower LR, clamped exp gates, gradient scaling.

Usage:
    python scripts/experiment_02_xlstm.py

After this completes, run evaluate_checkpoints.py to compare all 4.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_categories
from src.dataset import TimeSeriesDataset, normalize_streams
from src.trainer import train_model


RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiment_02"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class StableSLSTMCell(nn.Module):
    """sLSTM cell with clamped exponential gates for numerical stability."""

    def __init__(self, input_dim, hidden_dim, exp_clamp=5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.exp_clamp = exp_clamp

        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        # Initialize forget gate bias high (keep memory by default)
        nn.init.constant_(self.W_f.bias, 1.0)
        # Initialize input gate bias low (be selective)
        nn.init.constant_(self.W_i.bias, -1.0)

    def forward(self, x, h, c, n):
        combined = torch.cat([x, h], dim=-1)

        # Clamped exponential gates
        i = torch.exp(torch.clamp(self.W_i(combined), -self.exp_clamp, self.exp_clamp))
        f = torch.exp(torch.clamp(self.W_f(combined), -self.exp_clamp, self.exp_clamp))
        o = torch.sigmoid(self.W_o(combined))
        c_candidate = torch.tanh(self.W_c(combined))

        n = f * n + i
        c = f * c + i * c_candidate
        h = o * torch.tanh(self.ln(c / (n + 1e-6)))

        return h, c, n


class StableSLSTMBlock(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.cell = StableSLSTMCell(d_model, hidden_dim)
        self.norm = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(hidden_dim, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
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


class StableXLSTMForecaster(nn.Module):
    def __init__(
        self,
        num_streams,
        d_model=256,
        hidden_dim=256,
        num_layers=4,
        horizon=90,
        dropout=0.1,
    ):
        super().__init__()
        self.num_streams = num_streams
        self.horizon = horizon

        self.input_proj = nn.Linear(num_streams, d_model)
        self.layers = nn.ModuleList([
            StableSLSTMBlock(d_model, hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon * num_streams),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
            h = self.dropout(h)
        h = h[:, -1]
        out = self.output_proj(h)
        return out.reshape(-1, self.horizon, self.num_streams)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data (same as experiment_02_architecture.py)
    print("Loading data...")
    cat_df = load_categories(frozen=True)
    index_cols = sorted([c for c in cat_df.columns if 'Index' in c or 'index' in c])
    index_cols = [c for c in index_cols if '_year_ago' not in c and '_yoy' not in c]

    data = cat_df[index_cols].values.astype(np.float64)
    dates = cat_df["date"].values
    df_clean = pd.DataFrame(data, columns=index_cols).ffill().bfill()
    data = df_clean.values
    data_norm, means, stds = normalize_streams(data)

    train_end = np.searchsorted(dates, np.datetime64("2023-01-01"))
    val_end = np.searchsorted(dates, np.datetime64("2024-01-01"))
    train_data = data_norm[:train_end]
    val_data = data_norm[:val_end]

    horizon = 90
    context_len = 512

    train_ds = TimeSeriesDataset(train_data, context_len, horizon, stride=7)
    val_ds = TimeSeriesDataset(val_data[train_end - context_len:], context_len, horizon, stride=7)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    num_streams = len(index_cols)
    print(f"Streams: {num_streams}, Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Build model
    model = StableXLSTMForecaster(
        num_streams=num_streams,
        d_model=256,
        hidden_dim=256,
        num_layers=4,
        horizon=horizon,
        dropout=0.1,
    )
    print(f"xLSTM (stable) params: {model.count_params():,}")

    # Train with lower LR and higher grad clip tolerance
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=100,
        lr=3e-4,  # Lower LR for exponential gates
        patience=15,
        model_name="xlstm",
    )

    # Save checkpoint
    ckpt_path = RESULTS_DIR / "xlstm_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"xlstm_results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump({
            "num_params": model.count_params(),
            "training_time": result["training_time"],
            "best_metrics": result["best_metrics"],
        }, f, indent=2, default=str)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
