"""Decoder-only Transformer for time series forecasting.

Based on the architecture pattern that TimesFM, Moirai 2.0, and Sundial converged on.
Patches input, applies causal self-attention, predicts future patches.
"""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, d_model: int, num_streams: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * num_streams, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, num_streams]
        b, t, c = x.shape
        num_patches = t // self.patch_size
        x = x[:, :num_patches * self.patch_size]
        x = x.reshape(b, num_patches, self.patch_size * c)
        return self.proj(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerForecaster(nn.Module):
    """Decoder-only transformer for multivariate time series forecasting."""

    def __init__(
        self,
        num_streams: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = 16,
        horizon: int = 90,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_streams = num_streams
        self.horizon = horizon
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(patch_size, d_model, num_streams)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, horizon * num_streams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, context_len, num_streams]
        h = self.patch_embed(x)
        seq_len = h.size(1)
        h = h + self.pos_embed[:, :seq_len]

        for block in self.blocks:
            h = block(h)

        h = self.ln_final(h)
        h = h[:, -1]  # Take last patch representation
        out = self.output_proj(h)
        return out.reshape(-1, self.horizon, self.num_streams)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
