"""Reversible Instance Normalization (RevIN) for time series.

Per-instance normalization that normalizes each input window by its own
mean/std, and denormalizes predictions using those same statistics.
This prevents the model from learning to predict the global mean.

Based on: Kim et al., "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift", ICLR 2022.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Normalize at input (per-window), denormalize at output.
    The model learns in a scale-invariant space, but predictions
    are restored to the original scale.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        # These are set during forward pass, not learnable
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, num_features]
            mode: 'norm' to normalize input, 'denorm' to denormalize output
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor):
        """Compute per-instance mean and std. Detached — no gradient."""
        # x: [batch, seq_len, features]
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x
