"""Composite loss functions for economic time series forecasting.

Based on FinCast's PQ-Loss: Huber + Trend Consistency + Quantile.
Addresses the mean-reversion problem caused by pure MSE training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    """Huber loss — quadratic for small errors, linear for large ones.

    More robust to outliers than MSE. Critical for economic data
    where COVID-era observations are extreme outliers.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


class TrendConsistencyLoss(nn.Module):
    """Penalizes when the model predicts the wrong direction of change.

    Computes first-order differences of both predictions and actuals,
    then penalizes mismatch. This forces the model to get the DIRECTION
    right, not just the absolute level.

    From FinCast (Equation 23):
    L_trend = (1/(H-1)) * sum[(pred[t] - pred[t-1]) - (actual[t] - actual[t-1])]^2
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [batch, horizon, num_streams]
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return F.mse_loss(pred_diff, target_diff)


class DirectionalLoss(nn.Module):
    """Soft directional penalty using sigmoid approximation.

    Penalizes when predicted direction differs from actual direction.
    Uses a smooth approximation of sign() for differentiability.
    """

    def __init__(self, temperature: float = 10.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]

        # Soft sign: sigmoid(temp * x) ≈ step function
        pred_dir = torch.sigmoid(self.temperature * pred_diff)
        target_dir = torch.sigmoid(self.temperature * target_diff)

        return F.binary_cross_entropy(pred_dir, target_dir)


class CompositeLoss(nn.Module):
    """Combined loss: Huber + Trend Consistency + Directional.

    This is the loss function that should fix the 21% directional accuracy.

    Args:
        lambda_trend: weight for trend consistency loss (default 0.5)
        lambda_direction: weight for directional loss (default 0.3)
        huber_delta: threshold for Huber loss transition (default 1.0)
    """

    def __init__(
        self,
        lambda_trend: float = 0.5,
        lambda_direction: float = 0.3,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.huber = HuberLoss(delta=huber_delta)
        self.trend = TrendConsistencyLoss()
        self.direction = DirectionalLoss()
        self.lambda_trend = lambda_trend
        self.lambda_direction = lambda_direction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_huber = self.huber(pred, target)
        l_trend = self.trend(pred, target)
        l_direction = self.direction(pred, target)

        total = l_huber + self.lambda_trend * l_trend + self.lambda_direction * l_direction
        return total
