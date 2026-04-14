"""Sliding window dataset for time series training."""

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Creates sliding windows from multivariate time series for forecasting.

    Each sample is (context, target) where:
    - context: [context_len, num_streams] tensor of historical values
    - target: [horizon, num_streams] tensor of future values to predict
    """

    def __init__(
        self,
        data: np.ndarray,
        context_len: int = 512,
        horizon: int = 90,
        stride: int = 1,
    ):
        """
        Args:
            data: numpy array of shape [time_steps, num_streams]
            context_len: number of historical timesteps as input
            horizon: number of future timesteps to predict
            stride: step size between consecutive windows
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride
        self.total_len = context_len + horizon

        self.num_windows = max(0, (len(data) - self.total_len) // stride + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        context = self.data[start : start + self.context_len]
        target = self.data[start + self.context_len : start + self.total_len]
        return context, target


class UnivariateSliceDataset(Dataset):
    """Wraps a multivariate dataset to serve individual stream slices.

    Each sample picks one stream from one window, producing:
    - context: [context_len] single-stream history
    - target: [horizon] single-stream future
    - stream_id: int identifying which stream

    This is how univariate models (Chronos, TimesFM) see data.
    Multivariate models use TimeSeriesDataset directly.
    """

    def __init__(
        self,
        data: np.ndarray,
        context_len: int = 512,
        horizon: int = 90,
        stride: int = 7,
    ):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride
        self.total_len = context_len + horizon
        self.num_streams = data.shape[1]

        self.num_windows = max(0, (len(data) - self.total_len) // stride + 1)

    def __len__(self):
        return self.num_windows * self.num_streams

    def __getitem__(self, idx):
        window_idx = idx // self.num_streams
        stream_idx = idx % self.num_streams

        start = window_idx * self.stride
        context = self.data[start : start + self.context_len, stream_idx]
        target = self.data[start + self.context_len : start + self.total_len, stream_idx]
        return context, target, stream_idx


def normalize_streams(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-stream z-score normalization.

    Args:
        data: [time_steps, num_streams]

    Returns:
        (normalized_data, means, stds)
    """
    means = np.nanmean(data, axis=0, keepdims=True)
    stds = np.nanstd(data, axis=0, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)
    normalized = (data - means) / stds
    return normalized, means.squeeze(), stds.squeeze()
