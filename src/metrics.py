"""Evaluation metrics for Kairos experiments."""

import numpy as np
from scipy import stats


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    training: np.ndarray,
    seasonality: int = 365,
) -> float:
    """Mean Absolute Scaled Error.

    Divides MAE by the MAE of a seasonal naive baseline (predicting
    last year's value). MASE < 1 means you beat the naive baseline.

    Args:
        actual: Ground truth test values.
        predicted: Model predictions.
        training: Historical training values (for naive baseline).
        seasonality: Seasonal period (365 for daily economic data).
    """
    forecast_mae = mae(actual, predicted)

    naive_errors = np.abs(
        training[seasonality:] - training[:-seasonality]
    )
    naive_mae = np.mean(naive_errors)

    if naive_mae == 0:
        return np.inf
    return forecast_mae / naive_mae


def crps_gaussian(
    actual: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Continuous Ranked Probability Score for Gaussian predictions.

    Scores the predicted distribution (mu, sigma) against actual values.
    Lower is better.

    Args:
        actual: Ground truth values.
        mu: Predicted means.
        sigma: Predicted standard deviations.
    """
    z = (actual - mu) / sigma
    crps_values = sigma * (
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )
    return np.mean(crps_values)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of times the model predicts the correct direction of change.

    Args:
        actual: Ground truth values (changes or levels).
        predicted: Predicted values (changes or levels).

    Returns:
        Float between 0 and 1.
    """
    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    return np.mean(actual_direction == predicted_direction)


def evaluate_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    training: np.ndarray,
    predicted_mu: np.ndarray | None = None,
    predicted_sigma: np.ndarray | None = None,
) -> dict:
    """Run all metrics on a forecast.

    Returns:
        Dict with mae, rmse, mase, directional_accuracy, and optionally crps.
    """
    results = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mase": mase(actual, predicted, training),
        "directional_accuracy": directional_accuracy(actual, predicted),
    }

    if predicted_mu is not None and predicted_sigma is not None:
        results["crps"] = crps_gaussian(actual, predicted_mu, predicted_sigma)

    return results
