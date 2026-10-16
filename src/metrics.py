"""Regression metrics used consistently by classical and PyTorch models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)


def price_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
    return {
        "rmsle": float(
            np.sqrt(mean_squared_log_error(actual_values, predicted_values))
        ),
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)),
    }


def from_log_predictions(log_predictions: np.ndarray) -> np.ndarray:
    """Convert log1p predictions back to non-negative price predictions."""

    return np.clip(np.expm1(np.asarray(log_predictions, dtype=float)), 0.0, None)
