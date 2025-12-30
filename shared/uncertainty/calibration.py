"""Confidence calibration utilities."""

from typing import Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def calibrate_confidence(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> Any:
    """
    Calibrate model confidence scores.
    
    Args:
        model: Uncalibrated model
        X_val: Validation features
        y_val: Validation labels
        method: Calibration method ("isotonic" or "sigmoid")
        
    Returns:
        Calibrated model
    """
    if not hasattr(model, "predict_proba"):
        return model
    
    calibrated = CalibratedClassifierCV(model, method=method, cv=3)
    calibrated.fit(X_val, y_val)
    return calibrated


def compute_confidence_interval(
    predictions: np.ndarray,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute confidence interval for predictions.
    
    Args:
        predictions: Array of predictions
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence_level
    lower = np.percentile(predictions, (alpha / 2) * 100)
    upper = np.percentile(predictions, (1 - alpha / 2) * 100)
    return float(lower), float(upper)

