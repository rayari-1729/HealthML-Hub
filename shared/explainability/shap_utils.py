"""SHAP utilities for model explainability."""

from typing import Any, Dict, List, Optional

import numpy as np


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """
    Compute SHAP values for model predictions.
    
    Note: This is a placeholder that returns feature importance.
    For production use, install shap library and implement properly.
    
    Args:
        model: Trained model
        X: Input features
        feature_names: Optional feature names
        max_samples: Maximum samples for SHAP computation
        
    Returns:
        Dictionary with SHAP values and feature importance
    """
    # Placeholder implementation
    # In production, use: import shap; explainer = shap.Explainer(model); shap_values = explainer(X)
    
    n_samples = min(len(X), max_samples)
    X_sample = X[:n_samples]
    
    # Fallback: Use feature importance if available
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0]).tolist() if model.coef_.ndim > 1 else np.abs(model.coef_).tolist()
    else:
        importance = [1.0 / X.shape[1]] * X.shape[1]
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]
    
    return {
        "feature_importance": dict(zip(feature_names, importance)),
        "shap_values_available": False,
        "note": "Install 'shap' package for full SHAP support",
    }


def explain_prediction(
    model: Any,
    X: np.ndarray,
    prediction: Any,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Explain a single prediction.
    
    Args:
        model: Trained model
        X: Input features (single sample)
        prediction: Model prediction
        feature_names: Optional feature names
        
    Returns:
        Explanation dictionary
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    shap_info = compute_shap_values(model, X, feature_names, max_samples=1)
    
    return {
        "prediction": float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
        "feature_importance": shap_info["feature_importance"],
        "top_features": sorted(
            shap_info["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5],
    }

