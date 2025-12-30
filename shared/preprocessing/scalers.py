"""Scaler utilities for feature normalization."""

from typing import Literal, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class StandardScalerWrapper:
    """Wrapper for sklearn scalers with fit/transform interface."""
    
    def __init__(self, scaler_type: Literal["standard", "minmax", "robust"] = "standard"):
        """
        Initialize scaler.
        
        Args:
            scaler_type: Type of scaler to use
        """
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit(self, X: np.ndarray) -> "StandardScalerWrapper":
        """Fit the scaler."""
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        return self.scaler.inverse_transform(X)


def get_scaler(
    scaler_type: Literal["standard", "minmax", "robust"] = "standard"
) -> StandardScalerWrapper:
    """
    Get a scaler instance.
    
    Args:
        scaler_type: Type of scaler
        
    Returns:
        Scaler instance
    """
    return StandardScalerWrapper(scaler_type)

