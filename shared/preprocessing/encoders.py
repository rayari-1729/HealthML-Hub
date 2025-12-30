"""Encoder utilities for categorical features."""

from typing import List, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class LabelEncoderWrapper:
    """Wrapper for label encoding with fit/transform interface."""
    
    def __init__(self):
        """Initialize encoder."""
        self.encoder = LabelEncoder()
        self.classes_: Optional[np.ndarray] = None
    
    def fit(self, y: np.ndarray) -> "LabelEncoderWrapper":
        """Fit the encoder."""
        self.encoder.fit(y)
        self.classes_ = self.encoder.classes_
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels."""
        return self.encoder.transform(y)
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform labels."""
        result = self.encoder.fit_transform(y)
        self.classes_ = self.encoder.classes_
        return result
    
    def inverse_transform(self, y: np.ndarray) -> List[str]:
        """Inverse transform labels."""
        return self.encoder.inverse_transform(y)


def get_encoder(encoder_type: str = "label") -> LabelEncoderWrapper:
    """
    Get an encoder instance.
    
    Args:
        encoder_type: Type of encoder (currently only "label" supported)
        
    Returns:
        Encoder instance
    """
    if encoder_type == "label":
        return LabelEncoderWrapper()
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

