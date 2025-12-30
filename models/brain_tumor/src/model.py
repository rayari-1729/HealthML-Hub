"""Brain tumor detection model definition (Deep Learning)."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neural_network import MLPClassifier

from shared.utils.model_loader import load_model


class BrainTumorModel:
    """
    Brain tumor detection model using Multi-Layer Perceptron (Deep Learning).
    
    Note: In production, this would use a CNN for direct image input.
    This implementation uses extracted features with a neural network.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=32,
            learning_rate="adaptive",
            max_iter=500,
            random_state=42,
        )
        self.model_path = model_path
        self.feature_names = [
            "mean_intensity", "std_intensity", "contrast", "energy",
            "homogeneity", "correlation", "entropy", "area", "perimeter",
            "compactness", "eccentricity", "solidity", "extent",
            "aspect_ratio", "equivalent_diameter",
        ]
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def load(self, model_path: Path) -> None:
        """Load a pre-trained model."""
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BrainTumorModel":
        """Train the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict brain tumor presence."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict brain tumor probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_names(self) -> list:
        """Get feature names."""
        return self.feature_names.copy()

