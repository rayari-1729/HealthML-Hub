"""Heart disease prediction model definition."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from shared.utils.model_loader import load_model


class HeartDiseaseModel:
    """Heart disease prediction model using Gradient Boosting."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize heart disease model.
        
        Args:
            model_path: Optional path to pre-trained model
        """
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.model_path = model_path
        self.feature_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def load(self, model_path: Path) -> None:
        """Load a pre-trained model."""
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HeartDiseaseModel":
        """Train the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict heart disease status."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict heart disease probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_names(self) -> list:
        """Get feature names."""
        return self.feature_names.copy()

