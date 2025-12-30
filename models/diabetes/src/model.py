"""Diabetes prediction model definition."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from shared.utils.model_loader import load_model


class DiabetesModel:
    """Diabetes prediction model using Random Forest."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize diabetes model.
        
        Args:
            model_path: Optional path to pre-trained model
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        self.model_path = model_path
        self.feature_names = [
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree",
            "age",
        ]
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def load(self, model_path: Path) -> None:
        """Load a pre-trained model."""
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiabetesModel":
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict diabetes status.
        
        Args:
            X: Input features
            
        Returns:
            Predicted classes
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict diabetes probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def get_feature_names(self) -> list:
        """Get feature names."""
        return self.feature_names.copy()

