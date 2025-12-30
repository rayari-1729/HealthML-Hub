"""Breast cancer prediction model definition."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.svm import SVC

from shared.utils.model_loader import load_model


class BreastCancerModel:
    """Breast cancer prediction model using SVM."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = SVC(probability=True, random_state=42, C=1.0, kernel="rbf")
        self.model_path = model_path
        self.feature_names = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean",
            "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        ]
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def load(self, model_path: Path) -> None:
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BreastCancerModel":
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_names(self) -> list:
        return self.feature_names.copy()

