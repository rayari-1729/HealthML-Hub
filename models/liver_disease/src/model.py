"""Liver disease prediction model definition."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from shared.utils.model_loader import load_model


class LiverDiseaseModel:
    """Liver disease prediction model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model_path = model_path
        self.feature_names = [
            "age", "gender", "total_bilirubin", "direct_bilirubin",
            "alkaline_phosphotase", "alamine_aminotransferase",
            "aspartate_aminotransferase", "total_proteins", "albumin",
            "albumin_globulin_ratio",
        ]
        if model_path and model_path.exists():
            self.load(model_path)
    
    def load(self, model_path: Path) -> None:
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LiverDiseaseModel":
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_names(self) -> list:
        return self.feature_names.copy()

