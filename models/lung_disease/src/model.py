"""Lung disease prediction model definition."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from shared.utils.model_loader import load_model


class LungDiseaseModel:
    """Lung disease prediction model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model_path = model_path
        self.feature_names = [
            "age", "gender", "air_pollution", "alcohol_use", "dust_allergy",
            "occupational_hazards", "genetic_risk", "chronic_lung_disease",
            "balanced_diet", "obesity", "smoking", "passive_smoking",
            "chest_pain", "coughing_blood", "fatigue", "weight_loss",
            "shortness_of_breath", "wheezing", "swallowing_difficulty",
            "clubbing_of_finger_nails", "frequent_cold", "dry_cough", "snoring",
        ]
        if model_path and model_path.exists():
            self.load(model_path)
    
    def load(self, model_path: Path) -> None:
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LungDiseaseModel":
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_names(self) -> list:
        return self.feature_names.copy()

