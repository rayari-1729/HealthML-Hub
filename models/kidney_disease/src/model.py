"""Kidney disease prediction model definition."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from shared.utils.model_loader import load_model


class KidneyDiseaseModel:
    """Kidney disease prediction model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model_path = model_path
        self.feature_names = [
            "age", "blood_pressure", "specific_gravity", "albumin", "sugar",
            "red_blood_cells", "pus_cell", "pus_cell_clumps", "bacteria",
            "blood_glucose_random", "blood_urea", "serum_creatinine", "sodium",
            "potassium", "haemoglobin", "packed_cell_volume", "white_blood_cell_count",
            "red_blood_cell_count", "hypertension", "diabetes_mellitus",
            "coronary_artery_disease", "appetite", "peda_edema", "aanemia",
        ]
        if model_path and model_path.exists():
            self.load(model_path)
    
    def load(self, model_path: Path) -> None:
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KidneyDiseaseModel":
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_names(self) -> list:
        return self.feature_names.copy()

