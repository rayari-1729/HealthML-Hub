"""Parkinson's disease prediction model definition."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.svm import SVC

from shared.utils.model_loader import load_model


class ParkinsonsModel:
    """Parkinson's disease prediction model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = SVC(probability=True, random_state=42, C=1.0, kernel="rbf")
        self.model_path = model_path
        self.feature_names = [
            "mdvp_fo_hz", "mdvp_fhi_hz", "mdvp_flo_hz", "mdvp_jitter_percent",
            "mdvp_jitter_abs", "mdvp_rap", "mdvp_ppq", "jitter_ddp",
            "mdvp_shimmer", "mdvp_shimmer_db", "shimmer_apq3", "shimmer_apq5",
            "mdvp_apq", "shimmer_dda", "nhr", "hnr", "rpde", "dfa",
            "spread1", "spread2", "d2", "ppe",
        ]
        if model_path and model_path.exists():
            self.load(model_path)
    
    def load(self, model_path: Path) -> None:
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ParkinsonsModel":
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_names(self) -> list:
        return self.feature_names.copy()

