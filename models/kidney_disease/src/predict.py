"""Inference entrypoint for kidney disease prediction model."""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from shared.preprocessing.scalers import get_scaler
from shared.utils.logging_utils import setup_logger
from shared.utils.model_loader import load_model

from .model import KidneyDiseaseModel
from .schema import KidneyDiseaseInput, KidneyDiseaseOutput

logger = setup_logger(__name__)

_model_instance: Optional[KidneyDiseaseModel] = None
_scaler_instance = None


def _load_model_if_needed(model_dir: Optional[Path] = None) -> tuple[KidneyDiseaseModel, any]:
    global _model_instance, _scaler_instance
    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "artifacts"
    if _model_instance is None:
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}.")
        _model_instance = KidneyDiseaseModel(model_path=model_path)
        logger.info(f"Loaded model from {model_path}")
        scaler_path = model_dir / "scaler.joblib"
        if scaler_path.exists():
            scaler = load_model(scaler_path)
            scaler_wrapper = get_scaler("standard")
            scaler_wrapper.scaler = scaler
            _scaler_instance = scaler_wrapper
        else:
            _scaler_instance = get_scaler("standard")
    return _model_instance, _scaler_instance


def predict(input_data: KidneyDiseaseInput, model_dir: Optional[Path] = None) -> dict:
    """Predict kidney disease status."""
    try:
        model, scaler = _load_model_if_needed(model_dir)
        features = np.array([[
            input_data.age, input_data.blood_pressure, input_data.specific_gravity,
            input_data.albumin, input_data.sugar, input_data.red_blood_cells,
            input_data.pus_cell, input_data.pus_cell_clumps, input_data.bacteria,
            input_data.blood_glucose_random, input_data.blood_urea, input_data.serum_creatinine,
            input_data.sodium, input_data.potassium, input_data.haemoglobin,
            input_data.packed_cell_volume, input_data.white_blood_cell_count,
            input_data.red_blood_cell_count, input_data.hypertension,
            input_data.diabetes_mellitus, input_data.coronary_artery_disease,
            input_data.appetite, input_data.peda_edema, input_data.aanemia,
        ]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        probability = float(probabilities[1])
        confidence = float(2 * abs(probability - 0.5))
        
        registry_path = (model_dir or Path(__file__).parent.parent / "artifacts") / "registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
            model_name = registry.get("model_name", "kidney_disease_classifier")
            model_version = registry.get("model_version", "1.0.0")
            known_limitations = registry.get("known_limitations", [])
        else:
            model_name = "kidney_disease_classifier"
            model_version = "1.0.0"
            known_limitations = ["Trained on synthetic data"]
        
        warning = None
        if confidence < 0.7:
            warning = f"Low confidence prediction (confidence={confidence:.2f})."
        
        output = KidneyDiseaseOutput(
            prediction=int(prediction),
            probability=probability,
            confidence=confidence,
            model_name=model_name,
            model_version=model_version,
            known_limitations=known_limitations,
            warning=warning,
        )
        return output.model_dump()
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {
            "prediction": None,
            "probability": None,
            "confidence": 0.0,
            "model_name": "kidney_disease_classifier",
            "model_version": "1.0.0",
            "known_limitations": ["Trained on synthetic data"],
            "warning": f"Prediction failed: {str(e)}",
            "error": str(e),
        }

