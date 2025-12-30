"""Inference entrypoint for diabetes prediction model."""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from shared.preprocessing.scalers import get_scaler
from shared.utils.logging_utils import setup_logger
from shared.utils.model_loader import load_model

from .model import DiabetesModel
from .schema import DiabetesInput, DiabetesOutput

logger = setup_logger(__name__)

# Global model instance (loaded on first use)
_model_instance: Optional[DiabetesModel] = None
_scaler_instance = None


def _load_model_if_needed(model_dir: Optional[Path] = None) -> tuple[DiabetesModel, any]:
    """
    Load model and scaler if not already loaded.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Tuple of (model, scaler)
    """
    global _model_instance, _scaler_instance
    
    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "artifacts"
    
    if _model_instance is None:
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first using train.py"
            )
        
        _model_instance = DiabetesModel(model_path=model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load scaler
        scaler_path = model_dir / "scaler.joblib"
        if scaler_path.exists():
            scaler = load_model(scaler_path)
            scaler_wrapper = get_scaler("standard")
            scaler_wrapper.scaler = scaler
            _scaler_instance = scaler_wrapper
        else:
            logger.warning("Scaler not found, using default")
            _scaler_instance = get_scaler("standard")
    
    return _model_instance, _scaler_instance


def predict(input_data: DiabetesInput, model_dir: Optional[Path] = None) -> dict:
    """
    Predict diabetes status from input data.
    
    This is the SINGLE public inference entrypoint for the diabetes model.
    
    Args:
        input_data: DiabetesInput instance with patient data
        model_dir: Optional directory containing model artifacts
        
    Returns:
        Dictionary with prediction, confidence, and metadata
    """
    try:
        # Load model
        model, scaler = _load_model_if_needed(model_dir)
        
        # Prepare features
        features = np.array([[
            input_data.pregnancies,
            input_data.glucose,
            input_data.blood_pressure,
            input_data.skin_thickness,
            input_data.insulin,
            input_data.bmi,
            input_data.diabetes_pedigree,
            input_data.age,
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        probability = float(probabilities[1])  # Probability of diabetes
        
        # Calculate confidence (based on probability distance from 0.5)
        confidence = float(2 * abs(probability - 0.5))
        
        # Load registry for metadata
        registry_path = (model_dir or Path(__file__).parent.parent / "artifacts") / "registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
            model_name = registry.get("model_name", "diabetes_classifier")
            model_version = registry.get("model_version", "1.0.0")
            known_limitations = registry.get("known_limitations", [])
        else:
            model_name = "diabetes_classifier"
            model_version = "1.0.0"
            known_limitations = [
                "Trained on synthetic data for demonstration",
                "May not generalize to all populations",
            ]
        
        # Generate warning if confidence is low
        warning = None
        if confidence < 0.7:
            warning = (
                f"Low confidence prediction (confidence={confidence:.2f}). "
                "Consider additional clinical evaluation."
            )
        
        # Build output
        output = DiabetesOutput(
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
        # Return abstention response
        return {
            "prediction": None,
            "probability": None,
            "confidence": 0.0,
            "model_name": "diabetes_classifier",
            "model_version": "1.0.0",
            "known_limitations": [
                "Trained on synthetic data for demonstration",
                "May not generalize to all populations",
            ],
            "warning": f"Prediction failed: {str(e)}",
            "error": str(e),
        }

