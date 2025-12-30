"""Model loading and saving utilities."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


def save_model(
    model: Any,
    model_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a model to disk with optional metadata.
    
    Args:
        model: Model object to save
        model_path: Path where model will be saved
        metadata: Optional metadata dictionary
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if hasattr(model, "save"):
        model.save(str(model_path))
    else:
        joblib.dump(model, model_path)
    
    # Save metadata if provided
    if metadata:
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_model(model_path: Path) -> Any:
    """
    Load a model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model object
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Try joblib first (scikit-learn)
    try:
        return joblib.load(model_path)
    except Exception:
        pass
    
    # Try pickle
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    
    # If model has custom load method
    raise ValueError(f"Could not load model from {model_path}")

