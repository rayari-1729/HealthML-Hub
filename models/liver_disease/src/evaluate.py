"""Evaluation script for liver disease model."""

from pathlib import Path
from typing import Optional

import numpy as np

from shared.evaluation.metrics import calculate_binary_metrics
from shared.preprocessing.scalers import get_scaler
from shared.utils.logging_utils import setup_logger
from shared.utils.model_loader import load_model

from .model import LiverDiseaseModel

logger = setup_logger(__name__)


def evaluate_liver_disease_model(
    model_path: Path,
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    if output_dir is None:
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {model_path}")
    model = LiverDiseaseModel(model_path=model_path)
    
    scaler_path = model_path.parent / "scaler.joblib"
    if scaler_path.exists():
        scaler = load_model(scaler_path)
        scaler_wrapper = get_scaler("standard")
        scaler_wrapper.scaler = scaler
    else:
        scaler_wrapper = get_scaler("standard")
    
    np.random.seed(123)
    n_samples = 200
    X_test = np.random.rand(n_samples, 10)
    X_test[:, 2] = np.random.uniform(0, 30, n_samples)
    y_test = (X_test[:, 2] > 1.2).astype(int)
    
    X_test_scaled = scaler_wrapper.transform(X_test)
    
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = calculate_binary_metrics(y_test, y_pred, y_proba)
    
    logger.info(f"Test metrics: {metrics}")
    return metrics

