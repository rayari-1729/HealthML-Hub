"""Evaluation script for heart disease model."""

from pathlib import Path
from typing import Optional

import numpy as np

from shared.evaluation.metrics import calculate_binary_metrics
from shared.preprocessing.scalers import get_scaler
from shared.utils.logging_utils import setup_logger
from shared.utils.model_loader import load_model

from .model import HeartDiseaseModel

logger = setup_logger(__name__)


def evaluate_heart_disease_model(
    model_path: Path,
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Evaluate heart disease prediction model."""
    if output_dir is None:
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {model_path}")
    model = HeartDiseaseModel(model_path=model_path)
    
    scaler_path = model_path.parent / "scaler.joblib"
    if scaler_path.exists():
        scaler = load_model(scaler_path)
        scaler_wrapper = get_scaler("standard")
        scaler_wrapper.scaler = scaler
    else:
        scaler_wrapper = get_scaler("standard")
    
    np.random.seed(123)
    n_samples = 200
    X_test = np.random.rand(n_samples, 13)
    X_test[:, 0] = np.random.randint(30, 80, n_samples)
    X_test[:, 1] = np.random.randint(0, 2, n_samples)
    X_test[:, 2] = np.random.randint(0, 4, n_samples)
    X_test[:, 3] = np.random.uniform(90, 200, n_samples)
    X_test[:, 4] = np.random.uniform(100, 400, n_samples)
    X_test[:, 5] = np.random.randint(0, 2, n_samples)
    X_test[:, 6] = np.random.randint(0, 3, n_samples)
    X_test[:, 7] = np.random.uniform(70, 200, n_samples)
    X_test[:, 8] = np.random.randint(0, 2, n_samples)
    X_test[:, 9] = np.random.uniform(0, 6, n_samples)
    X_test[:, 10] = np.random.randint(0, 3, n_samples)
    X_test[:, 11] = np.random.randint(0, 4, n_samples)
    X_test[:, 12] = np.random.randint(0, 4, n_samples)
    
    y_test = (
        (X_test[:, 3] > 140).astype(int) +
        (X_test[:, 4] > 240).astype(int) +
        (X_test[:, 8] == 1).astype(int) +
        (X_test[:, 9] > 2).astype(int)
    ) > 1
    y_test = y_test.astype(int)
    
    X_test_scaled = scaler_wrapper.transform(X_test)
    
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = calculate_binary_metrics(y_test, y_pred, y_proba)
    
    logger.info(f"Test metrics: {metrics}")
    return metrics

