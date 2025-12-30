"""Evaluation script for diabetes model."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from shared.evaluation.metrics import calculate_binary_metrics
from shared.preprocessing.scalers import get_scaler
from shared.utils.logging_utils import setup_logger
from shared.utils.model_loader import load_model

from .model import DiabetesModel

logger = setup_logger(__name__)


def evaluate_diabetes_model(
    model_path: Path,
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Evaluate diabetes prediction model.
    
    Args:
        model_path: Path to trained model
        data_path: Path to evaluation data
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    if output_dir is None:
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = DiabetesModel(model_path=model_path)
    
    # Load scaler
    scaler_path = model_path.parent / "scaler.joblib"
    if scaler_path.exists():
        scaler = load_model(scaler_path)
        scaler_wrapper = get_scaler("standard")
        scaler_wrapper.scaler = scaler
    else:
        logger.warning("Scaler not found, using default")
        scaler_wrapper = get_scaler("standard")
    
    # Load test data
    # Placeholder: Generate synthetic test data
    np.random.seed(123)
    n_samples = 200
    X_test = np.random.rand(n_samples, 8)
    X_test[:, 0] = np.random.randint(0, 10, n_samples)
    X_test[:, 1] = np.random.uniform(70, 200, n_samples)
    X_test[:, 2] = np.random.uniform(50, 100, n_samples)
    X_test[:, 3] = np.random.uniform(10, 50, n_samples)
    X_test[:, 4] = np.random.uniform(0, 300, n_samples)
    X_test[:, 5] = np.random.uniform(20, 50, n_samples)
    X_test[:, 6] = np.random.uniform(0.1, 2.5, n_samples)
    X_test[:, 7] = np.random.randint(20, 80, n_samples)
    
    y_test = (
        (X_test[:, 1] > 140).astype(int) +
        (X_test[:, 5] > 30).astype(int) +
        (X_test[:, 7] > 50).astype(int)
    ) > 1
    y_test = y_test.astype(int)
    
    # Scale features
    X_test_scaled = scaler_wrapper.transform(X_test)
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = calculate_binary_metrics(y_test, y_pred, y_proba)
    
    logger.info(f"Test metrics: {metrics}")
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    return metrics


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    
    evaluate_diabetes_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )

