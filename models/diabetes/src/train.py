"""Training script for diabetes model."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from shared.evaluation.metrics import calculate_binary_metrics
from shared.preprocessing.scalers import get_scaler
from shared.utils.logging_utils import setup_logger
from shared.utils.model_loader import save_model

from .model import DiabetesModel

logger = setup_logger(__name__)


def load_training_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training data.
    
    Note: In production, this would load from a real dataset.
    This is a placeholder that generates synthetic data for demonstration.
    
    Args:
        data_path: Path to training data
        
    Returns:
        Tuple of (X, y)
    """
    # Placeholder: Generate synthetic data for demonstration
    # In production, load from actual dataset
    np.random.seed(42)
    n_samples = 1000
    
    X = np.random.rand(n_samples, 8)
    X[:, 0] = np.random.randint(0, 10, n_samples)  # pregnancies
    X[:, 1] = np.random.uniform(70, 200, n_samples)  # glucose
    X[:, 2] = np.random.uniform(50, 100, n_samples)  # blood_pressure
    X[:, 3] = np.random.uniform(10, 50, n_samples)  # skin_thickness
    X[:, 4] = np.random.uniform(0, 300, n_samples)  # insulin
    X[:, 5] = np.random.uniform(20, 50, n_samples)  # bmi
    X[:, 6] = np.random.uniform(0.1, 2.5, n_samples)  # diabetes_pedigree
    X[:, 7] = np.random.randint(20, 80, n_samples)  # age
    
    # Generate labels (diabetes risk increases with glucose, bmi, age)
    y = (
        (X[:, 1] > 140).astype(int) +
        (X[:, 5] > 30).astype(int) +
        (X[:, 7] > 50).astype(int)
    ) > 1
    y = y.astype(int)
    
    logger.info(f"Loaded {len(X)} training samples")
    return X, y


def train_diabetes_model(
    data_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> DiabetesModel:
    """
    Train diabetes prediction model.
    
    Args:
        data_path: Path to training data
        config_path: Path to config file
        output_dir: Directory to save trained model
        
    Returns:
        Trained model
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    
    # Load config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    logger.info("Loading training data...")
    X, y = load_training_data(data_path) if data_path else load_training_data(Path("dummy"))
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = get_scaler("standard")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    logger.info("Training model...")
    model = DiabetesModel()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    metrics = calculate_binary_metrics(y_val, y_pred, y_proba)
    
    logger.info(f"Validation metrics: {metrics}")
    
    # Save model and scaler
    model_path = output_dir / "model.joblib"
    scaler_path = output_dir / "scaler.joblib"
    save_model(model.model, model_path)
    save_model(scaler.scaler, scaler_path)
    
    # Save metadata
    registry = {
        "model_name": "diabetes_classifier",
        "model_version": "1.0.0",
        "metrics": metrics,
        "n_samples": len(X_train),
        "feature_names": model.get_feature_names(),
        "known_limitations": [
            "Trained on synthetic data for demonstration",
            "In production, train on real Pima Indians Diabetes Dataset",
            "May not generalize to all populations",
        ],
    }
    
    registry_path = output_dir / "registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    return model


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    
    train_diabetes_model(
        data_path=args.data_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
    )

