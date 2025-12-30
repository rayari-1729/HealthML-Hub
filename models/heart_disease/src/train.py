"""Training script for heart disease model."""

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

from .model import HeartDiseaseModel

logger = setup_logger(__name__)


def load_training_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load training data (placeholder with synthetic data)."""
    np.random.seed(42)
    n_samples = 1000
    
    X = np.random.rand(n_samples, 13)
    X[:, 0] = np.random.randint(30, 80, n_samples)  # age
    X[:, 1] = np.random.randint(0, 2, n_samples)  # sex
    X[:, 2] = np.random.randint(0, 4, n_samples)  # cp
    X[:, 3] = np.random.uniform(90, 200, n_samples)  # trestbps
    X[:, 4] = np.random.uniform(100, 400, n_samples)  # chol
    X[:, 5] = np.random.randint(0, 2, n_samples)  # fbs
    X[:, 6] = np.random.randint(0, 3, n_samples)  # restecg
    X[:, 7] = np.random.uniform(70, 200, n_samples)  # thalach
    X[:, 8] = np.random.randint(0, 2, n_samples)  # exang
    X[:, 9] = np.random.uniform(0, 6, n_samples)  # oldpeak
    X[:, 10] = np.random.randint(0, 3, n_samples)  # slope
    X[:, 11] = np.random.randint(0, 4, n_samples)  # ca
    X[:, 12] = np.random.randint(0, 4, n_samples)  # thal
    
    # Generate labels
    y = (
        (X[:, 3] > 140).astype(int) +
        (X[:, 4] > 240).astype(int) +
        (X[:, 8] == 1).astype(int) +
        (X[:, 9] > 2).astype(int)
    ) > 1
    y = y.astype(int)
    
    logger.info(f"Loaded {len(X)} training samples")
    return X, y


def train_heart_disease_model(
    data_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> HeartDiseaseModel:
    """Train heart disease prediction model."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    logger.info("Loading training data...")
    X, y = load_training_data(data_path) if data_path else load_training_data(Path("dummy"))
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = get_scaler("standard")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    logger.info("Training model...")
    model = HeartDiseaseModel()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    metrics = calculate_binary_metrics(y_val, y_pred, y_proba)
    
    logger.info(f"Validation metrics: {metrics}")
    
    model_path = output_dir / "model.joblib"
    scaler_path = output_dir / "scaler.joblib"
    save_model(model.model, model_path)
    save_model(scaler.scaler, scaler_path)
    
    registry = {
        "model_name": "heart_disease_classifier",
        "model_version": "1.0.0",
        "metrics": metrics,
        "n_samples": len(X_train),
        "feature_names": model.get_feature_names(),
        "known_limitations": [
            "Trained on synthetic data for demonstration",
            "In production, train on real Cleveland Heart Disease Dataset",
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
    
    train_heart_disease_model(
        data_path=args.data_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
    )

