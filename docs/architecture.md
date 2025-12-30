# HealthML-Hub Architecture

## Model-Only Architecture

HealthML-Hub follows a **model-only** architecture pattern. This means the repository contains **ONLY** machine learning and deep learning model implementations. It does not contain agent logic, UI components, or clinical decision-making systems.

## Core Principles

### 1. Single Responsibility

Each component has a single, well-defined responsibility:
- **Models**: Predict disease status from input features
- **Shared Utilities**: Reusable preprocessing, evaluation, explainability tools
- **Pipelines**: Batch training, evaluation, and inference automation

### 2. Stateless Inference

All models are **stateless**:
- No internal state between predictions
- No file writes during inference
- Deterministic outputs for identical inputs
- Safe for concurrent use

### 3. Stable Schemas

Every model uses **Pydantic schemas** for:
- Input validation
- Output consistency
- Type safety
- Documentation

### 4. Single Entry Point

Each disease model exposes **exactly one** public function:
```python
predict(input_data: DiseaseInput) -> dict
```

## Repository Structure

```
HealthML-Hub/
├── models/              # Disease-specific models
│   ├── diabetes/
│   ├── heart_disease/
│   └── ...
├── shared/              # Cross-disease utilities
│   ├── preprocessing/
│   ├── evaluation/
│   ├── explainability/
│   └── uncertainty/
├── pipelines/            # Automation scripts
│   ├── training/
│   ├── evaluation/
│   └── inference/
└── docs/                # Documentation
```

## Standard Disease Model Structure

Every disease model follows this structure:

```
models/<disease_name>/
├── README.md                 # Model card
├── src/
│   ├── schema.py             # Input/output schemas
│   ├── model.py              # Model definition
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── predict.py            # Inference entrypoint
├── configs/
│   └── default.yaml          # Hyperparameters
└── artifacts/
    └── registry.json         # Model metadata
```

## Model Definition Pattern

### Schema (schema.py)

```python
class DiseaseInput(BaseModel):
    feature1: float = Field(..., ge=0, le=100)
    feature2: int = Field(..., ge=0, le=10)

class DiseaseOutput(BaseModel):
    prediction: int
    probability: float
    confidence: float
    model_name: str
    model_version: str
    known_limitations: list
    warning: Optional[str]
```

### Model (model.py)

```python
class DiseaseModel:
    def __init__(self, model_path: Optional[Path] = None):
        self.model = SomeClassifier()
        # ...
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
```

### Inference (predict.py)

```python
def predict(input_data: DiseaseInput, model_dir: Optional[Path] = None) -> dict:
    # Load model
    # Preprocess input
    # Make prediction
    # Calculate confidence
    # Return standardized output
```

## Shared Utilities

### Preprocessing

- **Scalers**: StandardScaler, MinMaxScaler, RobustScaler
- **Encoders**: LabelEncoder, OneHotEncoder
- Consistent interface across all models

### Evaluation

- **Binary Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Multi-class Metrics**: Weighted averages
- **Regression Metrics**: MSE, RMSE, MAE, R²

### Explainability

- **SHAP Integration**: Feature importance
- **Prediction Explanations**: Top contributing features
- Placeholder for full SHAP support (requires `shap` package)

### Uncertainty

- **Confidence Calibration**: Isotonic, Sigmoid
- **Confidence Intervals**: Percentile-based

## Why Stable Schemas Matter

### For Consumers

1. **Type Safety**: Pydantic validates inputs at runtime
2. **Documentation**: Schemas serve as API documentation
3. **Consistency**: Same output format across all models
4. **Error Handling**: Clear validation errors

### For Maintainers

1. **Versioning**: Schema changes indicate breaking changes
2. **Testing**: Easy to generate test cases
3. **Integration**: Clear contracts for external systems

## Separation from Agents and UI

### What's NOT in HealthML-Hub

- **Agent Logic**: No reasoning, no orchestration
- **UI Components**: No Streamlit, no web interfaces
- **Clinical Pathways**: No decision trees
- **Patient Management**: No state tracking

### What IS in HealthML-Hub

- **Pure Models**: Input → Prediction → Output
- **Utilities**: Preprocessing, evaluation, explainability
- **Pipelines**: Batch automation scripts

## Integration Pattern

### For AI Doctor Systems

```python
# In AI Doctor system (separate repo)
from healthml_hub.models.diabetes.src.predict import predict
from healthml_hub.models.diabetes.src.schema import DiabetesInput

# Get patient data
patient_data = get_patient_data()

# Call HealthML-Hub model
diabetes_input = DiabetesInput(**patient_data)
diabetes_result = predict(diabetes_input)

# Use result in AI Doctor logic
if diabetes_result["prediction"] == 1 and diabetes_result["confidence"] > 0.8:
    # AI Doctor makes decision based on model output
    recommend_further_testing()
```

### For Other Systems

HealthML-Hub models can be used by:
- Research systems
- Clinical decision support tools (with proper validation)
- Educational platforms
- Benchmarking systems

## Design Decisions

### Why Pydantic?

- Runtime validation
- Type hints
- JSON schema generation
- Clear error messages

### Why Single Entry Point?

- Simple integration
- Consistent interface
- Easy to test
- Clear API contract

### Why Stateless?

- Thread-safe
- Scalable
- Testable
- Predictable

## Future Architecture Considerations

### Model Versioning

- Semantic versioning (1.0.0, 1.1.0, 2.0.0)
- Registry tracking
- Backward compatibility

### Model Registry

- Centralized metadata
- Performance metrics
- Known limitations
- Training information

### Model Serving

- REST API layer (separate repo)
- gRPC support (optional)
- Batch inference pipelines

## Conclusion

HealthML-Hub's architecture is designed for:
- **Simplicity**: Easy to understand and use
- **Extensibility**: Easy to add new models
- **Reliability**: Stable, tested, production-ready
- **Independence**: No dependencies on external systems

This architecture enables HealthML-Hub to serve as a solid foundation for future AI Doctor systems while remaining useful as a standalone medical model library.

