# HealthML-Hub

### A Model-Only Medical Machine Learning Foundation for Scalable Clinical AI Systems

---

## About the Project

In modern healthcare, the demand for **accurate, reusable, and extensible machine learning models** is rapidly increasing. While many healthcare AI projects focus on end-to-end applications, dashboards, or conversational systems, they often tightly couple **model logic, reasoning, and user interaction**, making them difficult to reuse, validate, or extend.

**HealthML-Hub** is designed to solve this problem.

HealthML-Hub is a **model-only medical machine learning repository** that provides **disease-specific predictive models** as clean, independent, and reusable components. Each model is implemented as a standalone medical intelligence unit with a **stable inference contract**, making it suitable for integration into larger systems such as clinical decision-support tools, research platforms, or future agentic AI Doctor systems.

The repository focuses exclusively on:

- Disease-specific ML and DL models
- Deterministic, schema-driven inference
- Reproducible training and evaluation
- Clear model boundaries and limitations

By intentionally excluding reasoning, orchestration, and UI logic, HealthML-Hub remains **lightweight, maintainable, and future-proof**.

---

## Core Motivation

Healthcare AI systems increasingly require:

- Modular disease-level predictors
- Consistent input/output contracts
- Confidence-aware predictions
- Clear separation between *prediction* and *decision-making*

HealthML-Hub acts as the **model foundation layer** in such systems.

Instead of building monolithic applications, HealthML-Hub enables developers and researchers to **compose intelligent systems by combining specialized medical models**, each responsible for a clearly defined predictive task.

---

## What HealthML-Hub Is

- A **medical ML model library**
- A collection of **disease-specific predictive capabilities**
- A reusable foundation for downstream systems
- A research-friendly and system-design-friendly codebase

Each disease model:

- Has its own schema-defined inputs and outputs
- Exposes a single, stable `predict()` entrypoint
- Reports prediction confidence and known limitations
- Can be versioned, evaluated, and replaced independently

---

## What HealthML-Hub Is NOT

To avoid ambiguity, HealthML-Hub is intentionally **not**:

- An AI Doctor
- A clinical workflow engine
- A conversational or agent-based system
- A recommendation or prescription engine
- A UI-driven healthcare application

HealthML-Hub focuses purely on **predictive modeling**, leaving reasoning, interaction, and orchestration to external systems.

---

## Supported Disease Domains

HealthML-Hub currently includes predictive models across multiple medical domains:

| Disease | Domain | Model Paradigm | Status |
|---------|--------|----------------|--------|
| Brain Tumor | Neurology | Deep Learning | Development |
| Breast Cancer | Oncology | Machine Learning | Development |
| Diabetes | Metabolic | Machine Learning | Development |
| Heart Disease | Cardiovascular | Machine Learning | Development |
| Kidney Disease | Renal | Machine Learning | Development |
| Liver Disease | Hepatic | Machine Learning | Development |
| Lung Disease | Pulmonary | Machine Learning | Development |
| Malaria | Infectious Disease | Machine Learning | Development |
| Parkinson's Disease | Neurology | Machine Learning | Development |

A complete and evolving catalog is maintained in `docs/model_catalog.md`.

---

## Architectural Philosophy

HealthML-Hub follows a **strict separation of concerns**:

```
External Systems (AI Doctor, Dashboards, Research Tools)
↓
Stable Model Inference Contracts
↓
HealthML-Hub
(Model-only prediction layer)
```

**Key architectural principles:**

- Models are stateless and deterministic
- Inference is explicit and reproducible
- Inputs and outputs are schema-validated
- No hidden reasoning or decision logic exists inside models

This makes HealthML-Hub safe to integrate, test, and extend over time.

---

## Inference Contract (Core Design Principle)

Each disease module exposes exactly **one public inference function**:

```python
predict(input_schema) -> {
    prediction,
    confidence,
    model_version,
    known_limitations,
    warning_if_any
}
```

This contract ensures:

- Downstream systems can rely on consistent behavior
- Models can be swapped or upgraded independently
- External orchestration systems remain decoupled from ML internals

---

## Repository Structure (High-Level)

```
HealthML-Hub/
├── models/        # Disease-specific ML/DL models
├── shared/        # Cross-disease utilities
├── pipelines/     # Automation for training & evaluation
├── docs/          # System-level documentation
├── tests/         # Model correctness & reliability tests
└── README.md
```

Each disease follows a uniform internal structure to enforce consistency and maintainability.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rayari-1729/HealthML-Hub.git
cd HealthML-Hub

# Install dependencies
pip install -r requirements.txt

# Install package (optional)
pip install -e .
```

### Basic Usage

#### Example: Diabetes Prediction

```python
from models.diabetes.src.schema import DiabetesInput
from models.diabetes.src.predict import predict

# Prepare input data
input_data = DiabetesInput(
    pregnancies=6,
    glucose=148,
    blood_pressure=72,
    skin_thickness=35,
    insulin=0,
    bmi=33.6,
    diabetes_pedigree=0.627,
    age=50
)

# Make prediction
result = predict(input_data)

# Result contains:
# - prediction: 0 or 1
# - probability: 0.0 to 1.0
# - confidence: 0.0 to 1.0
# - model_name, model_version
# - known_limitations
# - warning (if confidence is low)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

#### Example: Brain Tumor Detection

```python
from models.brain_tumor.src.schema import BrainTumorInput
from models.brain_tumor.src.predict import predict

input_data = BrainTumorInput(
    mean_intensity=120.5,
    std_intensity=25.3,
    contrast=0.8,
    energy=0.5,
    homogeneity=0.7,
    correlation=0.6,
    entropy=4.2,
    area=2500,
    perimeter=180,
    compactness=0.8,
    eccentricity=0.3,
    solidity=0.9,
    extent=0.85,
    aspect_ratio=1.2,
    equivalent_diameter=56
)

result = predict(input_data)
```

### Training a Model

```bash
# Train diabetes model
python models/diabetes/src/train.py --data-path /path/to/data --output-dir models/diabetes/artifacts

# Train brain tumor model
python models/brain_tumor/src/train.py --data-path /path/to/data --output-dir models/brain_tumor/artifacts
```

### Evaluating a Model

```bash
# Evaluate diabetes model
python models/diabetes/src/evaluate.py --model-path models/diabetes/artifacts/model.joblib
```

---

## Extensions & Future Use

HealthML-Hub is intentionally designed to serve as a foundation, not a final product.

### Planned Extensions

- Improved model benchmarking and comparison
- Model versioning and lifecycle management
- Enhanced explainability support (e.g., SHAP, Grad-CAM)
- Dataset provenance tracking
- Integration hooks for external orchestration systems

### Integration with AI Doctor Systems

In a future AI Doctor architecture:

- **HealthML-Hub provides** raw predictive intelligence
- **The AI Doctor handles**:
  - Patient interaction
  - Evidence gathering
  - Multi-model reasoning
  - Decision synthesis

This layered approach ensures long-term scalability, regulatory clarity, and system robustness.

---

## Development Status

**Status: ACTIVE DEVELOPMENT**

Current focus areas:

- Baseline model correctness
- Input/output schema stability
- Reproducible evaluation
- Clear documentation and contribution standards

Future phases are documented in `docs/roadmap.md`.

---

## Documentation

- **[Vision](docs/vision.md)**: Project vision and philosophy
- **[Architecture](docs/architecture.md)**: Technical architecture and design decisions
- **[Model Catalog](docs/model_catalog.md)**: Complete catalog of all disease models
- **[Evaluation Standards](docs/evaluation_standards.md)**: Metrics and evaluation methodology
- **[Clinical Disclaimer](docs/clinical_disclaimer.md)**: Important medical and legal disclaimers
- **[Roadmap](docs/roadmap.md)**: Development roadmap and future plans

---

## Contributing

HealthML-Hub welcomes contributions from:

- Machine learning engineers
- Data scientists
- Healthcare AI researchers
- MLOps practitioners

Due to the medical domain, contributions are reviewed with emphasis on:

- Code clarity
- Reproducibility
- Documentation quality
- Explicit model limitations

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## ⚠️ IMPORTANT CLINICAL DISCLAIMER

**THIS REPOSITORY IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- ❌ **NOT** a diagnostic system
- ❌ **NOT** FDA-approved or validated for clinical use
- ❌ **NOT** a replacement for medical professionals
- ❌ **NOT** suitable for use in patient care

See [Clinical Disclaimer](docs/clinical_disclaimer.md) for complete information.

**By using this repository, you acknowledge that you have read, understood, and agree to the clinical disclaimer.**

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Final Note

HealthML-Hub is best understood as **infrastructure, not an application**.

It exists to answer one question reliably:

**"What does the data suggest for this specific medical condition?"**

Everything beyond that belongs to systems built on top of it.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/rayari-1729/HealthML-Hub/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rayari-1729/HealthML-Hub/discussions)
- **Security**: See [SECURITY.md](SECURITY.md)
