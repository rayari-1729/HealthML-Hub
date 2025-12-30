# Contributing to HealthML-Hub

Thank you for your interest in contributing to HealthML-Hub! This document provides guidelines for contributing new disease models, improvements, and fixes.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Adding a New Disease Model](#adding-a-new-disease-model)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Requirements](#documentation-requirements)
- [Submission Process](#submission-process)
- [Review Process](#review-process)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Follow the project's coding standards
- Document your code clearly
- Test your contributions

## Getting Started

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/HealthML-Hub.git
   cd HealthML-Hub
   ```

3. **Set up upstream remote**:
   ```bash
   git remote add upstream https://github.com/originalowner/HealthML-Hub.git
   ```

### Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

We welcome:
- ✅ New disease models (following standard structure)
- ✅ Improvements to existing models
- ✅ Bug fixes
- ✅ Documentation improvements
- ✅ Test coverage improvements
- ✅ Performance optimizations
- ✅ Feature requests (open an issue first)

### What NOT to Contribute

- ❌ Agent logic or orchestration code
- ❌ UI components
- ❌ Clinical decision-making logic
- ❌ Placeholder or empty files
- ❌ Code that doesn't follow the standard structure

## Adding a New Disease Model

### Step 1: Create Directory Structure

```bash
mkdir -p models/<disease_name>/src
mkdir -p models/<disease_name>/configs
mkdir -p models/<disease_name>/artifacts
```

### Step 2: Implement Required Files

#### 1. Schema (`src/schema.py`)

```python
"""Input/output schema for <disease> prediction model."""

from typing import Optional
from pydantic import BaseModel, Field

class DiseaseInput(BaseModel):
    """Input schema for <disease> prediction."""
    feature1: float = Field(..., ge=0, le=100, description="...")
    feature2: int = Field(..., ge=0, le=10, description="...")
    # ... more features

class DiseaseOutput(BaseModel):
    """Output schema for <disease> prediction."""
    prediction: int = Field(..., description="0=No disease, 1=Disease")
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_name: str = Field(default="<disease>_classifier")
    model_version: str = Field(default="1.0.0")
    known_limitations: list = Field(default_factory=lambda: [...])
    warning: Optional[str] = None
```

#### 2. Model (`src/model.py`)

```python
"""<Disease> prediction model definition."""

from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # or appropriate model
from shared.utils.model_loader import load_model

class DiseaseModel:
    """<Disease> prediction model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = RandomForestClassifier(...)
        self.model_path = model_path
        self.feature_names = [...]
        if model_path and model_path.exists():
            self.load(model_path)
    
    # Implement: load, fit, predict, predict_proba, get_feature_names
```

#### 3. Training (`src/train.py`)

```python
"""Training script for <disease> model."""

# Implement:
# - load_training_data()
# - train_disease_model()
# - Save model and scaler
# - Save registry.json with metrics
```

#### 4. Evaluation (`src/evaluate.py`)

```python
"""Evaluation script for <disease> model."""

# Implement:
# - evaluate_disease_model()
# - Calculate metrics using shared.evaluation.metrics
# - Save evaluation results
```

#### 5. Inference (`src/predict.py`) - **MOST IMPORTANT**

```python
"""Inference entrypoint for <disease> prediction model."""

def predict(input_data: DiseaseInput, model_dir: Optional[Path] = None) -> dict:
    """
    Predict <disease> status from input data.
    
    This is the SINGLE public inference entrypoint.
    """
    # Load model
    # Preprocess input
    # Make prediction
    # Calculate confidence
    # Return standardized output
    # Handle errors gracefully
```

**Critical Requirements for `predict()`**:
- ✅ Must be stateless
- ✅ Must NOT perform training
- ✅ Must NOT write files
- ✅ Must return standardized output format
- ✅ Must handle errors gracefully (return abstention response)

#### 6. Configuration (`configs/default.yaml`)

```yaml
# Default configuration for <disease> model

model:
  name: <disease>_classifier
  type: <model_type>
  version: "1.0.0"

training:
  # Hyperparameters

preprocessing:
  scaler_type: "standard"

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc
```

#### 7. Registry (`artifacts/registry.json`)

```json
{
  "model_name": "<disease>_classifier",
  "model_version": "1.0.0",
  "status": "under_development",
  "metrics": {},
  "feature_names": [...],
  "known_limitations": [...]
}
```

#### 8. Documentation (`README.md`)

Include:
- Model overview
- Model type (ML/DL)
- Input schema description
- Usage examples
- Development status
- Clinical disclaimer

#### 9. Package Init (`src/__init__.py`)

```python
"""<Disease> prediction model package."""

from .predict import predict
from .schema import DiseaseInput, DiseaseOutput

__all__ = ["predict", "DiseaseInput", "DiseaseOutput"]
```

### Step 3: Reference Implementation

Study existing models for reference:
- `models/diabetes/` - Complete ML model example
- `models/brain_tumor/` - Deep learning example
- `models/heart_disease/` - Another ML example

### Step 4: Update Documentation

1. Add entry to `docs/model_catalog.md`
2. Update main `README.md` if needed
3. Ensure all documentation is accurate

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** for all function signatures
- Use **docstrings** for all functions and classes
- Maximum line length: **100 characters** (preferred) or 120

### Code Quality

- **No empty files**: Every file must contain real, working code
- **No placeholders**: No `TODO` comments without implementation
- **No hard-coded paths**: Use `Path` objects and relative paths
- **Error handling**: Always handle errors gracefully
- **Logging**: Use `shared.utils.logging_utils` for logging

### Imports

```python
# Standard library imports
from pathlib import Path
from typing import Optional

# Third-party imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Local imports
from shared.evaluation.metrics import calculate_binary_metrics
from shared.preprocessing.scalers import get_scaler

# Relative imports
from .model import DiseaseModel
from .schema import DiseaseInput, DiseaseOutput
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `DiabetesModel`)
- **Functions**: `snake_case` (e.g., `predict`, `train_diabetes_model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_MODEL_VERSION`)
- **Files**: `snake_case.py` (e.g., `schema.py`, `train.py`)

### Documentation

- **Module docstrings**: Describe the module's purpose
- **Class docstrings**: Describe the class and its methods
- **Function docstrings**: Use Google-style or NumPy-style
  ```python
  def predict(input_data: DiseaseInput) -> dict:
      """
      Predict disease status from input data.
      
      Args:
          input_data: DiseaseInput instance with patient data
          
      Returns:
          Dictionary with prediction, confidence, and metadata
      """
  ```

## Testing Requirements

### Test Coverage

- Write tests for all new code
- Aim for >80% code coverage
- Test both success and error cases

### Test Structure

```python
# tests/model_tests/test_diabetes.py

def test_diabetes_predict():
    """Test diabetes prediction."""
    from models.diabetes.src.schema import DiabetesInput
    from models.diabetes.src.predict import predict
    
    input_data = DiabetesInput(...)
    result = predict(input_data)
    
    assert "prediction" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/model_tests/test_diabetes.py

# Run with coverage
pytest --cov=models tests/
```

## Documentation Requirements

### Required Documentation

1. **Model README**: `models/<disease>/README.md`
2. **Code Comments**: Clear, concise comments
3. **Docstrings**: Complete docstrings for all functions/classes
4. **Type Hints**: Type hints for all function signatures

### Documentation Standards

- Use clear, concise language
- Include usage examples
- Document all parameters and return values
- Explain any non-obvious logic
- Include clinical disclaimers where appropriate

## Submission Process

### Before Submitting

1. **Ensure code works**:
   ```bash
   python models/<disease>/src/train.py
   python models/<disease>/src/evaluate.py --model-path models/<disease>/artifacts/model.joblib
   ```

2. **Run tests**:
   ```bash
   pytest tests/
   ```

3. **Check code style**:
   ```bash
   # Use black, flake8, or your preferred formatter
   black models/<disease>/
   flake8 models/<disease>/
   ```

4. **Update documentation**:
   - Update `docs/model_catalog.md`
   - Update main `README.md` if needed
   - Ensure all docstrings are complete

### Creating a Pull Request

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add <disease> prediction model"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**:
   - Go to GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out PR template
   - Submit

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New disease model
- [ ] Bug fix
- [ ] Documentation improvement
- [ ] Performance optimization
- [ ] Other (please describe)

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Documentation
- [ ] README updated
- [ ] Model catalog updated
- [ ] Code comments added

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] No hard-coded paths
- [ ] Error handling implemented
- [ ] Logging added where appropriate
```

## Review Process

### What Reviewers Look For

1. **Code Quality**:
   - Follows coding standards
   - No empty files or placeholders
   - Proper error handling
   - Good test coverage

2. **Structure**:
   - Follows standard disease model structure
   - All required files present
   - Consistent with existing models

3. **Functionality**:
   - Code works as expected
   - Tests pass
   - Documentation is accurate

4. **Integration**:
   - Uses shared utilities correctly
   - Follows inference contract
   - No breaking changes

### Review Timeline

- Initial review: Within 1 week
- Follow-up reviews: As needed
- Final approval: When all concerns addressed

### Addressing Review Comments

1. Read comments carefully
2. Ask questions if unclear
3. Make requested changes
4. Respond to comments
5. Request re-review when ready

## Getting Help

### Questions?

- Open an issue for questions
- Check existing documentation
- Review existing models for examples
- Ask in GitHub Discussions

### Stuck?

- Check [Architecture Documentation](docs/architecture.md)
- Review [Model Catalog](docs/model_catalog.md)
- Look at existing model implementations
- Open an issue with your question

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md (if created)
- Credited in release notes
- Acknowledged in documentation

## Thank You!

Thank you for contributing to HealthML-Hub! Your contributions help make medical ML more accessible and useful for research and education.

---

**Remember**: All contributions must follow the clinical disclaimer. Models are for educational and research purposes only.

