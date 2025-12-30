# Diabetes Prediction Model

## Overview

This model predicts diabetes status (binary classification) based on patient clinical features.

## Model Type

**Machine Learning (Random Forest Classifier)**

## Input Schema

The model requires the following inputs:

- `pregnancies` (int): Number of pregnancies (0-20)
- `glucose` (float): Plasma glucose concentration in mg/dL (0-200)
- `blood_pressure` (float): Diastolic blood pressure in mm Hg (0-150)
- `skin_thickness` (float): Triceps skin fold thickness in mm (0-100)
- `insulin` (float): 2-Hour serum insulin in mu U/ml (0-900)
- `bmi` (float): Body mass index in kg/m² (0-70)
- `diabetes_pedigree` (float): Diabetes pedigree function (0-3)
- `age` (int): Age in years (0-120)

## Output

The model returns:

- `prediction` (int): 0 = No diabetes, 1 = Diabetes
- `probability` (float): Probability of diabetes (0-1)
- `confidence` (float): Model confidence in prediction (0-1)
- `model_name` (str): Model identifier
- `model_version` (str): Model version
- `known_limitations` (list): List of known limitations
- `warning` (str, optional): Warning message if confidence is low

## Usage

```python
from models.diabetes.src.schema import DiabetesInput
from models.diabetes.src.predict import predict

# Prepare input
input_data = DiabetesInput(
    pregnancies=6,
    glucose=148,
    blood_pressure=72,
    skin_thickness=35,
    insulin=0,
    bmi=33.6,
    diabetes_pedigree=0.627,
    age=50,
)

# Predict
result = predict(input_data)
print(result)
```

## Training

To train the model:

```bash
python models/diabetes/src/train.py --data-path /path/to/data --output-dir models/diabetes/artifacts
```

## Evaluation

To evaluate the model:

```bash
python models/diabetes/src/evaluate.py --model-path models/diabetes/artifacts/model.joblib
```

## Known Limitations

1. Trained on synthetic data for demonstration purposes
2. In production, should be trained on real Pima Indians Diabetes Dataset
3. May not generalize to all populations
4. Requires all input features to be non-zero (missing data handling is strict)

## Development Status

**UNDER ACTIVE DEVELOPMENT**

This model is currently under active development and should not be used for clinical decision-making.

## Clinical Disclaimer

This model is for educational and research purposes only. It is NOT a diagnostic system and should NOT be used for clinical decision-making without proper validation and regulatory approval.

