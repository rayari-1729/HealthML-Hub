# Heart Disease Prediction Model

## Overview

This model predicts heart disease status (binary classification) based on patient clinical features.

## Model Type

**Machine Learning (Gradient Boosting Classifier)**

## Input Schema

The model requires 13 input features from the Cleveland Heart Disease Dataset format.

## Usage

```python
from models.heart_disease.src.schema import HeartDiseaseInput
from models.heart_disease.src.predict import predict

input_data = HeartDiseaseInput(
    age=63, sex=1, cp=3, trestbps=145, chol=233, fbs=1,
    restecg=0, thalach=150, exang=0, oldpeak=2.3, slope=0, ca=0, thal=1
)

result = predict(input_data)
```

## Development Status

**UNDER ACTIVE DEVELOPMENT**

## Clinical Disclaimer

This model is for educational and research purposes only. NOT for clinical decision-making.

