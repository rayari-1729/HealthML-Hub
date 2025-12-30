# Breast Cancer Prediction Model

## Overview

Binary classification model for breast cancer prediction based on cell nuclei features.

## Model Type

**Machine Learning (SVM Classifier)**

## Input Schema

10 features from Wisconsin Breast Cancer Dataset (mean values of cell nuclei characteristics).

## Usage

```python
from models.breast_cancer.src.schema import BreastCancerInput
from models.breast_cancer.src.predict import predict

input_data = BreastCancerInput(
    radius_mean=17.99, texture_mean=10.38, perimeter_mean=122.8,
    area_mean=1001.0, smoothness_mean=0.1184, compactness_mean=0.2776,
    concavity_mean=0.3001, concave_points_mean=0.1471,
    symmetry_mean=0.2419, fractal_dimension_mean=0.07871
)

result = predict(input_data)
```

## Development Status

**UNDER ACTIVE DEVELOPMENT**

## Clinical Disclaimer

For educational and research purposes only. NOT for clinical decision-making.

