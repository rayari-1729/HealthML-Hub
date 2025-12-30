# Brain Tumor Detection Model

## Overview

Deep learning model for brain tumor detection based on MRI image features.

## Model Type

**Deep Learning (Multi-Layer Perceptron)**

## Architecture

MLP with hidden layers: (128, 64, 32)

## Input Schema

15 features extracted from brain MRI images including intensity statistics, texture features, and shape characteristics.

## Usage

```python
from models.brain_tumor.src.schema import BrainTumorInput
from models.brain_tumor.src.predict import predict

input_data = BrainTumorInput(
    mean_intensity=120.5, std_intensity=25.3, contrast=0.8,
    energy=0.5, homogeneity=0.7, correlation=0.6,
    entropy=4.2, area=2500, perimeter=180,
    compactness=0.8, eccentricity=0.3, solidity=0.9,
    extent=0.85, aspect_ratio=1.2, equivalent_diameter=56
)

result = predict(input_data)
```

## Development Status

**UNDER ACTIVE DEVELOPMENT**

## Known Limitations

1. Trained on synthetic data for demonstration
2. Uses extracted features, not raw images
3. In production, should use CNN for direct image classification

## Clinical Disclaimer

For educational and research purposes only. NOT for clinical decision-making.

