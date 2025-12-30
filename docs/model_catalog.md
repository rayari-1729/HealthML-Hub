# HealthML-Hub Model Catalog

## Overview

This document provides a comprehensive catalog of all disease models available in HealthML-Hub.

## Model Status Legend

- **✅ Implemented**: Model code is complete and functional
- **🔄 Under Development**: Model is being actively developed
- **📊 Trained**: Model has been trained on real data
- **🧪 Synthetic Data**: Currently using synthetic data for demonstration

## Disease Models

### 1. Diabetes Prediction

- **Model Type**: Machine Learning (Random Forest)
- **Input Type**: Clinical features (8 features)
- **Output**: Binary classification (0=No diabetes, 1=Diabetes)
- **Status**: ✅ Implemented | 🔄 Under Development | 🧪 Synthetic Data
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree, Age
- **Dataset**: Pima Indians Diabetes Dataset (in production)
- **Location**: `models/diabetes/`

### 2. Heart Disease Prediction

- **Model Type**: Machine Learning (Gradient Boosting)
- **Input Type**: Clinical features (13 features)
- **Output**: Binary classification (0=No disease, 1=Disease)
- **Status**: ✅ Implemented | 🔄 Under Development | 🧪 Synthetic Data
- **Features**: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, FBS, Resting ECG, Max Heart Rate, Exercise Angina, ST Depression, Slope, Major Vessels, Thalassemia
- **Dataset**: Cleveland Heart Disease Dataset (in production)
- **Location**: `models/heart_disease/`

### 3. Breast Cancer Prediction

- **Model Type**: Machine Learning (SVM)
- **Input Type**: Cell nuclei features (10 features)
- **Output**: Binary classification (0=Benign, 1=Malignant)
- **Status**: ✅ Implemented | 🔄 Under Development | 🧪 Synthetic Data
- **Features**: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension (mean values)
- **Dataset**: Wisconsin Breast Cancer Dataset (in production)
- **Location**: `models/breast_cancer/`

### 4. Brain Tumor Detection

- **Model Type**: Deep Learning (Multi-Layer Perceptron)
- **Input Type**: Image features (15 features)
- **Output**: Binary classification (0=No tumor, 1=Tumor)
- **Status**: ✅ Implemented | 🔄 Under Development | 🧪 Synthetic Data
- **Features**: Intensity statistics, texture features (contrast, energy, homogeneity, correlation, entropy), shape features (area, perimeter, compactness, eccentricity, solidity, extent, aspect ratio, equivalent diameter)
- **Dataset**: Brain MRI Dataset (in production)
- **Architecture**: MLP (128, 64, 32 hidden layers)
- **Location**: `models/brain_tumor/`

### 5. Kidney Disease Prediction

- **Model Type**: Machine Learning (Random Forest)
- **Input Type**: Clinical features (24 features)
- **Output**: Binary classification (0=No disease, 1=Disease)
- **Status**: ✅ Implemented | 🔄 Under Development | 🧪 Synthetic Data
- **Features**: Age, Blood Pressure, Specific Gravity, Albumin, Sugar, Red Blood Cells, Pus Cell, Pus Cell Clumps, Bacteria, Blood Glucose, Blood Urea, Serum Creatinine, Sodium, Potassium, Haemoglobin, Packed Cell Volume, White Blood Cell Count, Red Blood Cell Count, Hypertension, Diabetes Mellitus, Coronary Artery Disease, Appetite, Peda Edema, Anemia
- **Location**: `models/kidney_disease/`

### 6. Liver Disease Prediction

- **Model Type**: Machine Learning (Gradient Boosting)
- **Input Type**: Clinical features (10 features)
- **Output**: Binary classification (0=No disease, 1=Disease)
- **Status**: ✅ Implemented | 🔄 Under Development | 🧪 Synthetic Data
- **Features**: Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphotase, Alanine Aminotransferase, Aspartate Aminotransferase, Total Proteins, Albumin, Albumin/Globulin Ratio
- **Location**: `models/liver_disease/`

### 7. Lung Disease Prediction

- **Model Type**: Machine Learning (Random Forest)
- **Input Type**: Clinical features and symptoms (23 features)
- **Output**: Binary classification (0=No disease, 1=Disease)
- **Status**: ✅ Implemented | 🔄 Under Development | 🧪 Synthetic Data
- **Features**: Age, Gender, Air Pollution, Alcohol Use, Dust Allergy, Occupational Hazards, Genetic Risk, Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoking, Chest Pain, Coughing Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring
- **Location**: `models/lung_disease/`

### 8. Malaria Prediction

- **Model Type**: Machine Learning (Random Forest)
- **Input Type**: Cell image features (10 features)
- **Output**: Binary classification (0=Uninfected, 1=Parasitized)
- **Status**: ✅ Implemented | 🔄 Under Development | 🧪 Synthetic Data
- **Features**: Mean Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
- **Location**: `models/malaria/`

### 9. Parkinson's Disease Prediction

- **Model Type**: Machine Learning (SVM)
- **Input Type**: Voice features (22 features)
- **Output**: Binary classification (0=Healthy, 1=Parkinson's)
- **Status**: ✅ Implemented | 🔄 Under Development | 🧪 Synthetic Data
- **Features**: MDVP Fo (Hz), MDVP Fhi (Hz), MDVP Flo (Hz), MDVP Jitter (%), MDVP Jitter (Abs), MDVP RAP, MDVP PPQ, Jitter DDP, MDVP Shimmer, MDVP Shimmer (dB), Shimmer APQ3, Shimmer APQ5, MDVP APQ, Shimmer DDA, NHR, HNR, RPDE, DFA, Spread1, Spread2, D2, PPE
- **Location**: `models/parkinsons/`

## Model Comparison Table

| Disease | Model Type | Input Features | Output Type | Status |
|---------|-----------|----------------|-------------|--------|
| Diabetes | ML (RF) | 8 | Binary | ✅ 🔄 🧪 |
| Heart Disease | ML (GB) | 13 | Binary | ✅ 🔄 🧪 |
| Breast Cancer | ML (SVM) | 10 | Binary | ✅ 🔄 🧪 |
| Brain Tumor | DL (MLP) | 15 | Binary | ✅ 🔄 🧪 |
| Kidney Disease | ML (RF) | 24 | Binary | ✅ 🔄 🧪 |
| Liver Disease | ML (GB) | 10 | Binary | ✅ 🔄 🧪 |
| Lung Disease | ML (RF) | 23 | Binary | ✅ 🔄 🧪 |
| Malaria | ML (RF) | 10 | Binary | ✅ 🔄 🧪 |
| Parkinson's | ML (SVM) | 22 | Binary | ✅ 🔄 🧪 |

**Legend:**
- ML = Machine Learning
- DL = Deep Learning
- RF = Random Forest
- GB = Gradient Boosting
- SVM = Support Vector Machine
- MLP = Multi-Layer Perceptron

## Model Output Format

All models return a standardized output dictionary:

```python
{
    "prediction": int,           # 0 or 1
    "probability": float,        # 0.0 to 1.0
    "confidence": float,         # 0.0 to 1.0
    "model_name": str,           # Model identifier
    "model_version": str,        # Version (e.g., "1.0.0")
    "known_limitations": list,    # List of limitations
    "warning": Optional[str]     # Warning if confidence is low
}
```

## Usage Examples

### Diabetes Model

```python
from models.diabetes.src.schema import DiabetesInput
from models.diabetes.src.predict import predict

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

result = predict(input_data)
```

### Brain Tumor Model

```python
from models.brain_tumor.src.schema import BrainTumorInput
from models.brain_tumor.src.predict import predict

input_data = BrainTumorInput(
    mean_intensity=120.5,
    std_intensity=25.3,
    contrast=0.8,
    # ... other features
)

result = predict(input_data)
```

## Adding New Models

To add a new disease model:

1. Create directory: `models/<disease_name>/`
2. Follow standard structure (see `docs/architecture.md`)
3. Implement all required files
4. Add entry to this catalog
5. Update `docs/roadmap.md` if needed

See `CONTRIBUTING.md` for detailed guidelines.

## Model Development Roadmap

See `docs/roadmap.md` for planned improvements and new models.

## Clinical Disclaimer

**All models in this catalog are for educational and research purposes only.**

- NOT validated for clinical use
- NOT FDA-approved
- NOT a diagnostic system
- NOT a replacement for medical professionals

See `docs/clinical_disclaimer.md` for complete disclaimer.

