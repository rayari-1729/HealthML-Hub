"""Input/output schema for heart disease prediction model."""

from typing import Optional

from pydantic import BaseModel, Field


class HeartDiseaseInput(BaseModel):
    """Input schema for heart disease prediction."""
    
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dL)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dL (0=no, 1=yes)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0=no, 1=yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by flourosopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }


class HeartDiseaseOutput(BaseModel):
    """Output schema for heart disease prediction."""
    
    prediction: int = Field(..., description="Predicted class (0=No disease, 1=Disease)")
    probability: float = Field(..., ge=0, le=1, description="Probability of heart disease")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence in prediction")
    model_name: str = Field(default="heart_disease_classifier", description="Model identifier")
    model_version: str = Field(default="1.0.0", description="Model version")
    known_limitations: list = Field(
        default_factory=lambda: [
            "Trained on Cleveland Heart Disease Dataset",
            "May not generalize to all populations",
            "Requires all 13 input features",
        ],
        description="Known model limitations",
    )
    warning: Optional[str] = Field(
        default=None,
        description="Warning message if confidence is low",
    )

