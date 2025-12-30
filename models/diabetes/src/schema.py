"""Input/output schema for diabetes prediction model."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DiabetesInput(BaseModel):
    """Input schema for diabetes prediction."""
    
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=200, description="Plasma glucose concentration (mg/dL)")
    blood_pressure: float = Field(..., ge=0, le=150, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, le=70, description="Body mass index (kg/m²)")
    diabetes_pedigree: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    age: int = Field(..., ge=0, le=120, description="Age (years)")
    
    @field_validator("glucose", "blood_pressure", "skin_thickness", "insulin", "bmi")
    @classmethod
    def validate_not_zero(cls, v: float) -> float:
        """Ensure critical values are not zero (likely missing data)."""
        if v == 0.0:
            raise ValueError("Value cannot be zero (likely missing data)")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 6,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 0,
                "bmi": 33.6,
                "diabetes_pedigree": 0.627,
                "age": 50,
            }
        }


class DiabetesOutput(BaseModel):
    """Output schema for diabetes prediction."""
    
    prediction: int = Field(..., description="Predicted class (0=No diabetes, 1=Diabetes)")
    probability: float = Field(..., ge=0, le=1, description="Probability of diabetes")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence in prediction")
    model_name: str = Field(default="diabetes_classifier", description="Model identifier")
    model_version: str = Field(default="1.0.0", description="Model version")
    known_limitations: list = Field(
        default_factory=lambda: [
            "Trained on Pima Indians Diabetes Dataset",
            "May not generalize to all populations",
            "Requires all input features to be non-zero",
        ],
        description="Known model limitations",
    )
    warning: Optional[str] = Field(
        default=None,
        description="Warning message if confidence is low",
    )

