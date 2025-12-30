"""Input/output schema for breast cancer prediction model."""

from typing import Optional

from pydantic import BaseModel, Field


class BreastCancerInput(BaseModel):
    """Input schema for breast cancer prediction (Wisconsin dataset features)."""
    
    radius_mean: float = Field(..., ge=0, description="Mean radius of cell nuclei")
    texture_mean: float = Field(..., ge=0, description="Mean texture of cell nuclei")
    perimeter_mean: float = Field(..., ge=0, description="Mean perimeter of cell nuclei")
    area_mean: float = Field(..., ge=0, description="Mean area of cell nuclei")
    smoothness_mean: float = Field(..., ge=0, description="Mean smoothness")
    compactness_mean: float = Field(..., ge=0, description="Mean compactness")
    concavity_mean: float = Field(..., ge=0, description="Mean concavity")
    concave_points_mean: float = Field(..., ge=0, description="Mean concave points")
    symmetry_mean: float = Field(..., ge=0, description="Mean symmetry")
    fractal_dimension_mean: float = Field(..., ge=0, description="Mean fractal dimension")
    
    class Config:
        json_schema_extra = {
            "example": {
                "radius_mean": 17.99,
                "texture_mean": 10.38,
                "perimeter_mean": 122.8,
                "area_mean": 1001.0,
                "smoothness_mean": 0.1184,
                "compactness_mean": 0.2776,
                "concavity_mean": 0.3001,
                "concave_points_mean": 0.1471,
                "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.07871,
            }
        }


class BreastCancerOutput(BaseModel):
    """Output schema for breast cancer prediction."""
    
    prediction: int = Field(..., description="Predicted class (0=Benign, 1=Malignant)")
    probability: float = Field(..., ge=0, le=1, description="Probability of malignancy")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    model_name: str = Field(default="breast_cancer_classifier")
    model_version: str = Field(default="1.0.0")
    known_limitations: list = Field(
        default_factory=lambda: [
            "Trained on Wisconsin Breast Cancer Dataset",
            "Based on cell nuclei features from fine needle aspirate",
        ],
    )
    warning: Optional[str] = None

