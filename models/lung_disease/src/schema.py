"""Input/output schema for lung disease prediction model."""

from typing import Optional

from pydantic import BaseModel, Field


class LungDiseaseInput(BaseModel):
    """Input schema for lung disease prediction."""
    
    age: int = Field(..., ge=0, le=120)
    gender: int = Field(..., ge=0, le=1)
    air_pollution: int = Field(..., ge=0, le=10)
    alcohol_use: int = Field(..., ge=0, le=1)
    dust_allergy: int = Field(..., ge=0, le=1)
    occupational_hazards: int = Field(..., ge=0, le=1)
    genetic_risk: int = Field(..., ge=0, le=1)
    chronic_lung_disease: int = Field(..., ge=0, le=1)
    balanced_diet: int = Field(..., ge=0, le=1)
    obesity: int = Field(..., ge=0, le=1)
    smoking: int = Field(..., ge=0, le=1)
    passive_smoking: int = Field(..., ge=0, le=1)
    chest_pain: int = Field(..., ge=0, le=1)
    coughing_blood: int = Field(..., ge=0, le=1)
    fatigue: int = Field(..., ge=0, le=1)
    weight_loss: int = Field(..., ge=0, le=1)
    shortness_of_breath: int = Field(..., ge=0, le=1)
    wheezing: int = Field(..., ge=0, le=1)
    swallowing_difficulty: int = Field(..., ge=0, le=1)
    clubbing_of_finger_nails: int = Field(..., ge=0, le=1)
    frequent_cold: int = Field(..., ge=0, le=1)
    dry_cough: int = Field(..., ge=0, le=1)
    snoring: int = Field(..., ge=0, le=1)


class LungDiseaseOutput(BaseModel):
    """Output schema for lung disease prediction."""
    
    prediction: int = Field(..., description="0=No disease, 1=Disease")
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_name: str = Field(default="lung_disease_classifier")
    model_version: str = Field(default="1.0.0")
    known_limitations: list = Field(default_factory=lambda: ["Trained on synthetic data"])
    warning: Optional[str] = None

