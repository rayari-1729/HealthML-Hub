"""Input/output schema for liver disease prediction model."""

from typing import Optional

from pydantic import BaseModel, Field


class LiverDiseaseInput(BaseModel):
    """Input schema for liver disease prediction."""
    
    age: int = Field(..., ge=0, le=120)
    gender: int = Field(..., ge=0, le=1)
    total_bilirubin: float = Field(..., ge=0, le=50)
    direct_bilirubin: float = Field(..., ge=0, le=20)
    alkaline_phosphotase: float = Field(..., ge=0, le=1000)
    alamine_aminotransferase: float = Field(..., ge=0, le=500)
    aspartate_aminotransferase: float = Field(..., ge=0, le=500)
    total_proteins: float = Field(..., ge=0, le=10)
    albumin: float = Field(..., ge=0, le=6)
    albumin_globulin_ratio: float = Field(..., ge=0, le=5)


class LiverDiseaseOutput(BaseModel):
    """Output schema for liver disease prediction."""
    
    prediction: int = Field(..., description="0=No disease, 1=Disease")
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_name: str = Field(default="liver_disease_classifier")
    model_version: str = Field(default="1.0.0")
    known_limitations: list = Field(default_factory=lambda: ["Trained on synthetic data"])
    warning: Optional[str] = None

