"""Input/output schema for kidney disease prediction model."""

from typing import Optional

from pydantic import BaseModel, Field


class KidneyDiseaseInput(BaseModel):
    """Input schema for kidney disease prediction."""
    
    age: float = Field(..., ge=0, le=120)
    blood_pressure: float = Field(..., ge=0, le=200)
    specific_gravity: float = Field(..., ge=1.0, le=1.1)
    albumin: float = Field(..., ge=0, le=5)
    sugar: float = Field(..., ge=0, le=5)
    red_blood_cells: int = Field(..., ge=0, le=1)
    pus_cell: int = Field(..., ge=0, le=1)
    pus_cell_clumps: int = Field(..., ge=0, le=1)
    bacteria: int = Field(..., ge=0, le=1)
    blood_glucose_random: float = Field(..., ge=0, le=500)
    blood_urea: float = Field(..., ge=0, le=200)
    serum_creatinine: float = Field(..., ge=0, le=20)
    sodium: float = Field(..., ge=0, le=200)
    potassium: float = Field(..., ge=0, le=10)
    haemoglobin: float = Field(..., ge=0, le=20)
    packed_cell_volume: float = Field(..., ge=0, le=100)
    white_blood_cell_count: float = Field(..., ge=0, le=50000)
    red_blood_cell_count: float = Field(..., ge=0, le=10)
    hypertension: int = Field(..., ge=0, le=1)
    diabetes_mellitus: int = Field(..., ge=0, le=1)
    coronary_artery_disease: int = Field(..., ge=0, le=1)
    appetite: int = Field(..., ge=0, le=1)
    peda_edema: int = Field(..., ge=0, le=1)
    aanemia: int = Field(..., ge=0, le=1)


class KidneyDiseaseOutput(BaseModel):
    """Output schema for kidney disease prediction."""
    
    prediction: int = Field(..., description="0=No disease, 1=Disease")
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_name: str = Field(default="kidney_disease_classifier")
    model_version: str = Field(default="1.0.0")
    known_limitations: list = Field(default_factory=lambda: ["Trained on synthetic data"])
    warning: Optional[str] = None

