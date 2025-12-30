"""Input/output schema for malaria prediction model."""

from typing import Optional

from pydantic import BaseModel, Field


class MalariaInput(BaseModel):
    """Input schema for malaria prediction (image-based features)."""
    
    # Extracted features from cell images
    mean_radius: float = Field(..., ge=0, description="Mean radius of infected cells")
    mean_texture: float = Field(..., ge=0, description="Mean texture")
    mean_perimeter: float = Field(..., ge=0, description="Mean perimeter")
    mean_area: float = Field(..., ge=0, description="Mean area")
    mean_smoothness: float = Field(..., ge=0, description="Mean smoothness")
    mean_compactness: float = Field(..., ge=0, description="Mean compactness")
    mean_concavity: float = Field(..., ge=0, description="Mean concavity")
    mean_concave_points: float = Field(..., ge=0, description="Mean concave points")
    mean_symmetry: float = Field(..., ge=0, description="Mean symmetry")
    mean_fractal_dimension: float = Field(..., ge=0, description="Mean fractal dimension")


class MalariaOutput(BaseModel):
    """Output schema for malaria prediction."""
    
    prediction: int = Field(..., description="0=Uninfected, 1=Parasitized")
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_name: str = Field(default="malaria_classifier")
    model_version: str = Field(default="1.0.0")
    known_limitations: list = Field(default_factory=lambda: ["Trained on synthetic data"])
    warning: Optional[str] = None

