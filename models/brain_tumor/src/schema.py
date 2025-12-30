"""Input/output schema for brain tumor detection model."""

from typing import Optional

from pydantic import BaseModel, Field


class BrainTumorInput(BaseModel):
    """Input schema for brain tumor detection (image feature extraction)."""
    
    # Extracted features from brain MRI images
    mean_intensity: float = Field(..., ge=0, description="Mean pixel intensity")
    std_intensity: float = Field(..., ge=0, description="Standard deviation of intensity")
    contrast: float = Field(..., ge=0, description="Image contrast")
    energy: float = Field(..., ge=0, description="Image energy")
    homogeneity: float = Field(..., ge=0, le=1, description="Image homogeneity")
    correlation: float = Field(..., ge=-1, le=1, description="Image correlation")
    entropy: float = Field(..., ge=0, description="Image entropy")
    area: float = Field(..., ge=0, description="Tumor area (if detected)")
    perimeter: float = Field(..., ge=0, description="Tumor perimeter")
    compactness: float = Field(..., ge=0, description="Shape compactness")
    eccentricity: float = Field(..., ge=0, le=1, description="Shape eccentricity")
    solidity: float = Field(..., ge=0, le=1, description="Shape solidity")
    extent: float = Field(..., ge=0, le=1, description="Extent ratio")
    aspect_ratio: float = Field(..., ge=0, description="Aspect ratio")
    equivalent_diameter: float = Field(..., ge=0, description="Equivalent diameter")


class BrainTumorOutput(BaseModel):
    """Output schema for brain tumor detection."""
    
    prediction: int = Field(..., description="0=No tumor, 1=Tumor detected")
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_name: str = Field(default="brain_tumor_classifier")
    model_version: str = Field(default="1.0.0")
    known_limitations: list = Field(
        default_factory=lambda: [
            "Trained on synthetic data",
            "Requires pre-processed image features",
            "Not a full image classification model",
        ],
    )
    warning: Optional[str] = None

