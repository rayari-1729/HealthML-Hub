"""Input/output schema for Parkinson's disease prediction model."""

from typing import Optional

from pydantic import BaseModel, Field


class ParkinsonsInput(BaseModel):
    """Input schema for Parkinson's disease prediction."""
    
    mdvp_fo_hz: float = Field(..., ge=0, description="Average vocal fundamental frequency")
    mdvp_fhi_hz: float = Field(..., ge=0, description="Maximum vocal fundamental frequency")
    mdvp_flo_hz: float = Field(..., ge=0, description="Minimum vocal fundamental frequency")
    mdvp_jitter_percent: float = Field(..., ge=0, description="Jitter percentage")
    mdvp_jitter_abs: float = Field(..., ge=0, description="Absolute jitter")
    mdvp_rap: float = Field(..., ge=0, description="Relative average perturbation")
    mdvp_ppq: float = Field(..., ge=0, description="Five-point period perturbation quotient")
    jitter_ddp: float = Field(..., ge=0, description="Difference of differences of periods")
    mdvp_shimmer: float = Field(..., ge=0, description="Shimmer")
    mdvp_shimmer_db: float = Field(..., ge=0, description="Shimmer in decibels")
    shimmer_apq3: float = Field(..., ge=0, description="Three-point amplitude perturbation quotient")
    shimmer_apq5: float = Field(..., ge=0, description="Five-point amplitude perturbation quotient")
    mdvp_apq: float = Field(..., ge=0, description="Amplitude perturbation quotient")
    shimmer_dda: float = Field(..., ge=0, description="Difference of differences of amplitudes")
    nhr: float = Field(..., ge=0, description="Noise-to-harmonics ratio")
    hnr: float = Field(..., ge=0, description="Harmonics-to-noise ratio")
    rpde: float = Field(..., ge=0, description="Recurrence period density entropy")
    dfa: float = Field(..., ge=0, description="Detrended fluctuation analysis")
    spread1: float = Field(..., description="Spread1")
    spread2: float = Field(..., description="Spread2")
    d2: float = Field(..., ge=0, description="D2")
    ppe: float = Field(..., ge=0, description="Pitch period entropy")


class ParkinsonsOutput(BaseModel):
    """Output schema for Parkinson's disease prediction."""
    
    prediction: int = Field(..., description="0=Healthy, 1=Parkinson's")
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_name: str = Field(default="parkinsons_classifier")
    model_version: str = Field(default="1.0.0")
    known_limitations: list = Field(default_factory=lambda: ["Trained on synthetic data"])
    warning: Optional[str] = None

