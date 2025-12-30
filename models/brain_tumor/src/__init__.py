"""Brain tumor detection model package."""

from .predict import predict
from .schema import BrainTumorInput, BrainTumorOutput

__all__ = ["predict", "BrainTumorInput", "BrainTumorOutput"]

