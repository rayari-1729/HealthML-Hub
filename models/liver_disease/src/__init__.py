"""Liver disease prediction model package."""

from .predict import predict
from .schema import LiverDiseaseInput, LiverDiseaseOutput

__all__ = ["predict", "LiverDiseaseInput", "LiverDiseaseOutput"]

