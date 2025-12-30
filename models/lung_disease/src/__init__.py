"""Lung disease prediction model package."""

from .predict import predict
from .schema import LungDiseaseInput, LungDiseaseOutput

__all__ = ["predict", "LungDiseaseInput", "LungDiseaseOutput"]

