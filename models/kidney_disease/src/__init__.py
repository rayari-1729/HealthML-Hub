"""Kidney disease prediction model package."""

from .predict import predict
from .schema import KidneyDiseaseInput, KidneyDiseaseOutput

__all__ = ["predict", "KidneyDiseaseInput", "KidneyDiseaseOutput"]

