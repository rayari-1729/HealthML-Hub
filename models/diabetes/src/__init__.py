"""Diabetes prediction model package."""

from .predict import predict
from .schema import DiabetesInput, DiabetesOutput

__all__ = [
    "predict",
    "DiabetesInput",
    "DiabetesOutput",
]

