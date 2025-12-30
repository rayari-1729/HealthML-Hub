"""Parkinson's disease prediction model package."""

from .predict import predict
from .schema import ParkinsonsInput, ParkinsonsOutput

__all__ = ["predict", "ParkinsonsInput", "ParkinsonsOutput"]

