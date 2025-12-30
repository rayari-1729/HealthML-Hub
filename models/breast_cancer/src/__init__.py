"""Breast cancer prediction model package."""

from .predict import predict
from .schema import BreastCancerInput, BreastCancerOutput

__all__ = ["predict", "BreastCancerInput", "BreastCancerOutput"]

