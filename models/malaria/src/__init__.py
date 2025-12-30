"""Malaria prediction model package."""

from .predict import predict
from .schema import MalariaInput, MalariaOutput

__all__ = ["predict", "MalariaInput", "MalariaOutput"]

