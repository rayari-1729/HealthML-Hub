"""Heart disease prediction model package."""

from .predict import predict
from .schema import HeartDiseaseInput, HeartDiseaseOutput

__all__ = ["predict", "HeartDiseaseInput", "HeartDiseaseOutput"]

