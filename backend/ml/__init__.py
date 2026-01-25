"""ML module for Intent2Model."""

from .profiler import profile_dataset
from .pipeline_builder import build_pipeline
from .trainer import train_classification, train_regression, compare_models
from .evaluator import evaluate_dataset

__all__ = [
    "profile_dataset",
    "build_pipeline",
    "train_classification",
    "train_regression",
    "compare_models",
    "evaluate_dataset"
]
