"""Tests for ml.evaluator."""
import pytest
import pandas as pd
from ml.evaluator import prune_features_aggressive, evaluate_dataset


def test_prune_features_drops_constant():
    """Constant columns are dropped."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5], "target": [0, 1, 0]})
    out, dropped = prune_features_aggressive(df, "target", "classification")
    assert "b" in dropped
    assert "b" not in out.columns


def test_prune_features_keeps_target():
    """Target is never dropped."""
    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    out, _ = prune_features_aggressive(df, "target", "classification")
    assert "target" in out.columns


def test_evaluate_dataset_regression():
    """evaluate_dataset runs for regression."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
        "b": [2.0, 4.0, 6.0, 8.0, 10.0] * 4,
        "target": [3.0, 6.0, 9.0, 12.0, 15.0] * 4,
    })
    result = evaluate_dataset(df, "target", "regression")
    assert "warnings" in result
    assert isinstance(result["warnings"], list)
