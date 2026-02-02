"""Tests for plan_normalizer."""
import pytest
from agents.plan_normalizer import normalize_plan_dict


def test_normalize_plan_dict_minimal():
    """Minimal plan with profile gets filled correctly."""
    profile = {
        "columns": ["a", "b", "target"],
        "dtypes": {"a": "float64", "b": "int64", "target": "object"},
        "nunique": {"a": 10, "b": 5, "target": 3},
        "numeric_cols": ["a", "b"],
        "identifier_like_columns": [],
        "missing_percent": {},
    }
    plan = {
        "inferred_target": "target",
        "task_type": "multiclass_classification",
        "model_candidates": [{"model_name": "logistic_regression", "reason_md": "Baseline.", "params": {}}],
    }
    out = normalize_plan_dict(plan, profile=profile)
    assert out["inferred_target"] == "target"
    assert out["task_type"] == "multiclass_classification"
    assert len(out["feature_transforms"]) >= 2  # a, b (target dropped from features)
    assert len(out["model_candidates"]) > 0


def test_normalize_plan_dict_column_name_alias():
    """column_name is normalized to name."""
    profile = {"columns": ["x", "y"], "dtypes": {}, "nunique": {}, "numeric_cols": ["x", "y"], "identifier_like_columns": [], "missing_percent": {}}
    plan = {
        "inferred_target": "y",
        "task_type": "regression",
        "feature_transforms": [{"column_name": "x", "kind": "continuous", "drop": False}],
    }
    out = normalize_plan_dict(plan, profile=profile)
    assert out["feature_transforms"][0]["name"] == "x"


def test_normalize_plan_dict_infers_target_from_profile():
    """When inferred_target missing, uses profile."""
    profile = {
        "columns": ["a", "b", "c"],
        "identifier_like_columns": [],
        "dtypes": {},
        "nunique": {"c": 2},
        "numeric_cols": ["a", "b"],
        "missing_percent": {},
    }
    plan = {"task_type": "binary_classification"}
    out = normalize_plan_dict(plan, profile=profile)
    assert out["inferred_target"] == "c"
