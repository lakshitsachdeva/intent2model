"""Tests for automl_agent."""
import pytest
from agents.automl_agent import _extract_json, _map_alternate_schema
from agents.plan_normalizer import normalize_plan_dict
from schemas.pipeline_schema import AutoMLPlan


def test_extract_json_from_fenced():
    """Extract JSON from markdown fenced block."""
    text = '''```json
{"inferred_target": "x", "task_type": "regression"}
```'''
    out = _extract_json(text)
    assert out["inferred_target"] == "x"
    assert out["task_type"] == "regression"


def test_extract_json_plain():
    """Extract JSON from plain text."""
    text = 'Some text {"inferred_target": "y", "task_type": "classification"} more'
    out = _extract_json(text)
    assert out["inferred_target"] == "y"
    assert out["task_type"] == "classification"


def test_extract_json_raises_on_no_json():
    """Raises when no JSON object found."""
    with pytest.raises(ValueError, match="No JSON object"):
        _extract_json("no json here at all")


def test_map_alternate_schema_dataset_config():
    """Map wrong schema (dataset_config) to our schema."""
    wrong = {
        "task_description": None,
        "dataset_config": {"target_column": None, "problem_type": None, "features": []},
        "preprocessing_pipeline": [],
        "model_selection_strategy": {"algorithms": []},
    }
    profile = {
        "columns": ["a", "b", "variety"],
        "dtypes": {"a": "float64", "b": "float64", "variety": "object"},
        "nunique": {"a": 35, "b": 23, "variety": 3},
        "numeric_cols": ["a", "b"],
        "identifier_like_columns": [],
    }
    mapped = _map_alternate_schema(wrong, profile, "variety")
    assert mapped["inferred_target"] == "variety"
    assert mapped["task_type"] in ("binary_classification", "multiclass_classification")
    assert len(mapped["model_candidates"]) > 0


def test_full_pipeline_alternate_schema_to_automl_plan():
    """Map alternate schema -> normalize -> AutoMLPlan validates."""
    wrong = {
        "dataset_config": {"target_column": None, "problem_type": None},
        "preprocessing_pipeline": [],
        "model_selection_strategy": {"algorithms": []},
    }
    profile = {
        "columns": ["a", "b", "y"],
        "dtypes": {"a": "float64", "b": "float64", "y": "int64"},
        "nunique": {"a": 10, "b": 8, "y": 2},
        "numeric_cols": ["a", "b"],
        "identifier_like_columns": [],
        "missing_percent": {},
    }
    mapped = _map_alternate_schema(wrong, profile, "y")
    norm = normalize_plan_dict(mapped, profile=profile, requested_target="y")
    plan = AutoMLPlan(**norm)
    assert plan.inferred_target == "y"
    assert plan.task_type == "binary_classification"
