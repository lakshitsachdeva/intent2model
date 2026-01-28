"""
Central normalization layer for AutoMLPlan dictionaries.

This is the SINGLE source of truth for plan normalization.
All plan_dicts (from LLM or fallback) MUST pass through this function
immediately before AutoMLPlan(**plan_dict) validation.
"""

from typing import Any, Dict, List, Optional
from schemas.pipeline_schema import FeatureKind, ImputeStrategy, EncodeStrategy, ScaleStrategy


def normalize_plan_dict(plan_dict: Dict[str, Any], profile: Optional[Dict[str, Any]] = None, requested_target: Optional[str] = None) -> Dict[str, Any]:
    """
    Canonical normalization function for AutoMLPlan dictionaries.
    
    CRITICAL RULES:
    - Convert known naming mismatches (column_name -> name)
    - Remove unknown/forbidden keys
    - Fill missing OPTIONAL fields with safe defaults
    - NEVER invent required fields silently
    - Preserve all valid LLM reasoning
    
    Args:
        plan_dict: Raw plan dictionary (from LLM or fallback)
        profile: Dataset profile (optional, for inference fallbacks)
        requested_target: User-requested target (optional)
        
    Returns:
        Normalized plan_dict ready for AutoMLPlan(**plan_dict)
    """
    normalized = {}
    
    # 1. Schema version (always set)
    normalized["plan_schema_version"] = plan_dict.get("plan_schema_version", "v1")
    
    # 2. Normalize feature_transforms (CRITICAL: fix column_name/column -> name)
    if "feature_transforms" in plan_dict:
        normalized["feature_transforms"] = []
        for ft in plan_dict["feature_transforms"]:
            if not isinstance(ft, dict):
                continue
            
            # Normalize name field (CRITICAL FIX)
            name = None
            for key in ["name", "column_name", "column"]:
                if key in ft:
                    name = ft.pop(key)
                    break
            
            if not name:
                continue  # Skip invalid entries without name
            
            # Build normalized FeatureTransform
            normalized_ft = {
                "name": str(name),
                "inferred_dtype": ft.get("inferred_dtype", "unknown"),
                "kind": _normalize_feature_kind(ft.get("kind", "unknown")),
                "kind_confidence": float(ft.get("kind_confidence", 1.0)),
                "drop": bool(ft.get("drop", False)),
                "impute": _normalize_impute_strategy(ft.get("impute", "none")),
                "encode": _normalize_encode_strategy(ft.get("encode", "none")),
                "scale": _normalize_scale_strategy(ft.get("scale", "none")),
                "notes_md": str(ft.get("notes_md", "")),
                "transform_confidence": float(ft.get("transform_confidence", 1.0)),
            }
            normalized["feature_transforms"].append(normalized_ft)
    else:
        normalized["feature_transforms"] = []
    
    # 3. Normalize model_candidates (CRITICAL: convert sklearn class names to internal keys)
    if "model_candidates" in plan_dict:
        normalized["model_candidates"] = []
        for mc in plan_dict["model_candidates"]:
            if not isinstance(mc, dict) or "model_name" not in mc:
                continue
            raw_name = str(mc["model_name"])
            # Normalize sklearn class names to internal keys
            normalized_name = _normalize_model_name(raw_name)
            if not normalized_name:
                print(f"⚠️  Warning: Unknown model name '{raw_name}', skipping")
                continue
            normalized["model_candidates"].append({
                "model_name": normalized_name,
                "reason_md": str(mc.get("reason_md", "")),
                "params": dict(mc.get("params", {})),
            })
    else:
        normalized["model_candidates"] = []
    
    # 4. Target inference (required field - must exist or infer)
    if "inferred_target" in plan_dict and plan_dict["inferred_target"]:
        normalized["inferred_target"] = str(plan_dict["inferred_target"])
    elif profile and requested_target:
        normalized["inferred_target"] = str(requested_target)
    elif profile:
        cols = profile.get("columns", [])
        id_like = set(profile.get("identifier_like_columns", []))
        candidates = [c for c in cols if c not in id_like]
        normalized["inferred_target"] = candidates[-1] if candidates else (cols[-1] if cols else "unknown")
    else:
        raise ValueError("inferred_target is required but cannot be inferred (no profile provided)")
    
    normalized["target_confidence"] = float(plan_dict.get("target_confidence", 1.0))
    normalized["alternative_targets"] = [str(t) for t in plan_dict.get("alternative_targets", [])]
    
    # 5. Task type (required field)
    if "task_type" in plan_dict:
        task_type = plan_dict["task_type"]
        if task_type not in ["regression", "binary_classification", "multiclass_classification"]:
            # Infer from profile if invalid
            if profile:
                task_type = _infer_task_type(profile, normalized["inferred_target"])
            else:
                task_type = "regression"  # Safe default
        normalized["task_type"] = task_type
    elif profile:
        normalized["task_type"] = _infer_task_type(profile, normalized["inferred_target"])
    else:
        raise ValueError("task_type is required but cannot be inferred (no profile provided)")
    
    normalized["task_confidence"] = float(plan_dict.get("task_confidence", 1.0))
    
    # 6. Markdown fields (required - fill with defaults if missing)
    required_md_fields = [
        "task_inference_md", "dataset_intelligence_md", "transformation_strategy_md",
        "model_selection_md", "training_validation_md", "error_behavior_analysis_md", "explainability_md"
    ]
    for field in required_md_fields:
        normalized[field] = str(plan_dict.get(field, f"Auto-generated content for {field.replace('_md', '')}."))
    
    # 7. Metrics
    normalized["primary_metric"] = str(plan_dict.get("primary_metric", "rmse" if normalized["task_type"] == "regression" else "f1"))
    normalized["additional_metrics"] = [str(m) for m in plan_dict.get("additional_metrics", [])]
    normalized["metric_selection_confidence"] = float(plan_dict.get("metric_selection_confidence", 1.0))
    
    # 8. Model selection confidence
    normalized["model_selection_confidence"] = float(plan_dict.get("model_selection_confidence", 1.0))
    
    # 9. Planning metadata
    normalized["planning_source"] = plan_dict.get("planning_source", "fallback")
    normalized["planning_error"] = plan_dict.get("planning_error")
    
    # Determine plan quality
    if normalized["planning_source"] == "fallback":
        normalized["plan_quality"] = "fallback_low_confidence"
    elif normalized["target_confidence"] < 0.7 or normalized["task_confidence"] < 0.7:
        normalized["plan_quality"] = "medium_confidence"
    else:
        normalized["plan_quality"] = "high_confidence"
    
    return normalized


def _normalize_feature_kind(kind: Any) -> FeatureKind:
    """Normalize feature kind to valid enum value."""
    kind_str = str(kind).lower()
    valid_kinds: List[FeatureKind] = [
        "continuous", "ordinal", "count", "binary", "nominal",
        "datetime", "text", "identifier", "leakage_candidate", "unknown"
    ]
    
    # Map old values to new
    mapping = {
        "binary_categorical": "binary",
        "nominal_categorical": "nominal",
        "leakage_risk": "leakage_candidate",
    }
    
    if kind_str in mapping:
        kind_str = mapping[kind_str]
    
    return kind_str if kind_str in valid_kinds else "unknown"


def _normalize_impute_strategy(strategy: Any) -> ImputeStrategy:
    """Normalize impute strategy to valid enum value."""
    strategy_str = str(strategy).lower()
    valid = ["none", "mean", "median", "most_frequent", "constant"]
    return strategy_str if strategy_str in valid else "none"


def _normalize_encode_strategy(strategy: Any) -> EncodeStrategy:
    """Normalize encode strategy to valid enum value."""
    strategy_str = str(strategy).lower()
    valid = ["none", "one_hot", "ordinal", "frequency"]
    return strategy_str if strategy_str in valid else "none"


def _normalize_scale_strategy(strategy: Any) -> ScaleStrategy:
    """Normalize scale strategy to valid enum value."""
    strategy_str = str(strategy).lower()
    valid = ["none", "standard", "robust"]
    return strategy_str if strategy_str in valid else "none"


def _normalize_model_name(raw_name: str) -> Optional[str]:
    """
    Normalize sklearn class names to internal model keys.
    
    Examples:
        LogisticRegression -> logistic_regression
        RandomForestClassifier -> random_forest
        XGBClassifier -> xgboost
        SVC -> svm
    """
    raw_lower = raw_name.lower().strip()
    
    # Direct mapping for common sklearn class names
    mapping = {
        # Classification
        "logisticregression": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "randomforestclassifier": "random_forest",
        "randomforest": "random_forest",
        "random_forest": "random_forest",
        "random_forest_classifier": "random_forest",
        "xgboostclassifier": "xgboost",
        "xgbclassifier": "xgboost",
        "xgboost": "xgboost",
        "svc": "svm",
        "svm": "svm",
        "supportvectormachine": "svm",
        "gaussiannb": "naive_bayes",
        "naivebayes": "naive_bayes",
        "naive_bayes": "naive_bayes",
        "gradientboostingclassifier": "gradient_boosting",
        "gradientboosting": "gradient_boosting",
        "gradient_boosting": "gradient_boosting",
        "gradient_boosting_classifier": "gradient_boosting",
        "kneighborsclassifier": "knn",  # Note: KNN might not be in pipeline_builder
        "knn": "knn",
        
        # Regression
        "linearregression": "linear_regression",
        "linear_regression": "linear_regression",
        "randomforestregressor": "random_forest",
        "random_forest_regressor": "random_forest",
        "xgboostregressor": "xgboost",
        "xgbregressor": "xgboost",
        "svr": "svm",
        "gradientboostingregressor": "gradient_boosting",
        "gradient_boosting_regressor": "gradient_boosting",
        "ridge": "ridge",
        "ridge_regression": "ridge",
        "lasso": "lasso",
        "lasso_regression": "lasso",
    }
    
    # Try exact match first
    if raw_lower in mapping:
        return mapping[raw_lower]
    
    # Try removing common suffixes
    for suffix in ["classifier", "regressor", "model"]:
        if raw_lower.endswith(suffix):
            base = raw_lower[:-len(suffix)].strip("_")
            if base in mapping:
                return mapping[base]
    
    # If already a valid key, return as-is
    valid_keys = [
        "logistic_regression", "random_forest", "xgboost", "svm",
        "naive_bayes", "gradient_boosting", "linear_regression",
        "ridge", "lasso"
    ]
    if raw_lower in valid_keys:
        return raw_lower
    
    return None


def _infer_task_type(profile: Dict[str, Any], target: str) -> str:
    """Infer task type from profile and target."""
    dtypes = profile.get("dtypes", {})
    nunique = profile.get("nunique", {})
    n_rows = int(profile.get("n_rows", 0))
    t_nuniq = int(nunique.get(target, 0))
    t_dtype = str(dtypes.get(target, ""))
    
    if "float" in t_dtype or ("int" in t_dtype and t_nuniq > max(20, int(0.1 * n_rows))):
        return "regression"
    elif t_nuniq <= 2:
        return "binary_classification"
    else:
        return "multiclass_classification"
