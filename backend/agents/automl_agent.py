"""
AutoML Engineer Agent

LLM is responsible for reasoning & producing a structured AutoMLPlan.
Python is responsible for executing the plan.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import pandas as pd

from agents.llm_interface import LLMInterface
from agents.plan_normalizer import normalize_plan_dict
from schemas.pipeline_schema import AutoMLPlan
from utils.api_key_manager import get_api_key


def plan_automl(df: pd.DataFrame, requested_target: Optional[str] = None, llm_provider: str = "gemini") -> AutoMLPlan:
    """
    Produce an AutoMLPlan from dataset alone (optionally validating a requested_target).
    """
    profile = _profile_for_llm(df)
    prompt = _build_prompt(profile, requested_target=requested_target)
    system_prompt = _system_prompt()

    api_key = get_api_key(provider=llm_provider)
    llm = LLMInterface(provider=llm_provider, api_key=api_key)

    max_retries = 3
    last_error = None
    last_response = None

    for attempt in range(max_retries):
        try:
            response = llm.generate(prompt, system_prompt)
            last_response = response
            plan_dict = _extract_json(response)
            
            # CENTRAL NORMALIZATION LAYER (single source of truth)
            plan_dict = normalize_plan_dict(plan_dict, profile=profile, requested_target=requested_target)
            plan_dict["planning_source"] = "llm"
            plan_dict["planning_error"] = None
            
            # Validate plan (logical consistency beyond schema)
            _validate_plan(plan_dict, profile)
            
            print(f"✅ AutoML planning attempt {attempt+1} succeeded")
            return AutoMLPlan(**plan_dict)
        except Exception as e:
            last_error = e
            error_str = str(e)
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ AutoML planning attempt {attempt+1} failed:")
            print(f"   Error: {error_str[:500]}")
            if "validation error" in error_str.lower():
                print(f"   Full traceback:\n{error_details[:1000]}")
            if last_response:
                print(f"   LLM response preview: {last_response[:500]}...")

    # Hard fallback: rule-based minimal plan (marked as low confidence)
    print("⚠️  LLM planning failed after all retries. Using rule-based fallback (low confidence).")
    plan_dict = _rule_based_plan(profile, requested_target=requested_target)
    plan_dict["planning_source"] = "fallback"
    plan_dict["planning_error"] = str(last_error)[:500]
    
    # CENTRAL NORMALIZATION LAYER (same path for fallback)
    plan_dict = normalize_plan_dict(plan_dict, profile=profile, requested_target=requested_target)
    
    # Validate fallback plan
    try:
        _validate_plan(plan_dict, profile)
        return AutoMLPlan(**plan_dict)
    except Exception as e2:
        # FAIL LOUDLY - no silent fallback
        error_msg = (
            f"AutoML planning FAILED completely.\n"
            f"LLM errors: {str(last_error)[:500]}\n"
            f"Fallback validation error: {str(e2)[:500]}\n"
            f"System cannot proceed safely. Please check dataset and LLM configuration."
        )
        print(f"💥 {error_msg}")
        raise RuntimeError(error_msg)


def _profile_for_llm(df: pd.DataFrame) -> Dict[str, Any]:
    # Lightweight, dataset-agnostic stats (safe to send to LLM)
    n_rows, n_cols = df.shape
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    missing_pct = {c: float(df[c].isna().mean() * 100.0) for c in df.columns}

    # small distribution summaries
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_summary: Dict[str, Any] = {}
    for c in numeric_cols[:40]:
        s = pd.to_numeric(df[c], errors="coerce")
        numeric_summary[c] = {
            "min": _to_float(s.min()),
            "p25": _to_float(s.quantile(0.25)),
            "median": _to_float(s.median()),
            "p75": _to_float(s.quantile(0.75)),
            "max": _to_float(s.max()),
            "skew": _to_float(s.dropna().skew()) if s.dropna().shape[0] >= 20 else None,
            "outlier_hint_iqr": _outlier_hint_iqr(s),
        }

    # top values for categoricals
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    categorical_top: Dict[str, Any] = {}
    for c in categorical_cols[:40]:
        vc = df[c].astype("string").value_counts(dropna=True).head(5)
        categorical_top[c] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]

    # likely identifiers
    id_like = []
    for c in df.columns:
        if nunique.get(c, 0) >= max(0.95 * n_rows, 25) and nunique.get(c, 0) > 0:
            id_like.append(c)

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "nunique": nunique,
        "missing_percent": missing_pct,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_summary": numeric_summary,
        "categorical_top_values": categorical_top,
        "identifier_like_columns": id_like,
    }


def _build_prompt(profile: Dict[str, Any], requested_target: Optional[str]) -> str:
    requested_target = (requested_target or "").strip()
    return f"""
You are an AutoML engineer agent.

You MUST follow this workflow:
STEP 0 Task inference (target + task type) from dataset alone; justify in markdown.
STEP 1 Dataset intelligence: inspect dtypes, uniques, missingness, distributions; classify each feature kind; detect leakage/id/outliers/skew/imbalance/high-cardinality; write findings in markdown.
STEP 2 Transformation strategy: per-feature imputation/encoding/scaling/drop decisions; justify WHY.
STEP 3 Model candidates: choose models based on dataset size/feature types/task/interpretability-performance tradeoffs; justify inclusions/exclusions.
STEP 4 Training & validation plan: CV default, metrics appropriate to task, overfit checks; justify.
STEP 5 Error/behavior analysis plan: what plots/analyses to do; justify.
STEP 6 Explainability plan: how to align post-encoding feature names; justify.

CRITICAL RULES:
- Do NOT assume scaling/encoding. Decide based on model needs & feature kinds.
- Do NOT output generic sklearn boilerplate.
- Output ONLY valid JSON matching the AutoMLPlan schema.

Dataset profile (JSON):
{json.dumps(profile)[:15000]}

Requested target (may be empty; validate against dataset if provided): {requested_target or "NONE"}

Output JSON for AutoMLPlan with:
- inferred_target (must be a column in dataset)
- task_type in [regression, binary_classification, multiclass_classification]
- markdown fields task_inference_md, dataset_intelligence_md, transformation_strategy_md, model_selection_md, training_validation_md, error_behavior_analysis_md, explainability_md
- primary_metric + additional_metrics appropriate for the task
- feature_transforms: list of FeatureTransform objects, one per column (drop=true for target and for identifiers/leakage)
- model_candidates: list of ModelCandidate objects with reasons and params (can be empty params)
""".strip()


def _system_prompt() -> str:
    return (
        "You are a senior ML engineer. "
        "You must be dataset-agnostic and data-driven. "
        "Return ONLY strict JSON that validates against the AutoMLPlan schema."
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Strict JSON extraction with validation.
    Requires presence of all top-level schema keys.
    """
    # Prefer fenced JSON, else first object
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in LLM response")
    
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM response: {str(e)}")
    
    # Validate top-level keys exist (required fields)
    required_keys = ["inferred_target", "task_type"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required top-level keys: {missing}")
    
    return data


def _validate_plan(plan_dict: Dict[str, Any], profile: Dict[str, Any]) -> None:
    """
    Plan verification layer: logical consistency beyond schema validation.
    
    Python is AUTHORITATIVE. LLM suggestions are ADVISORY.
    """
    # Verify inferred_target exists in dataset
    target = plan_dict.get("inferred_target")
    cols = profile.get("columns", [])
    if target not in cols:
        raise ValueError(f"inferred_target '{target}' not found in dataset columns: {cols}")
    
    # Block dropping high-variance or high-correlation features without justification
    feature_transforms = plan_dict.get("feature_transforms", [])
    for ft in feature_transforms:
        if ft.get("drop") and ft.get("kind") not in ["identifier", "leakage_candidate"]:
            # Allow dropping target (handled separately)
            if ft.get("name") == target:
                continue
            # Warn but don't block - LLM may have good reason
            print(f"⚠️  Warning: Dropping feature '{ft.get('name')}' of kind '{ft.get('kind')}' - ensure this is intentional")
    
    # Verify task_type matches target characteristics
    task_type = plan_dict.get("task_type")
    dtypes = profile.get("dtypes", {})
    nunique = profile.get("nunique", {})
    n_rows = int(profile.get("n_rows", 0))
    t_nuniq = int(nunique.get(target, 0))
    t_dtype = str(dtypes.get(target, ""))
    
    if task_type == "regression":
        if t_nuniq <= 2:
            print(f"⚠️  Warning: Task type 'regression' but target has only {t_nuniq} unique values - consider classification")
    elif task_type in ["binary_classification", "multiclass_classification"]:
        if "float" in t_dtype and t_nuniq > max(20, int(0.1 * n_rows)):
            print(f"⚠️  Warning: Task type '{task_type}' but target appears continuous - consider regression")


def _fix_plan_dict(plan_dict: Dict[str, Any], profile: Dict[str, Any], requested_target: Optional[str]) -> Dict[str, Any]:
    """
    DEPRECATED: Use normalize_plan_dict from plan_normalizer instead.
    Kept for backward compatibility during migration.
    """
    """
    Auto-fix common issues in LLM-generated plan_dict to make it schema-compliant.
    """
    # Ensure all required markdown fields exist
    required_md_fields = [
        "task_inference_md", "dataset_intelligence_md", "transformation_strategy_md",
        "model_selection_md", "training_validation_md", "error_behavior_analysis_md", "explainability_md"
    ]
    for field in required_md_fields:
        if field not in plan_dict or not plan_dict[field]:
            plan_dict[field] = f"LLM-generated content for {field.replace('_md', '')} (auto-filled)."
    
    # Ensure inferred_target exists and is valid
    if "inferred_target" not in plan_dict or not plan_dict["inferred_target"]:
        cols = profile.get("columns", [])
        id_like = set(profile.get("identifier_like_columns", []))
        candidates = [c for c in cols if c not in id_like]
        plan_dict["inferred_target"] = (requested_target or candidates[-1] if candidates else (cols[-1] if cols else "unknown")).strip()
    
    # Ensure task_type exists
    if "task_type" not in plan_dict:
        # Infer from target
        target = plan_dict.get("inferred_target", "")
        dtypes = profile.get("dtypes", {})
        nunique = profile.get("nunique", {})
        n_rows = int(profile.get("n_rows", 0))
        t_nuniq = int(nunique.get(target, 0))
        t_dtype = str(dtypes.get(target, ""))
        if "float" in t_dtype or ("int" in t_dtype and t_nuniq > max(20, int(0.1 * n_rows))):
            plan_dict["task_type"] = "regression"
        else:
            plan_dict["task_type"] = "binary_classification" if t_nuniq <= 2 else "multiclass_classification"
    
    # Ensure primary_metric exists
    if "primary_metric" not in plan_dict:
        task_type = plan_dict.get("task_type", "regression")
        plan_dict["primary_metric"] = "rmse" if task_type == "regression" else "f1"
    
    # Ensure additional_metrics exists
    if "additional_metrics" not in plan_dict:
        task_type = plan_dict.get("task_type", "regression")
        plan_dict["additional_metrics"] = ["mae", "r2"] if task_type == "regression" else ["precision", "recall"]
    
    # Ensure feature_transforms exists and is a list
    if "feature_transforms" not in plan_dict or not isinstance(plan_dict["feature_transforms"], list):
        # Generate minimal feature_transforms from profile
        cols = profile.get("columns", [])
        feature_transforms = []
        for c in cols:
            dtypes_dict = profile.get("dtypes", {})
            nunique_dict = profile.get("nunique", {})
            kind = "unknown"
            if c in profile.get("numeric_cols", []):
                kind = "continuous"
            else:
                nunq = int(nunique_dict.get(c, 0))
                kind = "binary_categorical" if nunq == 2 else "nominal_categorical"
            
            feature_transforms.append({
                "name": c,
                "inferred_dtype": str(dtypes_dict.get(c, "unknown")),
                "kind": kind,
                "drop": False,
                "impute": "none",
                "encode": "none",
                "scale": "none",
                "notes_md": ""
            })
        plan_dict["feature_transforms"] = feature_transforms
    
    # Ensure model_candidates exists and is a list
    if "model_candidates" not in plan_dict or not isinstance(plan_dict["model_candidates"], list):
        task_type = plan_dict.get("task_type", "regression")
        if task_type == "regression":
            plan_dict["model_candidates"] = [
                {"model_name": "linear_regression", "reason_md": "Baseline linear model.", "params": {}},
                {"model_name": "random_forest", "reason_md": "Nonlinear baseline.", "params": {}}
            ]
        else:
            plan_dict["model_candidates"] = [
                {"model_name": "logistic_regression", "reason_md": "Baseline classification model.", "params": {}},
                {"model_name": "random_forest", "reason_md": "Nonlinear baseline.", "params": {}}
            ]
    
    # Fix feature_transforms: ensure all have required fields
    if "feature_transforms" in plan_dict:
        for ft in plan_dict["feature_transforms"]:
            if not isinstance(ft, dict):
                continue
            # Normalize column_name -> name
            if "column_name" in ft and "name" not in ft:
                ft["name"] = ft.pop("column_name")
            # Add defaults for missing fields
            if "inferred_dtype" not in ft:
                ft["inferred_dtype"] = "unknown"
            if "kind" not in ft:
                ft["kind"] = "unknown"
            if "drop" not in ft:
                ft["drop"] = False
            if "impute" not in ft:
                ft["impute"] = "none"
            if "encode" not in ft:
                ft["encode"] = "none"
            if "scale" not in ft:
                ft["scale"] = "none"
            if "notes_md" not in ft:
                ft["notes_md"] = ""
    
    # Fix model_candidates: ensure all have required fields
    if "model_candidates" in plan_dict:
        for mc in plan_dict["model_candidates"]:
            if not isinstance(mc, dict):
                continue
            if "model_name" not in mc:
                continue  # Skip invalid entries
            if "reason_md" not in mc:
                mc["reason_md"] = f"Selected {mc['model_name']} for this task."
            if "params" not in mc:
                mc["params"] = {}
    
    return plan_dict


def _rule_based_plan(profile: Dict[str, Any], requested_target: Optional[str]) -> Dict[str, Any]:
    # Minimal data-driven fallback: choose target as last non-id column, infer task by nunique/ dtype
    cols = profile.get("columns", [])
    id_like = set(profile.get("identifier_like_columns", []))
    target = (requested_target or "").strip()
    if target not in cols:
        candidates = [c for c in cols if c not in id_like]
        target = candidates[-1] if candidates else (cols[-1] if cols else "")

    dtypes = profile.get("dtypes", {})
    nunique = profile.get("nunique", {})
    n_rows = int(profile.get("n_rows", 0))
    t_nuniq = int(nunique.get(target, 0))
    t_dtype = str(dtypes.get(target, ""))

    if "float" in t_dtype or ("int" in t_dtype and t_nuniq > max(20, int(0.1 * n_rows))):
        task_type = "regression"
        primary_metric = "rmse"
        additional = ["mae", "r2"]
    else:
        task_type = "binary_classification" if t_nuniq <= 2 else "multiclass_classification"
        primary_metric = "f1"
        additional = ["precision", "recall", "roc_auc"] if task_type == "binary_classification" else ["precision", "recall"]

    feature_transforms = []
    for c in cols:
        kind = "unknown"
        drop = False
        if c == target:
            drop = True
        elif c in id_like:
            kind = "identifier"
            drop = True
        elif c in profile.get("numeric_cols", []):
            kind = "continuous"
        else:
            nunq = int(nunique.get(c, 0))
            kind = "binary" if nunq == 2 else "nominal"  # Use new schema values

        missing = float(profile.get("missing_percent", {}).get(c, 0.0))
        impute = "median" if c in profile.get("numeric_cols", []) and missing > 0 else ("most_frequent" if missing > 0 else "none")
        encode = "none" if c in profile.get("numeric_cols", []) else ("one_hot" if int(nunique.get(c, 0)) <= 30 else "frequency")
        scale = "none"

        feature_transforms.append(
            {
                "name": c,
                "inferred_dtype": str(dtypes.get(c, "")),
                "kind": kind,
                "drop": drop,
                "impute": impute,
                "encode": encode,
                "scale": scale,
                "notes_md": "",
            }
        )

    # Model candidates: include baselines + 1-2 robust models
    if task_type == "regression":
        models = [
            {"model_name": "linear_regression", "reason_md": "Baseline linear model for calibration.", "params": {}},
            {"model_name": "random_forest", "reason_md": "Nonlinear baseline; robust to mixed feature effects.", "params": {"n_estimators": 300}},
            {"model_name": "gradient_boosting", "reason_md": "Strong tabular booster; can improve over RF on structured data.", "params": {}},
        ]
    else:
        models = [
            {"model_name": "logistic_regression", "reason_md": "Strong interpretable baseline for classification.", "params": {"max_iter": 2000}},
            {"model_name": "random_forest", "reason_md": "Nonlinear baseline; robust and handles interactions.", "params": {"n_estimators": 300}},
            {"model_name": "gradient_boosting", "reason_md": "Often strong on tabular classification without heavy tuning.", "params": {}},
            {"model_name": "naive_bayes", "reason_md": "Fast baseline for high-dimensional sparse encodings.", "params": {}},
        ]

    # Fallback plan MUST match schema exactly (all new fields included)
    return {
        "plan_schema_version": "v1",
        "inferred_target": target,
        "target_confidence": 0.5,  # Low confidence for fallback
        "alternative_targets": [],
        "task_type": task_type,
        "task_confidence": 0.6,  # Low confidence for fallback
        "task_inference_md": "⚠️ Rule-based fallback task inference (LLM unavailable). Low confidence.",
        "dataset_intelligence_md": "⚠️ Rule-based fallback dataset intelligence (LLM unavailable). Limited analysis.",
        "transformation_strategy_md": "⚠️ Rule-based fallback transformation strategy (LLM unavailable). Conservative defaults.",
        "model_selection_md": "⚠️ Rule-based fallback model selection (LLM unavailable). Baseline models only.",
        "training_validation_md": "Use cross-validation by default with task-appropriate metrics.",
        "error_behavior_analysis_md": "Analyze residuals/confusion matrix and error slices.",
        "explainability_md": "Use feature_importances_ when available and align post-encoding names.",
        "primary_metric": primary_metric,
        "additional_metrics": additional,
        "metric_selection_confidence": 0.7,
        "feature_transforms": [
            {
                "name": ft["name"],
                "inferred_dtype": ft["inferred_dtype"],
                "kind": ft["kind"],
                "kind_confidence": 0.6,  # Low confidence for fallback
                "drop": ft["drop"],
                "impute": ft["impute"],
                "encode": ft["encode"],
                "scale": ft["scale"],
                "notes_md": ft["notes_md"],
                "transform_confidence": 0.6,  # Low confidence for fallback
            }
            for ft in feature_transforms
        ],
        "model_candidates": models,
        "model_selection_confidence": 0.5,  # Low confidence for fallback
    }


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _outlier_hint_iqr(s: pd.Series) -> Optional[float]:
    try:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.shape[0] < 20:
            return None
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0.0
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        frac = float(((s < lo) | (s > hi)).mean())
        return frac
    except Exception:
        return None

