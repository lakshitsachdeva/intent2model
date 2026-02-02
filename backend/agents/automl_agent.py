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

    # Gemini CLI often returns "I am ready..." or {} instead of JSON. Use API when any key is set.
    api_key = get_api_key(provider="gemini")
    if llm_provider == "gemini_cli" and api_key:
        llm_provider = "gemini"
        print("   Using Gemini API for planning (CLI unreliable for structured JSON)")
    elif not api_key:
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

            # Map alternate LLM schemas (task_description, dataset_config, etc.) to our schema
            plan_dict = _map_alternate_schema(plan_dict, profile=profile, requested_target=requested_target)

            # Set planning_source BEFORE normalization (so it's preserved)
            plan_dict["planning_source"] = "llm"
            plan_dict["planning_error"] = None

            # CENTRAL NORMALIZATION LAYER (single source of truth)
            plan_dict = normalize_plan_dict(plan_dict, profile=profile, requested_target=requested_target)
            
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
    import sys
    if sys.platform == "win32":
        print("   💡 On Windows: Add GEMINI_API_KEY or GOOGLE_API_KEY to .env (project root) for reliable LLM planning.")
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
    cols = profile.get("columns", [])
    example_target = requested_target or (cols[-1] if cols else "target_col")
    # Build example feature_transforms from actual columns (so LLM sees real names)
    example_transforms = []
    for c in cols[:6]:
        is_target = c == example_target
        is_num = c in profile.get("numeric_cols", [])
        example_transforms.append({
            "name": c,
            "inferred_dtype": str(profile.get("dtypes", {}).get(c, "float64")),
            "kind": "continuous" if is_num else "nominal",
            "kind_confidence": 0.9,
            "drop": is_target,
            "impute": "median" if is_num else "none",
            "encode": "none" if is_num else "one_hot",
            "scale": "standard" if is_num else "none",
            "notes_md": "",
            "transform_confidence": 0.8,
        })
    example_json = {
        "plan_schema_version": "v1",
        "inferred_target": example_target,
        "target_confidence": 0.95,
        "alternative_targets": [],
        "task_type": "multiclass_classification",
        "task_confidence": 0.9,
        "task_inference_md": "Target has few unique values; classification task.",
        "dataset_intelligence_md": "Dataset analyzed. Numeric and categorical columns identified.",
        "transformation_strategy_md": "StandardScaler for numeric, one_hot for categorical.",
        "model_selection_md": "Logistic regression baseline, random_forest, gradient_boosting.",
        "training_validation_md": "Stratified K-fold CV, primary metric f1.",
        "error_behavior_analysis_md": "Confusion matrix, per-class metrics.",
        "explainability_md": "Feature importances from tree models.",
        "primary_metric": "f1",
        "additional_metrics": ["precision", "recall"],
        "metric_selection_confidence": 0.9,
        "feature_transforms": example_transforms,
        "model_candidates": [
            {"model_name": "logistic_regression", "reason_md": "Baseline.", "params": {}},
            {"model_name": "random_forest", "reason_md": "Nonlinear.", "params": {}},
        ],
        "model_selection_confidence": 0.8,
    }
    return f"""TASK: Analyze the dataset below and output a single JSON object. Use EXACTLY this schema - no other format.

CRITICAL: Your JSON MUST have top-level keys "inferred_target" (a column name from the dataset) and "task_type" (one of: regression, binary_classification, multiclass_classification). Do NOT use task_description, dataset_config, preprocessing_pipeline, model_selection_strategy, evaluation_config, or output_config - those are WRONG.

Dataset profile:
{json.dumps(profile)[:15000]}

Requested target: {requested_target or "NONE"}

Required keys: plan_schema_version, inferred_target, target_confidence, alternative_targets, task_type, task_confidence, task_inference_md, dataset_intelligence_md, transformation_strategy_md, model_selection_md, training_validation_md, error_behavior_analysis_md, explainability_md, primary_metric, additional_metrics, metric_selection_confidence, feature_transforms, model_candidates, model_selection_confidence.

Example (adapt column names to the actual dataset):
{json.dumps(example_json, indent=2)[:2000]}

Rules: inferred_target must be a column from the dataset. task_type in [regression, binary_classification, multiclass_classification]. feature_transforms: one object per column with name, inferred_dtype, kind, kind_confidence, drop, impute, encode, scale, notes_md, transform_confidence. model_candidates: list of {{model_name, reason_md, params}}.

Output ONLY the JSON object, nothing else:""".strip()


def _system_prompt() -> str:
    return (
        "You are an AutoML planning agent. Your ONLY job is to output a valid JSON object. "
        "NEVER ask questions. NEVER say 'I am ready' or 'please provide'. NEVER search for files or schemas. "
        "The user prompt contains the dataset and a JSON example. Output ONLY the JSON object, no other text."
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from LLM response.
    Does NOT require schema keys - normalization handles missing fields.
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
    
    return data


def _map_alternate_schema(plan_dict: Dict[str, Any], profile: Dict[str, Any], requested_target: Optional[str]) -> Dict[str, Any]:
    """
    Map LLM responses that use a different schema (e.g. task_description, dataset_config,
    preprocessing_pipeline, model_selection_strategy) to our AutoMLPlan schema.
    """
    # Already in our schema (both required fields present and non-empty)
    if plan_dict.get("inferred_target") and plan_dict.get("task_type"):
        return plan_dict

    mapped: Dict[str, Any] = {}
    cols = profile.get("columns", [])
    id_like = set(profile.get("identifier_like_columns", []))

    # inferred_target: from dataset_config.target_column, or infer from profile
    target = None
    if "dataset_config" in plan_dict and isinstance(plan_dict["dataset_config"], dict):
        target = plan_dict["dataset_config"].get("target_column")
    if not target and requested_target and requested_target in cols:
        target = requested_target
    if not target:
        candidates = [c for c in cols if c not in id_like]
        target = candidates[-1] if candidates else (cols[-1] if cols else "unknown")

    mapped["inferred_target"] = str(target).strip() if target else "unknown"

    # task_type: from dataset_config.problem_type, or infer from profile
    task_type = None
    if "dataset_config" in plan_dict and isinstance(plan_dict["dataset_config"], dict):
        pt = plan_dict["dataset_config"].get("problem_type")
        if pt:
            pt_lower = str(pt).lower()
            if "regress" in pt_lower:
                task_type = "regression"
            elif "binary" in pt_lower or "classif" in pt_lower:
                task_type = "binary_classification" if profile.get("nunique", {}).get(target, 0) <= 2 else "multiclass_classification"
            else:
                task_type = "multiclass_classification"

    if not task_type and profile:
        dtypes = profile.get("dtypes", {})
        nunique = profile.get("nunique", {})
        n_rows = int(profile.get("n_rows", 0))
        t_nuniq = int(nunique.get(target, 0))
        t_dtype = str(dtypes.get(target, ""))
        if "float" in t_dtype or ("int" in t_dtype and t_nuniq > max(20, int(0.1 * n_rows))):
            task_type = "regression"
        else:
            task_type = "binary_classification" if t_nuniq <= 2 else "multiclass_classification"

    mapped["task_type"] = task_type or "regression"

    # feature_transforms: from preprocessing_pipeline or dataset_config.features, or generate from profile
    if "feature_transforms" in plan_dict and plan_dict["feature_transforms"]:
        mapped["feature_transforms"] = plan_dict["feature_transforms"]
    elif "preprocessing_pipeline" in plan_dict and plan_dict["preprocessing_pipeline"]:
        # Convert preprocessing_pipeline items to feature_transforms (best-effort)
        mapped["feature_transforms"] = []
        for item in plan_dict["preprocessing_pipeline"]:
            if isinstance(item, dict) and item.get("name"):
                mapped["feature_transforms"].append({
                    "name": item["name"],
                    "inferred_dtype": item.get("inferred_dtype", "unknown"),
                    "kind": item.get("kind", "unknown"),
                    "kind_confidence": 0.8,
                    "drop": item.get("drop", False),
                    "impute": item.get("impute", "none"),
                    "encode": item.get("encode", "none"),
                    "scale": item.get("scale", "none"),
                    "notes_md": item.get("notes_md", ""),
                    "transform_confidence": 0.8,
                })
    # else: normalize_plan_dict will generate from profile

    # model_candidates: from model_selection_strategy.algorithms
    if "model_candidates" in plan_dict and plan_dict["model_candidates"]:
        mapped["model_candidates"] = plan_dict["model_candidates"]
    elif "model_selection_strategy" in plan_dict and isinstance(plan_dict["model_selection_strategy"], dict):
        algs = plan_dict["model_selection_strategy"].get("algorithms", [])
        if algs:
            mapped["model_candidates"] = [
                {"model_name": a if isinstance(a, str) else a.get("name", "random_forest"), "reason_md": "From LLM.", "params": {}}
                for a in algs
            ]

    # When mapping from alternate schema, ensure we have models (LLM often returns empty)
    if not mapped.get("model_candidates"):
        tt = mapped.get("task_type", "regression")
        if tt == "regression":
            mapped["model_candidates"] = [
                {"model_name": "linear_regression", "reason_md": "Baseline.", "params": {}},
                {"model_name": "ridge", "reason_md": "L2 regularized.", "params": {}},
                {"model_name": "random_forest", "reason_md": "Nonlinear.", "params": {}},
                {"model_name": "gradient_boosting", "reason_md": "Strong tabular.", "params": {}},
            ]
        else:
            mapped["model_candidates"] = [
                {"model_name": "logistic_regression", "reason_md": "Baseline.", "params": {}},
                {"model_name": "random_forest", "reason_md": "Nonlinear.", "params": {}},
                {"model_name": "gradient_boosting", "reason_md": "Strong tabular.", "params": {}},
            ]

    # Copy any other fields we recognize
    for k in ["plan_schema_version", "target_confidence", "alternative_targets", "task_confidence",
              "task_inference_md", "dataset_intelligence_md", "transformation_strategy_md",
              "model_selection_md", "training_validation_md", "error_behavior_analysis_md", "explainability_md",
              "primary_metric", "additional_metrics", "metric_selection_confidence", "model_selection_confidence"]:
        if k in plan_dict and plan_dict[k] is not None:
            mapped[k] = plan_dict[k]

    if mapped.get("inferred_target") and mapped.get("task_type"):
        print("   Mapped alternate LLM schema to AutoMLPlan format")
    return mapped


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
        scale = "standard" if (kind == "continuous" and task_type == "regression") else "none"

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

    # Model candidates: include baselines + ridge/lasso (good for correlated features like iris)
    if task_type == "regression":
        models = [
            {"model_name": "linear_regression", "reason_md": "Baseline linear model for calibration.", "params": {}},
            {"model_name": "ridge", "reason_md": "L2 regularized; good when features are correlated (e.g. iris).", "params": {}},
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
    # If target was explicitly requested, we can have high confidence even in fallback
    target_conf = 0.95 if requested_target else 0.7  # High if explicit, medium if inferred
    task_conf = 0.9  # Rule-based task inference is usually confident (regression vs classification)
    
    return {
        "plan_schema_version": "v1",
        "inferred_target": target,
        "target_confidence": target_conf,
        "alternative_targets": [],
        "task_type": task_type,
        "task_confidence": task_conf,
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

