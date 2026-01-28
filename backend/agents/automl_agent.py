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

    try:
        response = llm.generate(prompt, system_prompt)
        plan_dict = _extract_json(response)
        plan_dict["planning_source"] = "llm"
        plan_dict["planning_error"] = None
        return AutoMLPlan(**plan_dict)
    except Exception as e:
        # Hard fallback: rule-based minimal plan (still data-driven, not template)
        plan_dict = _rule_based_plan(profile, requested_target=requested_target)
        plan_dict["planning_source"] = "fallback"
        plan_dict["planning_error"] = str(e)[:200]
        try:
            return AutoMLPlan(**plan_dict)
        except Exception as e2:
            raise RuntimeError(f"AutoML planning failed (LLM and fallback). LLM err={e}; fallback err={e2}")


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
    # Prefer fenced JSON, else first object
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(m.group(0))


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
            kind = "binary_categorical" if nunq == 2 else ("nominal_categorical" if nunq <= 30 else "nominal_categorical")

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

    return {
        "inferred_target": target,
        "task_type": task_type,
        "task_inference_md": "Rule-based fallback task inference (LLM unavailable).",
        "dataset_intelligence_md": "Rule-based fallback dataset intelligence (LLM unavailable).",
        "transformation_strategy_md": "Rule-based fallback transformation strategy (LLM unavailable).",
        "model_selection_md": "Rule-based fallback model selection (LLM unavailable).",
        "training_validation_md": "Use cross-validation by default with task-appropriate metrics.",
        "error_behavior_analysis_md": "Analyze residuals/confusion matrix and error slices.",
        "explainability_md": "Use feature_importances_ when available and align post-encoding names.",
        "primary_metric": primary_metric,
        "additional_metrics": additional,
        "feature_transforms": feature_transforms,
        "model_candidates": models,
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

