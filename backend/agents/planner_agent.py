"""
Planner agent for Intent2Model.

Converts dataset profile + user answers → PipelineConfig using LLM with rule-based validation.
"""

import json
from typing import Dict, Optional
from schemas.pipeline_schema import PipelineConfig, UserIntent
from agents.llm_interface import LLMInterface


def plan_pipeline(
    profile: Dict,
    user_intent: UserIntent,
    llm_provider: str = "gemini"
) -> PipelineConfig:
    """
    Convert dataset profile + user intent → PipelineConfig using LLM.
    
    Args:
        profile: Dataset profile from profiler
        user_intent: UserIntent with answers
        llm_provider: LLM provider to use ("openai", "gemini", "groq")
        
    Returns:
        Validated PipelineConfig
    """
    # Build prompt for LLM
    prompt = _build_planning_prompt(profile, user_intent)
    
    # Get LLM response
    llm = LLMInterface(provider=llm_provider)
    system_prompt = _get_planning_system_prompt()
    
    try:
        response = llm.generate(prompt, system_prompt)
        # Try to extract JSON from response
        config_dict = _extract_json_from_response(response)
    except Exception as e:
        # Fallback to rule-based planning if LLM fails
        print(f"LLM planning failed: {e}. Using rule-based fallback.")
        config_dict = _rule_based_plan(profile, user_intent)
    
    # Validate and apply sanity checks
    config_dict = _apply_sanity_checks(config_dict, profile, user_intent)
    
    # Create and validate PipelineConfig
    try:
        config = PipelineConfig(**config_dict)
    except Exception as e:
        # If validation fails, use rule-based fallback
        print(f"PipelineConfig validation failed: {e}. Using rule-based fallback.")
        config_dict = _rule_based_plan(profile, user_intent)
        config = PipelineConfig(**config_dict)
    
    return config


def _build_planning_prompt(profile: Dict, user_intent: UserIntent) -> str:
    """Build prompt for LLM planning."""
    prompt = f"""Given the following dataset profile and user intent, generate a pipeline configuration.

Dataset Profile:
- Rows: {profile.get('n_rows')}
- Columns: {profile.get('n_cols')}
- Numeric columns: {', '.join(profile.get('numeric_cols', [])[:10])}
- Categorical columns: {', '.join(profile.get('categorical_cols', [])[:10])}
- Missing values: {len([k for k, v in profile.get('missing_percent', {}).items() if v > 0])} columns have missing values

User Intent:
- Target column: {user_intent.target_column or 'Not specified'}
- Task type: {user_intent.task_type or 'Not specified'}
- Priority metric: {user_intent.priority_metric or 'Not specified'}
- Business context: {user_intent.business_context or 'Not specified'}

Generate a JSON configuration with the following structure:
{{
  "task": "classification" or "regression",
  "target": "column_name",
  "metric": "accuracy|precision|recall|f1|roc_auc|rmse|r2|mae",
  "preprocessing": ["standard_scaler", "one_hot", "imputer"],
  "model_candidates": ["model_name"],
  "cv_strategy": "stratified_kfold" or "kfold"
}}

Rules:
1. Task must match user intent or be inferred from target column type
2. Metric must match task type
3. Use standard_scaler for numeric features, one_hot for categorical
4. Use imputer if missing values > 10%
5. For small datasets (<100 rows), prefer simpler models
6. For classification with imbalance, prefer recall or f1 over accuracy

Output ONLY valid JSON, no other text."""
    
    return prompt


def _get_planning_system_prompt() -> str:
    """Get system prompt for planning."""
    return """You are an expert ML engineer creating pipeline configurations.
Your output must be valid JSON that matches the PipelineConfig schema exactly.
Be precise and follow all rules."""


def _extract_json_from_response(response: str) -> Dict:
    """Extract JSON from LLM response."""
    # Try to find JSON in response
    import re
    
    # Look for JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    # Try parsing entire response as JSON
    try:
        return json.loads(response)
    except:
        pass
    
    raise ValueError("Could not extract JSON from LLM response")


def _rule_based_plan(profile: Dict, user_intent: UserIntent) -> Dict:
    """
    Rule-based fallback planning without LLM.
    """
    # Determine task
    task = user_intent.task_type
    if not task:
        # Infer from target column
        target = user_intent.target_column
        if target and target in profile.get("numeric_cols", []):
            task = "regression"
        else:
            task = "classification"
    
    # Determine metric
    metric = user_intent.priority_metric
    if not metric:
        if task == "classification":
            metric = "accuracy"
        else:
            metric = "r2"
    
    # Determine preprocessing
    preprocessing = []
    missing_cols = [k for k, v in profile.get("missing_percent", {}).items() if v > 10]
    if missing_cols:
        preprocessing.append("imputer")
    
    if profile.get("numeric_cols"):
        preprocessing.append("standard_scaler")
    if profile.get("categorical_cols"):
        preprocessing.append("one_hot")
    
    # Determine model
    n_rows = profile.get("n_rows", 0)
    if task == "classification":
        if n_rows < 100:
            model = "logistic_regression"
        else:
            model = "random_forest"
    else:
        if n_rows < 100:
            model = "linear_regression"
        else:
            model = "random_forest"
    
    # Determine CV strategy
    cv_strategy = "stratified_kfold" if task == "classification" else "kfold"
    
    return {
        "task": task,
        "target": user_intent.target_column or profile.get("candidate_targets", [""])[0] if profile.get("candidate_targets") else "",
        "metric": metric,
        "preprocessing": preprocessing,
        "model_candidates": [model],
        "cv_strategy": cv_strategy
    }


def _apply_sanity_checks(config_dict: Dict, profile: Dict, user_intent: UserIntent) -> Dict:
    """
    Apply rule-based sanity checks to config.
    """
    # Check target exists
    target = config_dict.get("target")
    if target and target not in profile.get("numeric_cols", []) + profile.get("categorical_cols", []):
        # Try to use user intent target or candidate
        if user_intent.target_column:
            config_dict["target"] = user_intent.target_column
        elif profile.get("candidate_targets"):
            config_dict["target"] = profile["candidate_targets"][0]
    
    # Check metric matches task
    task = config_dict.get("task")
    metric = config_dict.get("metric")
    classification_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    regression_metrics = ["rmse", "r2", "mae"]
    
    if task == "classification" and metric not in classification_metrics:
        config_dict["metric"] = "accuracy"
    elif task == "regression" and metric not in regression_metrics:
        config_dict["metric"] = "r2"
    
    # Check model compatibility with data size
    n_rows = profile.get("n_rows", 0)
    model_candidates = config_dict.get("model_candidates", [])
    
    if n_rows < 50 and "xgboost" in model_candidates:
        # XGBoost needs more data
        model_candidates = [m for m in model_candidates if m != "xgboost"]
        if not model_candidates:
            model_candidates = ["logistic_regression"] if task == "classification" else ["linear_regression"]
        config_dict["model_candidates"] = model_candidates
    
    return config_dict
