"""
Model Explainer Agent - Uses LLM to explain why models perform better/worse.
"""

from typing import Dict, List, Any, Optional
from agents.llm_interface import LLMInterface


def explain_model_performance(
    model_name: str,
    metrics: Dict[str, float],
    dataset_info: Dict[str, Any],
    task: str,
    comparison_with: Optional[List[Dict[str, Any]]] = None,
    llm_provider: str = "gemini"
) -> Dict[str, str]:
    """
    Use LLM to explain why a model performs well or poorly.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics for this model
        dataset_info: Information about the dataset (shape, columns, etc.)
        task: "classification" or "regression"
        comparison_with: Optional list of other models' results for comparison
        llm_provider: LLM provider to use
        
    Returns:
        Dictionary with:
        - explanation: Why this model performs this way
        - strengths: What this model is good at
        - weaknesses: What this model struggles with
        - recommendation: When to use this model
    """
    # Build prompt with data context
    prompt = _build_explanation_prompt(
        model_name, metrics, dataset_info, task, comparison_with
    )
    
    system_prompt = """You are an expert machine learning engineer explaining model performance.
Analyze the model's performance in the context of the dataset characteristics.
Be specific about why certain models work better for this data.
Explain in clear, non-technical language when possible, but include technical details when relevant."""
    
    llm = LLMInterface(provider=llm_provider)
    
    try:
        response = llm.generate(prompt, system_prompt)
        # Parse structured response
        return _parse_explanation(response, model_name, metrics)
    except Exception as e:
        print(f"LLM explanation failed for {model_name}: {e}")
        return _rule_based_explanation(model_name, metrics, dataset_info, task)


def _build_explanation_prompt(
    model_name: str,
    metrics: Dict[str, float],
    dataset_info: Dict[str, Any],
    task: str,
    comparison_with: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build prompt for LLM explanation."""
    
    prompt = f"""Analyze the performance of {model_name} model for a {task} task.

Dataset Information:
- Rows: {dataset_info.get('n_rows', 'unknown')}
- Columns: {dataset_info.get('n_cols', 'unknown')}
- Numeric columns: {len(dataset_info.get('numeric_cols', []))}
- Categorical columns: {len(dataset_info.get('categorical_cols', []))}
- Target column: {dataset_info.get('target', 'unknown')}
- Target unique values: {dataset_info.get('target_unique_count', 'unknown')}
- Missing values: {dataset_info.get('missing_percent', {})}

Model Performance Metrics:
"""
    
    for metric_name, metric_value in metrics.items():
        prompt += f"- {metric_name}: {metric_value:.4f}\n"
    
    if comparison_with:
        prompt += "\nComparison with other models:\n"
        for other in comparison_with:
            prompt += f"- {other.get('model_name', 'unknown')}: {other.get('primary_metric', 0):.4f}\n"
    
    prompt += f"""
Explain:
1. Why {model_name} performs this way given the dataset characteristics
2. What are the strengths of this model for this specific dataset
3. What are potential weaknesses or limitations
4. When would you recommend using this model vs others

Be specific and reference the dataset characteristics (size, feature types, etc.) in your explanation."""
    
    return prompt


def _parse_explanation(response: str, model_name: str, metrics: Dict[str, float]) -> Dict[str, str]:
    """Parse LLM response into structured format."""
    # Try to extract structured sections
    explanation = response
    
    # Try to find sections
    strengths = ""
    weaknesses = ""
    recommendation = ""
    
    if "strength" in response.lower() or "good at" in response.lower():
        # Try to extract strengths
        import re
        strength_match = re.search(r'(?:strength|good at)[:]\s*(.+?)(?:\n\n|\n(?:weakness|limitation|when)|$)', response, re.IGNORECASE | re.DOTALL)
        if strength_match:
            strengths = strength_match.group(1).strip()
    
    if "weakness" in response.lower() or "limitation" in response.lower():
        weakness_match = re.search(r'(?:weakness|limitation)[:]\s*(.+?)(?:\n\n|\n(?:recommend|when)|$)', response, re.IGNORECASE | re.DOTALL)
        if weakness_match:
            weaknesses = weakness_match.group(1).strip()
    
    if "recommend" in response.lower() or "when" in response.lower():
        rec_match = re.search(r'(?:recommend|when)[:]\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
        if rec_match:
            recommendation = rec_match.group(1).strip()
    
    return {
        "explanation": explanation,
        "strengths": strengths or "See explanation above.",
        "weaknesses": weaknesses or "See explanation above.",
        "recommendation": recommendation or "See explanation above."
    }


def _rule_based_explanation(
    model_name: str,
    metrics: Dict[str, float],
    dataset_info: Dict[str, Any],
    task: str
) -> Dict[str, str]:
    """Rule-based fallback explanation."""
    
    n_rows = dataset_info.get('n_rows', 0)
    n_cols = dataset_info.get('n_cols', 0)
    
    explanation = f"{model_name} achieved "
    if task == "classification":
        accuracy = metrics.get('accuracy', 0)
        explanation += f"{accuracy:.1%} accuracy. "
    else:
        r2 = metrics.get('r2', 0)
        explanation += f"R² of {r2:.3f}. "
    
    # Model-specific insights
    if model_name == "random_forest":
        explanation += "Random Forest is good at capturing non-linear relationships and handling mixed data types."
        strengths = "Handles non-linear patterns, robust to outliers, provides feature importance"
        weaknesses = "Can overfit on small datasets, less interpretable than linear models"
    elif model_name == "logistic_regression" or model_name == "linear_regression":
        explanation += "Linear models are interpretable and work well when relationships are approximately linear."
        strengths = "Highly interpretable, fast training, good baseline, less prone to overfitting"
        weaknesses = "Assumes linear relationships, may miss complex patterns"
    elif "xgboost" in model_name.lower():
        explanation += "XGBoost is powerful for complex patterns but requires more data."
        strengths = "Excellent performance on complex patterns, handles missing values well"
        weaknesses = "Requires more data, slower training, less interpretable"
    else:
        strengths = "See model documentation"
        weaknesses = "See model documentation"
    
    recommendation = f"For this dataset ({n_rows} rows, {n_cols} features), "
    if n_rows < 100:
        recommendation += "consider simpler models like logistic/linear regression to avoid overfitting."
    elif n_rows < 1000:
        recommendation += "random forest is a good balance of performance and robustness."
    else:
        recommendation += "you can use more complex models like XGBoost for better performance."
    
    return {
        "explanation": explanation,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendation": recommendation
    }
