"""
Explainer agent for Intent2Model.

Explains model results, feature importance, and provides recommendations.
"""

from typing import Dict, Optional, List
from agents.llm_interface import LLMInterface


def explain_results(
    metrics: Dict,
    warnings: List[str],
    feature_importance: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None,
    llm_provider: str = "gemini"
) -> Dict[str, str]:
    """
    Generate natural language explanation of model results.
    
    Args:
        metrics: Dictionary of model metrics
        warnings: List of evaluator warnings
        feature_importance: Optional dictionary of feature importances
        config: Optional pipeline configuration used
        llm_provider: LLM provider to use
        
    Returns:
        Dictionary with:
        - summary: Brief summary
        - explanation: Detailed explanation
        - recommendations: Improvement recommendations
        - markdown_report: Full markdown report
    """
    # Build explanation prompt
    prompt = _build_explanation_prompt(metrics, warnings, feature_importance, config)
    
    # Get LLM explanation
    llm = LLMInterface(provider=llm_provider)
    system_prompt = _get_explanation_system_prompt()
    
    try:
        llm_explanation = llm.generate(prompt, system_prompt)
    except Exception as e:
        # Fallback to rule-based explanation
        print(f"LLM explanation failed: {e}. Using rule-based fallback.")
        llm_explanation = _rule_based_explanation(metrics, warnings, feature_importance, config)
    
    # Generate structured output
    summary = _generate_summary(metrics, warnings)
    explanation = llm_explanation
    recommendations = _generate_recommendations(warnings, metrics, feature_importance)
    markdown_report = _generate_markdown_report(metrics, warnings, feature_importance, explanation, recommendations)
    
    return {
        "summary": summary,
        "explanation": explanation,
        "recommendations": recommendations,
        "markdown_report": markdown_report
    }


def _build_explanation_prompt(
    metrics: Dict,
    warnings: List[str],
    feature_importance: Optional[Dict[str, float]],
    config: Optional[Dict]
) -> str:
    """Build prompt for LLM explanation."""
    prompt = f"""Explain the following machine learning model results in natural language.

Model Metrics:
{_format_metrics(metrics)}

Warnings/Issues:
{chr(10).join(f"- {w}" for w in warnings) if warnings else "None"}

Feature Importance (Top 10):
{_format_feature_importance(feature_importance) if feature_importance else "Not available"}

Pipeline Configuration:
{_format_config(config) if config else "Not specified"}

Provide a clear, non-technical explanation covering:
1. What the model does and how well it performs
2. What features are most important
3. Any concerns or limitations
4. What the metrics mean in practical terms

Write in a conversational, accessible style."""
    
    return prompt


def _get_explanation_system_prompt() -> str:
    """Get system prompt for explanation."""
    return """You are a data science educator explaining ML results to non-experts.
Be clear, concise, and avoid unnecessary jargon. Focus on practical implications."""


def _format_metrics(metrics: Dict) -> str:
    """Format metrics for display."""
    lines = []
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _format_feature_importance(feature_importance: Dict[str, float], top_n: int = 10) -> str:
    """Format feature importance for display."""
    if not feature_importance:
        return "Not available"
    
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    lines = []
    for feature, importance in sorted_features:
        lines.append(f"- {feature}: {importance:.4f}")
    return "\n".join(lines)


def _format_config(config: Dict) -> str:
    """Format config for display."""
    if not config:
        return "Not specified"
    
    lines = []
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _rule_based_explanation(
    metrics: Dict,
    warnings: List[str],
    feature_importance: Optional[Dict[str, float]],
    config: Optional[Dict]
) -> str:
    """Generate rule-based explanation without LLM."""
    explanation_parts = []
    
    # Performance summary
    if "accuracy" in metrics:
        acc = metrics["accuracy"]
        explanation_parts.append(
            f"The model achieves {acc:.1%} accuracy on the training data. "
        )
        if acc > 0.9:
            explanation_parts.append("This is a strong performance.")
        elif acc > 0.7:
            explanation_parts.append("This is a moderate performance.")
        else:
            explanation_parts.append("There is room for improvement.")
    
    if "r2" in metrics:
        r2 = metrics["r2"]
        explanation_parts.append(
            f"The model explains {r2:.1%} of the variance in the target variable (R² = {r2:.3f}). "
        )
        if r2 > 0.8:
            explanation_parts.append("This indicates a good fit.")
        elif r2 > 0.5:
            explanation_parts.append("This indicates a moderate fit.")
        else:
            explanation_parts.append("The model may need improvement.")
    
    # Feature importance
    if feature_importance:
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_features:
            explanation_parts.append(
                f"The most important features are: {', '.join([f[0] for f in top_features])}. "
            )
    
    # Warnings
    if warnings:
        explanation_parts.append(
            f"Note: {len(warnings)} potential issue(s) were detected that may affect model performance."
        )
    
    return " ".join(explanation_parts)


def _generate_summary(metrics: Dict, warnings: List[str]) -> str:
    """Generate brief summary."""
    summary_parts = []
    
    if "accuracy" in metrics:
        summary_parts.append(f"Accuracy: {metrics['accuracy']:.1%}")
    if "r2" in metrics:
        summary_parts.append(f"R²: {metrics['r2']:.3f}")
    if "f1" in metrics:
        summary_parts.append(f"F1: {metrics['f1']:.3f}")
    
    summary = " | ".join(summary_parts) if summary_parts else "Model trained successfully"
    
    if warnings:
        summary += f" ({len(warnings)} warning(s))"
    
    return summary


def _generate_recommendations(
    warnings: List[str],
    metrics: Dict,
    feature_importance: Optional[Dict[str, float]]
) -> str:
    """Generate improvement recommendations."""
    recommendations = []
    
    # Check for specific warnings and provide recommendations
    warning_lower = [w.lower() for w in warnings]
    
    if any("imbalance" in w for w in warning_lower):
        recommendations.append(
            "- Consider using class weights or resampling techniques to handle class imbalance."
        )
    
    if any("leakage" in w for w in warning_lower):
        recommendations.append(
            "- Review and remove columns that may cause data leakage (e.g., IDs, duplicates of target)."
        )
    
    if any("small" in w for w in warning_lower):
        recommendations.append(
            "- Collect more data to improve model generalization."
        )
    
    if any("missing" in w for w in warning_lower):
        recommendations.append(
            "- Consider better imputation strategies or removing columns with too many missing values."
        )
    
    # Performance-based recommendations
    if "accuracy" in metrics and metrics["accuracy"] < 0.7:
        recommendations.append(
            "- Try feature engineering or collecting more relevant features."
        )
        recommendations.append(
            "- Consider trying different model types or hyperparameter tuning."
        )
    
    if "r2" in metrics and metrics["r2"] < 0.5:
        recommendations.append(
            "- The model may benefit from additional features or non-linear transformations."
        )
    
    if not recommendations:
        recommendations.append("- Model performance looks good. Consider testing on held-out data.")
    
    return "\n".join(recommendations)


def _generate_markdown_report(
    metrics: Dict,
    warnings: List[str],
    feature_importance: Optional[Dict[str, float]],
    explanation: str,
    recommendations: str
) -> str:
    """Generate full markdown report."""
    report = ["# Model Results Report\n"]
    
    # Summary
    report.append("## Summary\n")
    report.append(_generate_summary(metrics, warnings))
    report.append("\n")
    
    # Metrics
    report.append("## Metrics\n")
    report.append("```")
    report.append(_format_metrics(metrics))
    report.append("```\n")
    
    # Explanation
    report.append("## Explanation\n")
    report.append(explanation)
    report.append("\n")
    
    # Feature Importance
    if feature_importance:
        report.append("## Top Features\n")
        report.append("```")
        report.append(_format_feature_importance(feature_importance, top_n=10))
        report.append("```\n")
    
    # Warnings
    if warnings:
        report.append("## Warnings\n")
        for warning in warnings:
            report.append(f"- ⚠️ {warning}\n")
        report.append("\n")
    
    # Recommendations
    report.append("## Recommendations\n")
    report.append(recommendations)
    report.append("\n")
    
    return "\n".join(report)
