"""
Error analysis agent for Intent2Model.

Uses LLM to analyze training errors and provide helpful explanations.
"""

from typing import Dict, Any, Optional
from agents.llm_interface import LLMInterface


def analyze_training_error(
    error: Exception,
    error_msg: str,
    dataset_info: Dict[str, Any],
    target_column: str,
    task_type: str,
    llm_provider: str = "gemini"
) -> Dict[str, str]:
    """
    Use LLM to analyze training errors and provide helpful explanations.
    
    Args:
        error: The exception that occurred
        error_msg: String representation of the error
        dataset_info: Dictionary with dataset information (shape, columns, dtypes, etc.)
        target_column: The target column being predicted
        task_type: "classification" or "regression"
        llm_provider: LLM provider to use
        
    Returns:
        Dictionary with:
        - explanation: Human-readable explanation of what went wrong
        - root_cause: The likely root cause
        - suggestions: List of suggestions to fix the issue
    """
    try:
        llm = LLMInterface(provider=llm_provider)
        
        system_prompt = """You are an expert ML engineer helping diagnose training errors.
Analyze the error and provide:
1. A clear explanation of what went wrong
2. The root cause
3. Actionable suggestions to fix it

Be specific and helpful. Focus on what the user can do to fix the issue."""

        prompt = f"""A machine learning training error occurred. Please analyze it:

**Error Message:**
{error_msg}

**Error Type:**
{type(error).__name__}

**Dataset Information:**
- Shape: {dataset_info.get('shape', 'unknown')}
- Columns: {', '.join(dataset_info.get('columns', [])[:10])}
- Target column: {target_column}
- Task type: {task_type}
- Target dtype: {dataset_info.get('target_dtype', 'unknown')}
- Target unique values: {dataset_info.get('target_unique_count', 'unknown')}
- Missing values in target: {dataset_info.get('target_missing_count', 'unknown')}

**Context:**
The user is trying to train a {task_type} model to predict "{target_column}".

Please provide:
1. A clear explanation of what went wrong (in simple terms)
2. The root cause (what specifically caused this error)
3. 2-3 actionable suggestions to fix the issue

Format your response as JSON:
{{
  "explanation": "clear explanation",
  "root_cause": "root cause",
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}}"""

        response = llm.generate(prompt, system_prompt)
        
        # Try to extract JSON from response
        import json
        import re
        
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                return {
                    "explanation": result.get("explanation", "An error occurred during training."),
                    "root_cause": result.get("root_cause", "Unknown"),
                    "suggestions": result.get("suggestions", ["Try a different column", "Check your data"])
                }
            except:
                pass
        
        # Fallback: parse from text
        return {
            "explanation": response.split('\n')[0] if response else "An error occurred during training.",
            "root_cause": "See explanation above",
            "suggestions": ["Try a different target column", "Check for missing values", "Verify data types"]
        }
        
    except Exception as e:
        # If LLM analysis fails, return basic info
        return {
            "explanation": f"Training failed: {error_msg}",
            "root_cause": f"Error type: {type(error).__name__}",
            "suggestions": [
                f"Try a different target column (current: {target_column})",
                "Check if the target column has missing values",
                "Verify the data types are correct"
            ]
        }
