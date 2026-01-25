"""
Question agent for Intent2Model.

Determines what information is missing and asks minimal clarification questions.
"""

from typing import Dict, List, Optional
from schemas.pipeline_schema import UserIntent


class Question:
    """Represents a clarification question."""
    
    def __init__(self, question_id: str, question_text: str, question_type: str):
        self.question_id = question_id
        self.question_text = question_text
        self.question_type = question_type
    
    def to_dict(self):
        return {
            "question_id": self.question_id,
            "question": self.question_text,
            "type": self.question_type
        }


def generate_questions(
    profile: Dict,
    known_info: Optional[UserIntent] = None
) -> List[Question]:
    """
    Generate minimal clarification questions based on dataset profile and known info.
    
    Args:
        profile: Dataset profile from profiler
        known_info: Optional UserIntent with already known information
        
    Returns:
        List of Question objects
    """
    questions = []
    
    # Extract known information
    known_target = known_info.target_column if known_info else None
    known_task = known_info.task_type if known_info else None
    known_metric = known_info.priority_metric if known_info else None
    
    # Question 1: Target column (if not known)
    if not known_target:
        candidate_targets = profile.get("candidate_targets", [])
        if candidate_targets:
            question_text = (
                f"Which column should be predicted? "
                f"Candidate columns detected: {', '.join(candidate_targets[:5])}. "
                f"Or specify another column name."
            )
        else:
            question_text = (
                f"Which column should be predicted? "
                f"Available columns: {', '.join(profile.get('numeric_cols', []) + profile.get('categorical_cols', []))[:10]}"
            )
        questions.append(Question("target_column", question_text, "target_selection"))
    
    # Question 2: Task type (if not known and target is known)
    if not known_task and known_target:
        # Infer task type from target column
        target_col = known_target
        if target_col in profile.get("numeric_cols", []):
            # Could be regression, but ask to confirm
            question_text = (
                f"The target column '{target_col}' is numeric. "
                f"Is this a regression task (predicting a continuous value) "
                f"or classification (predicting categories/buckets)?"
            )
        elif target_col in profile.get("categorical_cols", []):
            unique_count = profile.get("unique_counts", {}).get(target_col, 0)
            if unique_count > 20:
                question_text = (
                    f"The target column '{target_col}' has {unique_count} unique values. "
                    f"Is this classification (predicting categories) or should we bin it for regression?"
                )
            else:
                question_text = (
                    f"The target column '{target_col}' appears to be categorical. "
                    f"Confirm this is a classification task?"
                )
        else:
            question_text = "What type of prediction is this? Classification (categories) or Regression (continuous values)?"
        
        questions.append(Question("task_type", question_text, "task_selection"))
    elif not known_task:
        question_text = "What type of prediction is this? Classification (categories) or Regression (continuous values)?"
        questions.append(Question("task_type", question_text, "task_selection"))
    
    # Question 3: Metric priority (if task is known)
    if known_task and not known_metric:
        if known_task == "classification":
            question_text = (
                "Which error is worse for your use case?\n"
                "- False positives (predicting positive when it's negative)\n"
                "- False negatives (predicting negative when it's positive)\n"
                "- Neither (balanced accuracy is fine)\n"
                "This helps us choose the right metric (precision, recall, or accuracy)."
            )
        else:
            question_text = (
                "What matters most for your regression task?\n"
                "- Overall fit (R² score)\n"
                "- Avoiding large errors (RMSE)\n"
                "- Average error magnitude (MAE)"
            )
        questions.append(Question("priority_metric", question_text, "metric_selection"))
    
    # Question 4: Business context (optional, but helpful)
    if len(questions) <= 2:  # Only ask if we don't have many other questions
        question_text = (
            "What is the business context or use case? "
            "(e.g., 'predict customer churn', 'forecast sales', 'detect fraud') "
            "This helps us make better recommendations."
        )
        questions.append(Question("business_context", question_text, "context"))
    
    # Question 5: Explainability (optional)
    if known_task and len(questions) < 3:
        question_text = (
            "Do you need explainable models? "
            "(Some models are more interpretable but may have lower accuracy)"
        )
        questions.append(Question("explainability", question_text, "preference"))
    
    return questions


def determine_missing_info(profile: Dict, known_info: Optional[UserIntent] = None) -> List[str]:
    """
    Determine what information is still missing.
    
    Returns list of missing field names.
    """
    missing = []
    
    if not known_info or not known_info.target_column:
        missing.append("target_column")
    
    if not known_info or not known_info.task_type:
        missing.append("task_type")
    
    if not known_info or not known_info.priority_metric:
        missing.append("priority_metric")
    
    return missing
