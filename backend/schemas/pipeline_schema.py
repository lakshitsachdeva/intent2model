"""
Pydantic schemas for Intent2Model pipeline configuration.

All LLM outputs must validate against these schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class PipelineConfig(BaseModel):
    """Strict schema for pipeline configuration."""
    
    task: Literal["classification", "regression"] = Field(
        ...,
        description="Type of machine learning task"
    )
    
    target: str = Field(
        ...,
        description="Name of the target column to predict"
    )
    
    metric: str = Field(
        ...,
        description="Metric to optimize. Classification: accuracy, precision, recall, f1, roc_auc. Regression: rmse, r2, mae"
    )
    
    preprocessing: List[str] = Field(
        default_factory=lambda: ["standard_scaler", "one_hot"],
        description="List of preprocessing steps: standard_scaler, one_hot, imputer"
    )
    
    model_candidates: List[str] = Field(
        default_factory=lambda: ["random_forest"],
        description="List of model names to try: logistic_regression, random_forest, xgboost (classification) or linear_regression, random_forest, xgboost (regression)"
    )
    
    cv_strategy: str = Field(
        default="stratified_kfold",
        description="Cross-validation strategy: stratified_kfold (classification) or kfold (regression)"
    )
    
    class Config:
        extra = "forbid"  # Reject any extra fields


class QuestionResponse(BaseModel):
    """Schema for user responses to clarification questions."""
    
    question_id: str = Field(..., description="ID of the question being answered")
    answer: str = Field(..., description="User's answer to the question")


class UserIntent(BaseModel):
    """Schema for capturing user intent and answers."""
    
    target_column: Optional[str] = Field(None, description="Target column name if known")
    task_type: Optional[Literal["classification", "regression"]] = Field(None, description="Task type if known")
    priority_metric: Optional[str] = Field(None, description="Priority metric if known")
    business_context: Optional[str] = Field(None, description="Business context or use case")
    answers: List[QuestionResponse] = Field(default_factory=list, description="Answers to clarification questions")
