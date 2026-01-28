"""
Pydantic schemas for Intent2Model pipeline configuration & AutoML planning.

All LLM outputs must validate against these schemas.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


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


FeatureKind = Literal[
    "continuous",
    "ordinal",
    "binary_categorical",
    "nominal_categorical",
    "identifier",
    "leakage_risk",
    "unknown",
]

ImputeStrategy = Literal["none", "mean", "median", "most_frequent", "constant"]
ScaleStrategy = Literal["none", "standard", "robust"]
EncodeStrategy = Literal["none", "one_hot", "ordinal", "frequency"]


class FeatureTransform(BaseModel):
    """Per-feature transformation decision."""

    name: str
    inferred_dtype: str = Field(..., description="String dtype representation from pandas")
    kind: FeatureKind
    drop: bool = Field(False, description="Whether to drop this feature")

    impute: ImputeStrategy = "none"
    encode: EncodeStrategy = "none"
    scale: ScaleStrategy = "none"

    notes_md: str = Field("", description="Markdown justification for this feature decision")

    class Config:
        extra = "forbid"


class ModelCandidate(BaseModel):
    """Candidate model with justification."""

    model_name: str = Field(..., description="Internal model key (e.g., logistic_regression, random_forest, xgboost)")
    reason_md: str = Field(..., description="Markdown justification")
    params: Dict[str, Any] = Field(default_factory=dict, description="Optional hyperparameters")

    class Config:
        extra = "forbid"


class AutoMLPlan(BaseModel):
    """
    Dataset-agnostic AutoML engineer plan.
    LLM must produce this BEFORE any model code is executed.
    """

    inferred_target: str = Field(..., description="Target column inferred from dataset (or validated)")
    task_type: Literal["regression", "binary_classification", "multiclass_classification"]
    task_inference_md: str

    dataset_intelligence_md: str
    transformation_strategy_md: str
    model_selection_md: str
    training_validation_md: str
    error_behavior_analysis_md: str
    explainability_md: str

    primary_metric: str = Field(..., description="Main metric to optimize")
    additional_metrics: List[str] = Field(default_factory=list)

    feature_transforms: List[FeatureTransform] = Field(default_factory=list)
    model_candidates: List[ModelCandidate] = Field(default_factory=list)

    planning_source: Literal["llm", "fallback"] = Field(
        default="fallback",
        description="Whether this plan was produced by the LLM or the rule-based fallback",
    )
    planning_error: Optional[str] = Field(
        default=None,
        description="If fallback was used, a short reason (e.g., rate limit / invalid JSON)",
    )

    class Config:
        extra = "forbid"


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
