"""
Failure Report Schema for Self-Healing ML Engineer.

When training fails, we construct a structured FailureReport
and send it to the LLM for diagnosis and recovery.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


class FailureReport(BaseModel):
    """
    Structured failure report sent to LLM for diagnosis.
    This is the agent's debug log.
    """
    
    failure_stage: Literal["compiler", "training", "evaluation"] = Field(
        ...,
        description="Stage at which failure occurred"
    )
    
    failed_gates: List[str] = Field(
        default_factory=list,
        description="List of error gates that failed (e.g., 'RMSE > 0.5 * target_std')"
    )
    
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="All computed metrics: RMSE, MAE, normalized_RMSE, normalized_MAE, R², accuracy, etc."
    )
    
    target_stats: Dict[str, float] = Field(
        default_factory=dict,
        description="Target statistics: mean, std, IQR, MAD, skew, min, max"
    )
    
    residual_diagnostics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Residual analysis: is_heteroscedastic, variance_ratio, description (for regression)"
    )
    
    feature_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Feature preprocessing summary: feature_count_before, feature_count_after, dropped_features, encoded_features"
    )
    
    model_used: str = Field(
        ...,
        description="Model that was trained (e.g., 'random_forest')"
    )
    
    model_hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters used for the model"
    )
    
    previous_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Snapshot of the AutoMLPlan that led to this failure (legacy)"
    )
    
    last_execution_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Snapshot of the ExecutionPlan that led to this failure (agentic)"
    )
    
    error_message: str = Field(
        ...,
        description="Original error message or failure description"
    )
    
    error_traceback: Optional[str] = Field(
        default=None,
        description="Full traceback if available"
    )
    
    attempt_number: int = Field(
        ...,
        description="Which attempt this failure occurred on (1-indexed)"
    )


class DiagnosisResponse(BaseModel):
    """
    LLM's diagnosis and recovery plan.
    """
    
    diagnosis_md: str = Field(
        ...,
        description="Markdown explanation of why the model failed and what assumptions were wrong"
    )
    
    plan_changes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy: structured changes to apply. Prefer repair_plan for agentic flow."
    )
    
    repair_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured diff (RepairPlan) to apply to last ExecutionPlan for next attempt"
    )
    
    recovery_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence that the proposed changes will fix the issue (0-1)"
    )
    
    is_task_learnable: bool = Field(
        True,
        description="Whether this task is learnable with the available data"
    )
    
    suggested_stop: bool = Field(
        False,
        description="Whether the agent should stop trying (e.g., task is not learnable)"
    )
