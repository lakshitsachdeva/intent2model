"""
Run state and agent event schemas for execution visibility.

Frontend must NEVER infer state — it gets status, attempt_count, and full event timeline from GET /runs/{run_id}.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# Event step names the agent emits (step_name) — string values from _log_run_event(stage=...)
# Examples: planning_started, planning_completed, training_started, training_failed,
# metric_gate_failed, llm_diagnosis_started, llm_diagnosis_completed, plan_revision_created,
# retry_started, run_halted, run_succeeded, repair, diagnose, error, success, init, plan, train, etc.


class AgentEvent(BaseModel):
    """Single event in the agent execution backlog."""
    run_id: str
    timestamp: str  # ISO format
    step_name: str  # stage string (e.g. plan, train, diagnose, repair, retry, error)
    status: Literal["started", "completed", "failed", "info", "warning"] = "info"
    message: str = ""
    payload: Optional[Dict[str, Any]] = None  # metrics, reasons, markdown, failed_gates, etc.

    class Config:
        extra = "allow"


class RunState(BaseModel):
    """Current run state + full event backlog. Returned by GET /runs/{run_id}."""
    run_id: str
    status: Literal[
        "planning",
        "training",
        "diagnosing",
        "retrying",
        "failed",
        "success",
        "refused",
        "init",
    ] = "init"
    current_step: str = ""
    attempt_count: int = 0
    progress: float = 0.0
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Full ordered event timeline (backlog)")

    class Config:
        extra = "allow"
