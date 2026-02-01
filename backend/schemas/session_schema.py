"""
Persistent Session schema for chat-first, real-time ML.

Nothing happens outside a session.
Cursor-for-ML: State → Propose → Apply → Observe → Discuss → Iterate.
Session holds: dataset ref, ModelState (source of truth), chat, live notebook, user_constraints.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


PerformanceMode = Literal["conservative", "balanced", "aggressive"]


class ModelState(BaseModel):
    """
    Live ML state — single source of truth for UI.
    Every training step updates this. Chat and panels read from it.
    """
    dataset_summary: Dict[str, Any] = Field(default_factory=dict)  # n_rows, n_cols, numeric_cols, categorical_cols, etc.
    current_features: List[str] = Field(default_factory=list)  # feature names after drops
    preprocessing_steps: List[str] = Field(default_factory=list)  # e.g. ["StandardScaler", "OneHotEncoder"]
    current_model: Optional[str] = None  # e.g. "random_forest"
    previous_model: Optional[str] = None  # for diff display
    metrics: Dict[str, Any] = Field(default_factory=dict)  # accuracy, f1, rmse, etc.
    error_analysis: Dict[str, Any] = Field(default_factory=dict)  # confusion_matrix, feature_importance, hardest_classes, residuals
    attempt_number: int = 0
    last_diff: Dict[str, Any] = Field(default_factory=dict)  # model: A→B, preprocessing: +X, dropped_features: [...]
    status: Literal["idle", "proposing", "training", "evaluating", "success", "refused", "error"] = "idle"
    status_message: str = ""

    class Config:
        extra = "allow"


class ChatMessage(BaseModel):
    """Single message in session chat (user or agent)."""
    role: Literal["user", "agent"]
    content: str = ""
    ts: Optional[str] = None  # ISO timestamp
    payload: Optional[Dict[str, Any]] = None  # e.g. data_overview, metrics, choices


class SessionState(BaseModel):
    """
    In-memory session state (dataset stored separately by dataset_id).
    Cursor-for-ML: model_state is source of truth; chat_history + live_notebook_cells drive the conversation.
    """
    session_id: str
    dataset_id: str
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)  # [{role, content, ts?, payload?}]
    model_state: Optional[Dict[str, Any]] = None  # ModelState as dict for JSON; single source of truth for UI
    structural_plan: Optional[Dict[str, Any]] = None
    execution_plans: List[Dict[str, Any]] = Field(default_factory=list)
    failure_history: List[Dict[str, Any]] = Field(default_factory=list)
    notebook_cells: List[Dict[str, Any]] = Field(default_factory=list)  # append-only live code buffer
    performance_mode: PerformanceMode = Field(default="conservative")
    current_run_id: Optional[str] = None
    refused: bool = False
    refusal_reason: str = ""
    user_constraints: Dict[str, Any] = Field(default_factory=dict)  # drop_columns, target, force_model, etc.
    cancel_requested: bool = False  # user can set to true to stop next iteration

    class Config:
        extra = "allow"
