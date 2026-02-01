"""
Session state for the drift CLI.
Holds dataset ref, plan summary, last metrics, run_id, and chat history
so the REPL can maintain context and the backend can be queried correctly.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    """In-memory CLI session: dataset, plan, run, metrics, chat."""

    dataset_path: Optional[str] = None
    dataset_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    plan_summary: Optional[str] = None
    last_metrics: Dict[str, Any] = field(default_factory=dict)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)

    def has_session(self) -> bool:
        return bool(self.session_id)

    def has_dataset(self) -> bool:
        return bool(self.dataset_id and self.session_id)

    def clear_run(self) -> None:
        self.run_id = None
        self.last_metrics = {}

    def update_from_upload(self, session_id: str, dataset_id: str, initial_message: Optional[Dict] = None) -> None:
        self.session_id = session_id
        self.dataset_id = dataset_id
        self.run_id = None
        self.last_metrics = {}
        if initial_message:
            self.chat_history = [initial_message]

    def update_from_chat(self, chat_history: List[Dict[str, Any]], run_id: Optional[str] = None) -> None:
        self.chat_history = chat_history or self.chat_history
        if run_id is not None:
            self.run_id = run_id

    def update_from_session(self, session: Dict[str, Any]) -> None:
        self.session_id = session.get("session_id") or self.session_id
        self.dataset_id = session.get("dataset_id") or self.dataset_id
        self.run_id = session.get("current_run_id") or self.run_id
        self.chat_history = session.get("chat_history") or self.chat_history
        ms = session.get("model_state") or {}
        self.last_metrics = ms.get("metrics") or self.last_metrics
        self.plan_summary = None
        if session.get("structural_plan"):
            self.plan_summary = _summarize_plan(session["structural_plan"])

    def update_after_train(self, run_id: str, metrics: Dict[str, Any], agent_message: Optional[str] = None) -> None:
        self.run_id = run_id
        self.last_metrics = metrics or self.last_metrics
        if agent_message and self.chat_history:
            self.chat_history[-1]["content"] = agent_message


def _summarize_plan(structural_plan: Dict[str, Any]) -> str:
    """One-line summary of structural plan for display (StructuralPlan has inferred_target, task_type; no models)."""
    if not structural_plan:
        return ""
    target = structural_plan.get("inferred_target") or structural_plan.get("target") or "?"
    task = structural_plan.get("task_type") or "?"
    return f"Target: {target}, task: {task}"
