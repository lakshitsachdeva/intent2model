"""
Base contract for drift LLM adapters.
The CLI uses the backend as the planner/executor; adapters return
structured ML intent, plan updates, and explanations when used locally.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Structured response from an LLM adapter: intent, plan updates, explanation."""

    raw: str
    intent: Optional[Dict[str, Any]] = None  # target, drop_columns, performance_mode, start_training
    plan_summary: Optional[str] = None
    explanation: Optional[str] = None


class BaseLLMAdapter(ABC):
    """
    LLM adapter contract.
    The LLM is the PLANNER, not the executor; it updates plans and explains outcomes.
    Execution is assumed to happen automatically (backend).
    """

    @abstractmethod
    def generate(self, user_message: str, system_prompt: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Turn user message + optional context into a structured response.
        Returns intent, plan summary, and explanation — never claims to edit files or execute.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter identifier (e.g. gemini_cli, local_llm)."""
        pass
