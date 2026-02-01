"""
Pluggable LLM adapters for drift CLI.
Primary planning/execution uses the backend; these adapters support
optional local processing (e.g. help, summaries) or future offline use.
"""

from drift.llm_adapters.base import BaseLLMAdapter, LLMResponse

__all__ = ["BaseLLMAdapter", "LLMResponse"]
