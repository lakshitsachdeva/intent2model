"""
Local LLM adapter — ollama / llama.cpp style.
Used when drift CLI is configured to use a local model instead of the backend.
"""

import os
import re
from typing import Any, Dict, Optional

from drift.llm_adapters.base import BaseLLMAdapter, LLMResponse


class LocalLLMAdapter(BaseLLMAdapter):
    """
    Adapter for local LLM (e.g. Ollama, llama.cpp server).
    Expects OLLAMA_BASE_URL or LOCAL_LLM_URL; model from OLLAMA_MODEL or LOCAL_LLM_MODEL.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or os.environ.get("LOCAL_LLM_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or os.environ.get("LOCAL_LLM_MODEL", "llama2")

    @property
    def name(self) -> str:
        return "local_llm"

    def generate(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        prompt = user_message
        if system_prompt:
            prompt = f"{system_prompt}\n\n---\n\n{prompt}"
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        raw = self._call_api(prompt)
        intent = self._parse_intent(raw)
        return LLMResponse(raw=raw, intent=intent, explanation=raw)

    def _call_api(self, prompt: str) -> str:
        try:
            import requests
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            if r.status_code >= 400:
                return f"[Local LLM error: {r.status_code} {r.text[:200]}]"
            data = r.json()
            return (data.get("response") or "").strip()
        except Exception as e:
            return f"[Local LLM error: {e}]"

    def _parse_intent(self, raw: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"INTENT_JSON:\s*(\{.*?\})\s*$", raw, re.DOTALL)
        if not match:
            return None
        try:
            import json
            return json.loads(match.group(1).strip())
        except Exception:
            return None
