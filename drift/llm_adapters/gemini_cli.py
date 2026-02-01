"""
Gemini CLI adapter — uses gemini CLI (default if available) for local LLM calls.
Used when drift CLI runs in a mode that calls a local LLM instead of the backend.
"""

import os
import re
import subprocess
from typing import Any, Dict, Optional

from drift.llm_adapters.base import BaseLLMAdapter, LLMResponse


class GeminiCLIAdapter(BaseLLMAdapter):
    """Adapter that shells out to gemini CLI (e.g. `gemini` or GEMINI_CLI_CMD)."""

    def __init__(self, cmd: Optional[str] = None):
        self.cmd = (cmd or os.environ.get("GEMINI_CLI_CMD", "gemini")).strip()

    @property
    def name(self) -> str:
        return "gemini_cli"

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
        raw = self._call_cli(prompt)
        intent = self._parse_intent(raw)
        return LLMResponse(raw=raw, intent=intent, explanation=raw)

    def _call_cli(self, prompt: str) -> str:
        try:
            proc = subprocess.run(
                [self.cmd, "run", "-"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                return f"[gemini CLI error: {proc.stderr or proc.stdout or 'unknown'}]"
            return (proc.stdout or "").strip()
        except FileNotFoundError:
            return f"[gemini CLI not found: {self.cmd}]"
        except subprocess.TimeoutExpired:
            return "[gemini CLI timeout]"

    def _parse_intent(self, raw: str) -> Optional[Dict[str, Any]]:
        """Extract INTENT_JSON from response if present."""
        match = re.search(r"INTENT_JSON:\s*(\{.*?\})\s*$", raw, re.DOTALL)
        if not match:
            return None
        try:
            import json
            return json.loads(match.group(1).strip())
        except Exception:
            return None
