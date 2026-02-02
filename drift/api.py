"""
Programmatic API for drift — use in your own Python scripts.

Example:
    from drift import Drift

    d = Drift()
    d.load("data.csv")
    d.chat("predict price")
    result = d.train()
    print(result["metrics"])
"""

import os
from typing import Any, Dict, Optional

from drift.cli.client import BackendClient, BackendError


class Drift:
    """
    High-level client for drift. Load data, chat, train — in your code.

    Uses a local engine (auto-started if needed) or DRIFT_BACKEND_URL.
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Args:
            base_url: Engine URL. If None, uses DRIFT_BACKEND_URL or auto-starts engine.
        """
        if base_url:
            self._client = BackendClient(base_url=base_url)
        else:
            url = os.environ.get("DRIFT_BACKEND_URL")
            if not url:
                from drift.engine_launcher import ensure_engine

                if not ensure_engine():
                    raise BackendError(
                        "Could not start engine. Set DRIFT_BACKEND_URL or check ~/.drift/bin/.engine-stderr.log"
                    )
                port = os.environ.get("DRIFT_ENGINE_PORT", "8000")
                url = f"http://127.0.0.1:{port}"
            self._client = BackendClient(base_url=url)
        self._session_id: Optional[str] = None
        self._dataset_id: Optional[str] = None

    def load(self, path: str) -> "Drift":
        """
        Load a CSV file. Returns self for chaining.

        Args:
            path: Path to CSV file.

        Returns:
            self
        """
        out = self._client.upload_csv(path)
        self._session_id = out["session_id"]
        self._dataset_id = out["dataset_id"]
        return self

    def chat(self, message: str) -> Dict[str, Any]:
        """
        Send a chat message. Returns chat_history and trigger_training.

        Args:
            message: Natural language, e.g. "predict price", "try something stronger".

        Returns:
            {"chat_history": [...], "trigger_training": bool}
        """
        if not self._session_id:
            raise BackendError("Load a dataset first: drift.load('data.csv')")
        return self._client.chat(self._session_id, message)

    def train(self) -> Dict[str, Any]:
        """
        Run one training attempt. Blocking.

        Returns:
            run_id, metrics, refused, agent_message, etc.
        """
        if not self._session_id:
            raise BackendError("Load a dataset first: drift.load('data.csv')")
        return self._client.train(self._session_id)

    def get_last_reply(self, chat_result: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get the last agent reply from chat_history."""
        hist = (chat_result or {}).get("chat_history") if chat_result else None
        if not hist and self._session_id:
            sess = self._client.get_session(self._session_id)
            hist = sess.get("chat_history") or []
        if not hist:
            return None
        for m in reversed(hist):
            if m.get("role") == "agent" and m.get("content"):
                return m["content"]
        return None

    def download_notebook(self, run_id: str) -> Optional[bytes]:
        """Download training notebook as bytes."""
        return self._client.download_notebook(run_id)

    def download_model(self, run_id: str) -> Optional[bytes]:
        """Download trained model pickle as bytes."""
        return self._client.download_model(run_id)

    def health(self) -> Dict[str, Any]:
        """Check engine health and LLM status."""
        return self._client.health()

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id
