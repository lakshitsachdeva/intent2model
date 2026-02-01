"""
HTTP client for the existing backend API.
Uses the same endpoints as the web app: upload, chat, session, runs, train.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

DEFAULT_BASE_URL = os.environ.get("DRIFT_BACKEND_URL", "http://localhost:8000")


class BackendError(Exception):
    """Raised when the backend returns an error or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        self.message = message
        self.status_code = status_code
        self.body = body
        super().__init__(message)


class BackendClient:
    """Client for backend upload, chat, session, runs, and train endpoints."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if requests is None:
            raise BackendError(
                "The 'requests' library is required for the drift CLI. Install it with: pip install requests"
            )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> Dict[str, Any]:
        """GET /health — check backend is up and LLM status."""
        r = requests.get(self._url("/health"), timeout=10)
        r.raise_for_status()
        return r.json()

    def upload_csv(self, file_path: str) -> Dict[str, Any]:
        """
        POST /upload — upload a local CSV and create a session.
        Returns session_id, dataset_id, profile, initial_message.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise BackendError(f"File not found: {path}")
        with open(path, "rb") as f:
            name = path.name
            files = {"file": (name, f, "text/csv")}
            r = requests.post(self._url("/upload"), files=files, timeout=60)
        if r.status_code >= 400:
            try:
                body = r.json()
                detail = body.get("detail", str(r.text))
            except Exception:
                detail = r.text or str(r.status_code)
            raise BackendError(f"Upload failed: {detail}", status_code=r.status_code, body=r.text)
        return r.json()

    def chat(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        POST /session/{session_id}/chat — send a user message, get agent reply.
        Returns chat_history, trigger_training.
        """
        r = requests.post(
            self._url(f"/session/{session_id}/chat"),
            json={"message": message},
            timeout=120,
        )
        if r.status_code >= 400:
            try:
                body = r.json()
                detail = body.get("detail", str(r.text))
            except Exception:
                detail = r.text or str(r.status_code)
            raise BackendError(f"Chat failed: {detail}", status_code=r.status_code, body=r.text)
        return r.json()

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """GET /session/{session_id} — full session state (chat, model_state, current_run_id, etc.)."""
        r = requests.get(self._url(f"/session/{session_id}"), timeout=30)
        if r.status_code >= 400:
            try:
                body = r.json()
                detail = body.get("detail", str(r.text))
            except Exception:
                detail = r.text or str(r.status_code)
            raise BackendError(f"Get session failed: {detail}", status_code=r.status_code, body=r.text)
        return r.json()

    def get_run_state(self, run_id: str) -> Dict[str, Any]:
        """GET /runs/{run_id} — run state: status, current_step, progress, events[]."""
        r = requests.get(self._url(f"/runs/{run_id}"), timeout=30)
        if r.status_code >= 400:
            try:
                body = r.json()
                detail = body.get("detail", str(r.text))
            except Exception:
                detail = r.text or str(r.status_code)
            raise BackendError(f"Get run failed: {detail}", status_code=r.status_code, body=r.text)
        return r.json()

    def train(self, session_id: str) -> Dict[str, Any]:
        """
        POST /session/{session_id}/train — run one training attempt (blocking).
        Returns run_id, session_id, metrics, refused, refusal_reason, agent_message.
        """
        r = requests.post(self._url(f"/session/{session_id}/train"), timeout=self.timeout)
        if r.status_code >= 400:
            try:
                body = r.json()
                detail = body.get("detail", str(r.text))
            except Exception:
                detail = r.text or str(r.status_code)
            raise BackendError(f"Train failed: {detail}", status_code=r.status_code, body=r.text)
        return r.json()

    def download_notebook(self, run_id: str) -> Optional[bytes]:
        """GET /download/{run_id}/notebook — download notebook bytes, or None if failed."""
        r = requests.get(self._url(f"/download/{run_id}/notebook"), timeout=60)
        if r.status_code != 200:
            return None
        return r.content

    def download_model(self, run_id: str) -> Optional[bytes]:
        """GET /download/{run_id}/model — download model pickle bytes, or None if refused/failed."""
        r = requests.get(self._url(f"/download/{run_id}/model"), timeout=60)
        if r.status_code != 200:
            return None
        return r.content
