"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client. Import app after path setup."""
    import sys
    from pathlib import Path
    backend = Path(__file__).resolve().parent.parent / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from main import app
    return TestClient(app)


def test_health(client):
    """Health endpoint returns 200."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data or "llm_available" in data


def test_root(client):
    """Root returns message."""
    r = client.get("/")
    assert r.status_code == 200
