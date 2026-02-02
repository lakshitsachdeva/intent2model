"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

# Add backend to path
backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend))
