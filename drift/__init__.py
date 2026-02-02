"""
drift — terminal-first, chat-based AutoML. Use as CLI or library.

CLI:
    pip install drift-ml
    drift
    drift › load data.csv
    drift › predict price

Library:
    from drift import Drift

    d = Drift()
    d.load("data.csv")
    d.chat("predict price")
    result = d.train()
"""

__version__ = "0.2.12"

from drift.api import Drift
from drift.cli.client import BackendClient, BackendError
from drift.engine_launcher import ensure_engine

__all__ = ["Drift", "BackendClient", "BackendError", "ensure_engine", "__version__"]
