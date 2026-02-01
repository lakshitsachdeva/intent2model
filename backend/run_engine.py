"""
Entry point for the drift engine binary (PyInstaller).
Runs the same HTTP API as the web app. No user-facing code; closed distribution.
"""

import os
import sys

# Ensure backend is on path when frozen (PyInstaller) or when run as script
if getattr(sys, "frozen", False):
    _base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
else:
    _base = os.path.dirname(os.path.abspath(__file__))
if _base not in sys.path:
    sys.path.insert(0, _base)

if __name__ == "__main__":
    import uvicorn
    from main import app

    port = int(os.environ.get("DRIFT_ENGINE_PORT", "8000"))
    host = os.environ.get("DRIFT_ENGINE_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
