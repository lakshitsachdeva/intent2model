"""
Entry point for `python -m drift` or `drift` command.
Runs the chat-based REPL. If no backend is running, downloads and starts the engine.
"""

import os
import sys

from drift.cli.repl import run_repl


def main() -> None:
    if "--version" in sys.argv or "-v" in sys.argv:
        from importlib.metadata import version
        print(f"drift-ml {version('drift-ml')}")
        return
    base_url = os.environ.get("DRIFT_BACKEND_URL")
    if not base_url:
        from drift.engine_launcher import ensure_engine

        if not ensure_engine():
            print("drift: Failed to start engine. Check ~/.drift/bin/.engine-stderr.log or set DRIFT_BACKEND_URL", file=sys.stderr)
            sys.exit(1)
        base_url = f"http://127.0.0.1:{os.environ.get('DRIFT_ENGINE_PORT', '8000')}"
    run_repl(base_url=base_url)


if __name__ == "__main__":
    main()
