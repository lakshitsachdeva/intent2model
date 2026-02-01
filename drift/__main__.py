"""
Entry point for `python -m drift` or `drift` command.
Runs the chat-based REPL.
"""

from drift.cli.repl import run_repl


def main() -> None:
    run_repl()


if __name__ == "__main__":
    main()
