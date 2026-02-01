"""
Chat-based CLI loop for drift.
Natural language input; maintains session state; reuses backend planner + executor.
"""

import re
import sys
import threading
import time
from typing import Any, Dict, Optional

from drift.cli.client import BackendClient, BackendError, DEFAULT_BASE_URL
from drift.cli.session import SessionState


PROMPT = "drift › "
LOAD_PATTERN = re.compile(r"^\s*load\s+(.+)$", re.IGNORECASE)


def run_repl(base_url: Optional[str] = None) -> None:
    """Run the chat-based REPL. Uses backend at base_url (default from DRIFT_BACKEND_URL or localhost:8000)."""
    try:
        client = BackendClient(base_url=base_url or DEFAULT_BASE_URL)
    except BackendError as e:
        print(f"drift: {e.message}", file=sys.stderr)
        sys.exit(1)
    session = SessionState()
    _print_banner(client)
    while True:
        try:
            line = input(PROMPT).strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        try:
            _handle_input(line, client, session)
        except BackendError as e:
            print(f"Error: {e.message}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
            continue


def _print_banner(client: BackendClient) -> None:
    _print_welcome()
    engine_ok = False
    llm_name = ""
    try:
        health = client.health()
        engine_ok = (health.get("status") or "").lower() == "healthy"
        llm_name = health.get("llm_provider") or health.get("current_model") or ""
    except Exception:
        pass
    _print_status(engine_ok=engine_ok, llm_name=llm_name)
    _print_examples()
    print()


def _print_welcome() -> None:
    """drift by Lakshit Sachdeva. Local-first ML engineer."""
    print()
    print("  ----------------------------------------")
    print("  drift by Lakshit Sachdeva")
    print("  Local-first ML engineer.")
    print("  Same engine as the web app.")
    print("  No commands to memorize.")
    print("  ----------------------------------------")
    print()


def _print_status(engine_ok: bool, llm_name: str) -> None:
    """Engine running, LLM detected, Ready."""
    if engine_ok:
        print("  \u2713  Engine running")
        if llm_name:
            print(f"  \u2713  LLM detected ({llm_name})")
        else:
            print("  \u2713  LLM detected (Gemini CLI / Ollama / local)")
        print("  \u2713  Ready")
    else:
        print("  \u2717  Engine not running")
        print("      Start the engine or set DRIFT_BACKEND_URL to a running engine URL.")
    print()


def _print_examples() -> None:
    """Examples: load, predict, try something stronger, quit."""
    print("  Examples:")
    print("    load data.csv")
    print("    predict price")
    print("    try something stronger")
    print("    why is accuracy capped")
    print("    quit")
    print()


def _handle_input(line: str, client: BackendClient, session: SessionState) -> None:
    load_match = LOAD_PATTERN.match(line)
    if load_match:
        path = load_match.group(1).strip().strip('"\'')
        _do_load(path, client, session)
        return
    if not session.has_session():
        print("Load a dataset first: load path/to/file.csv")
        return
    _do_chat(line, client, session)


def _do_load(path: str, client: BackendClient, session: SessionState) -> None:
    out = client.upload_csv(path)
    session.dataset_path = path
    session.update_from_upload(
        session_id=out["session_id"],
        dataset_id=out["dataset_id"],
        initial_message=out.get("initial_message"),
    )
    print("Dataset loaded.")
    if out.get("initial_message"):
        content = out["initial_message"].get("content") or ""
        if content:
            _print_message(content)
    print()


def _do_chat(message: str, client: BackendClient, session: SessionState) -> None:
    sid = session.session_id
    out = client.chat(sid, message)
    chat_history = out.get("chat_history") or []
    session.update_from_chat(chat_history)
    # Show latest agent reply
    for m in reversed(chat_history):
        if m.get("role") == "agent" and m.get("content"):
            _print_message(m["content"])
            break
    trigger = out.get("trigger_training") is True
    if trigger:
        _run_training_and_show(client, session)
    print()


def _run_training_and_show(client: BackendClient, session: SessionState) -> None:
    sid = session.session_id
    if not sid:
        return
    train_result: Optional[Dict[str, Any]] = None
    train_error: Optional[Exception] = None

    def do_train() -> None:
        nonlocal train_result, train_error
        try:
            train_result = client.train(sid)
        except Exception as e:
            train_error = e

    thread = threading.Thread(target=do_train, daemon=True)
    thread.start()
    print("Training started…")
    run_id: Optional[str] = None
    last_event_count = 0
    poll_interval = 0.8
    timeout_sec = 300
    start = time.time()
    while thread.is_alive() and (time.time() - start) < timeout_sec:
        time.sleep(poll_interval)
        try:
            sess = client.get_session(sid)
            run_id = sess.get("current_run_id")
            if run_id:
                break
        except Exception:
            pass
    if run_id:
        while thread.is_alive() and (time.time() - start) < timeout_sec:
            time.sleep(poll_interval)
            try:
                run_state = client.get_run_state(run_id)
                events = run_state.get("events") or []
                for ev in events[last_event_count:]:
                    msg = ev.get("message") or ev.get("step_name") or ""
                    if msg:
                        print(f"  {msg}")
                last_event_count = len(events)
                status = run_state.get("status")
                if status in ("success", "failed", "refused"):
                    break
            except Exception:
                pass
    thread.join(timeout=1.0)
    if train_error:
        print(f"Training error: {train_error}", file=sys.stderr)
        return
    if train_result:
        session.update_after_train(
            run_id=train_result.get("run_id", ""),
            metrics=train_result.get("metrics") or {},
            agent_message=train_result.get("agent_message"),
        )
        agent_message = train_result.get("agent_message")
        if agent_message:
            _print_message(agent_message)
        metrics = train_result.get("metrics") or {}
        if metrics:
            primary = metrics.get("primary_metric_value") or metrics.get("accuracy") or metrics.get("r2")
            if primary is not None:
                print(f"  Primary metric: {primary}")


def _print_message(content: str) -> None:
    """Print agent message; strip markdown bold for terminal if desired, keep newlines."""
    text = content.strip()
    if not text:
        return
    # Optional: replace **x** with x for readability in terminal
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    for line in text.split("\n"):
        print(line)
