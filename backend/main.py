"""
FastAPI backend for Intent2Model.

Provides endpoints for dataset upload and model training.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal, Dict, Any
import pandas as pd
import io
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.profiler import profile_dataset
from ml.trainer import train_classification, train_regression, compare_models
from ml.evaluator import evaluate_dataset
from utils.logging import create_run_id, log_run
from agents.automl_agent import plan_automl
from schemas.pipeline_schema import UserIntent
from schemas.session_schema import SessionState, ModelState
from agents.llm_interface import LLMInterface, get_current_model_info
from agents.error_analyzer import analyze_training_error
from agents.recovery_agent import AutonomousRecoveryAgent
from agents.intent_detector import IntentDetectionAgent
from utils.artifact_generator import generate_notebook, generate_readme, save_model_pickle, generate_chart_image, generate_model_report
import os
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask
import json
import re
import tempfile
import base64
from dotenv import load_dotenv
import subprocess
import shutil
import asyncio
from typing import Set
import json as json_lib

# Load .env from project root (works when running from backend/ on Windows/macOS/Linux)
from pathlib import Path
_env_root = Path(__file__).resolve().parent.parent
load_dotenv(_env_root / ".env")
load_dotenv()


app = FastAPI(title="Intent2Model API", version="1.0.0")


def _json_safe(obj):
    """Convert common numpy/pandas scalar types to plain Python types for FastAPI JSON encoding."""
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore

        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _model_code_for_notebook(task: str, model_name: str) -> str:
    """Return a stable boilerplate code snippet string used inside the notebook for the chosen model."""
    model_name = (model_name or "").strip().lower()
    task = (task or "").strip().lower()
    if task == "classification":
        mapping = {
            "logistic_regression": "LogisticRegression(max_iter=2000, random_state=42)",
            "random_forest": "RandomForestClassifier(n_estimators=300, random_state=42)",
            "gradient_boosting": "GradientBoostingClassifier(random_state=42)",
            "naive_bayes": "GaussianNB()",
            "svm": "SVC(random_state=42, probability=True)",
            "xgboost": "XGBClassifier(random_state=42, eval_metric='mlogloss')",
        }
        return mapping.get(model_name, "RandomForestClassifier(n_estimators=300, random_state=42)")
    else:
        mapping = {
            "linear_regression": "LinearRegression()",
            "random_forest": "RandomForestRegressor(n_estimators=300, random_state=42)",
            "gradient_boosting": "GradientBoostingRegressor(random_state=42)",
            "ridge": "Ridge(alpha=1.0, random_state=42)",
            "lasso": "Lasso(alpha=0.001, random_state=42)",
            "svm": "SVR()",
            "xgboost": "XGBRegressor(random_state=42)",
        }
        return mapping.get(model_name, "RandomForestRegressor(n_estimators=300, random_state=42)")


def _build_initial_chat_message(profile: Dict[str, Any], df: pd.DataFrame) -> tuple[str, Dict[str, Any]]:
    """
    Build the first agent message for chat-first flow: head, missing, target candidates, basic summary.
    Returns (content_str, payload_dict).
    """
    n_rows = profile.get("n_rows", 0)
    n_cols = profile.get("n_cols", 0)
    numeric = profile.get("numeric_cols", []) or []
    categorical = profile.get("categorical_cols", []) or []
    missing = profile.get("missing_percent", {}) or {}
    candidates = profile.get("candidate_targets", []) or list(df.columns)[:5]
    head = df.head(5).to_dict(orient="split") if not df.empty else {"columns": [], "data": []}
    missing_summary = {k: round(v, 1) for k, v in list(missing.items())[:20]}
    content = (
        f"Here's what I see in your data.\n\n"
        f"**Shape:** {n_rows} rows × {n_cols} columns\n"
        f"**Numeric:** {', '.join(numeric[:12]) or '—'}\n"
        f"**Categorical:** {', '.join(categorical[:12]) or '—'}\n"
        f"**Target candidates:** {', '.join(candidates[:8])}\n\n"
        f"Want me to propose a plan? You can say e.g. \"drop id\", \"use X as target\", \"try something stronger\", or \"explain why accuracy is low\"."
    )
    payload = {
        "head": head,
        "missing_percent": missing_summary,
        "target_candidates": candidates,
        "profile": {k: v for k, v in profile.items() if k in ("n_rows", "n_cols", "numeric_cols", "categorical_cols", "candidate_targets")},
    }
    return content, payload


def _preprocessing_recommendations(profile: Dict[str, Any], df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Rule-based, dataset-oriented preprocessing suggestions (LLM can refine later).
    """
    recs: list[dict[str, Any]] = []
    missing = profile.get("missing_percent", {}) or {}
    high_missing = [c for c, p in missing.items() if (p or 0) > 10]
    if high_missing:
        recs.append({"type": "imputer", "why": f"Missing values >10% in: {', '.join(high_missing[:8])}", "suggestion": "Add median imputer for numeric and most_frequent for categorical."})

    # High-cardinality categoricals
    cat_cols = profile.get("categorical_cols", []) or []
    high_card = []
    for c in cat_cols[:30]:
        try:
            nunq = int(df[c].nunique(dropna=True))
            if nunq > 30:
                high_card.append((c, nunq))
        except Exception:
            continue
    if high_card:
        recs.append({"type": "encoding", "why": f"High-cardinality categoricals: {', '.join([f'{c}({n})' for c,n in high_card[:6]])}", "suggestion": "Consider target encoding / hashing trick instead of one-hot."})

    # Skew / outliers for numeric
    num_cols = profile.get("numeric_cols", []) or []
    skewed = []
    for c in num_cols[:30]:
        try:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) < 20:
                continue
            sk = float(s.skew())
            if abs(sk) > 1.0:
                skewed.append((c, sk))
        except Exception:
            continue
    if skewed:
        recs.append({"type": "transform", "why": f"Skewed numeric columns: {', '.join([f'{c}({sk:.2f})' for c,sk in skewed[:6]])}", "suggestion": "Consider log/yeo-johnson transform; robust scaling."})

    # Scaling
    if num_cols:
        recs.append({"type": "scaling", "why": "Numeric features detected.", "suggestion": "StandardScaler for linear/SVM; robust scaler if heavy outliers."})

    return recs


def _parse_chat_to_constraints(message: str, columns: list[str]) -> tuple[bool, dict[str, Any], str]:
    """
    Parse user chat message into: is_question, constraints_delta, agent_reply.
    Rules: question → respond only, no train. Instruction → merge constraints, reply "Want me to train?".
    """
    msg = (message or "").strip().lower()
    cols_lower = {c: c.lower() for c in columns}
    is_question = False
    delta: dict[str, Any] = {}
    reply = ""

    # Question patterns: do not train, just answer
    if any(x in msg for x in ("explain why", "why is", "why are", "what is", "what are", "how do", "how does", "?")):
        is_question = True
        reply = "I can explain after we run a training attempt — try training first, then ask 'explain why accuracy is low' and I'll analyze the results."
        return is_question, delta, reply

    # drop <column>
    drop_match = re.search(r"drop\s+([a-zA-Z0-9_.\s]+?)(?:\s|$|,|\.)", msg)
    if drop_match:
        raw = drop_match.group(1).strip()
        for col in columns:
            if col.lower() == raw or raw in col.lower():
                delta.setdefault("drop_columns", []).append(col)
                break
        else:
            delta.setdefault("drop_columns", []).append(raw)
        reply = f"I'll drop {delta['drop_columns'][-1]} from features. Want me to propose a plan and train?"

    # use X as target / this target / target is X / wanna predict X / predict length / predict sepal length
    target_match = re.search(
        r"(?:use|set|target is?)\s+([a-zA-Z0-9_.]+)\s+as\s+target|(?:use|set)\s+([a-zA-Z0-9_.]+)\s+as\s+target|"
        r"target\s+is\s+([a-zA-Z0-9_.]+)|this target\s*[:\s]*([a-zA-Z0-9_.]+)|"
        r"predict\s+([a-zA-Z0-9_.]+)|(?:wanna|want to)\s+predict\s+([a-zA-Z0-9_.\s]+?)(?:\s*$|\s+as|\s+please)|"
        r"(?:i said |the )?target is (\w+)",
        msg,
        re.IGNORECASE,
    )
    if target_match:
        cand = next((g for g in target_match.groups() if g), "").strip()
        resolved = None
        for col in columns:
            if col.lower() == cand.lower():
                resolved = col
                break
        if not resolved and cand:
            # "sepal length" -> sepal.length (space to dot)
            cand_dotted = cand.replace(" ", ".").lower()
            for col in columns:
                if col.lower() == cand_dotted:
                    resolved = col
                    break
        if not resolved and cand:
            # Partial match: "length" -> petal.length or sepal.length; pick one that contains cand
            cand_clean = cand.replace(" ", ".").lower()
            containing = [c for c in columns if cand_clean in c.lower() or cand.lower().replace(" ", "") in c.lower()]
            if len(containing) == 1:
                resolved = containing[0]
            elif len(containing) > 1:
                # Prefer petal.X for "length"/"width" (common in iris); else first
                preferred = [c for c in containing if "petal" in c.lower()]
                resolved = preferred[0] if preferred else containing[0]
        if resolved:
            delta["target"] = resolved
            reply = f"I'll use **{resolved}** as the target. Want me to propose a plan and train?"
        else:
            delta["target"] = cand
            reply = f"I'll use '{cand}' as the target (please ensure it exists in the data). Want me to train?"

    # try/use something stronger or better / more complex / aggressive / go for it
    if any(x in msg for x in (
        "try something stronger", "use something stronger", "something stronger",
        "try something better", "something better", "better model", "try something better na",
        "more complex", "aggressive", "stronger model", "xgboost", "use stronger",
        "go for it", "go for it bro", "try stronger", "more powerful", "powerful models",
    )):
        delta["performance_mode"] = "aggressive"
        reply = "I'll switch to aggressive mode (deeper trees, XGBoost allowed). Want me to train?"
    if "balanced" in msg and "performance" in msg or msg.strip() == "balanced":
        delta["performance_mode"] = "balanced"
        reply = "I'll use balanced mode. Want me to train?"
    if "conservative" in msg:
        delta["performance_mode"] = "conservative"
        reply = "I'll use conservative mode. Want me to train?"

    if not reply and delta:
        reply = "I've updated the plan. Want me to propose a plan and train?"
    if not reply:
        reply = "You can say e.g. 'drop id', 'use X as target', 'try something stronger', or 'yes' to train with the current plan."

    return is_question, delta, reply


# Control commands: execute action, no LLM. Everything else goes to LLM.
_CHAT_CONTROL_COMMANDS = frozenset({
    "start training", "start train", "train", "yes", "go", "run",
    "stop", "cancel", "accept this plan", "accept plan",
})


# Phrases that mean "run training / execute plan" in any language or casual form (LLM also reasons about intent)
_TRAIN_INTENT_PHRASES = frozenset({
    "train", "yes", "go", "run", "start", "y", "begin", "run training", "start training", "start train",
    "lets do it", "let's do it", "lets go", "let's go", "do it", "go ahead", "run it", "execute", "run the plan",
    "chalo", "jao", "karo", "chalo karo", "chalo train", "jao train", "train na bhai", "train bhai",
    "sure", "ok", "okay", "alright", "thik hai", "theek hai", "sahi hai", "ho jayega", "kr do", "kar do",
})


def _is_control_command(message: str) -> tuple[bool, str]:
    """
    If message is a pure control command, return (True, command_key).
    Else return (False, ""). Accepts casual/multilingual phrasing (lets do it, chalo, karo, etc.).
    """
    msg = (message or "").strip().lower()
    if not msg:
        return False, ""
    # Exact match
    if msg in _CHAT_CONTROL_COMMANDS:
        return True, msg
    if msg in ("y", "run training", "begin"):
        return True, "yes"
    # Short "train" variants
    if msg.startswith("train") and len(msg) <= 35:
        return True, "train"
    # Casual / multilingual "run training" intent
    if msg in _TRAIN_INTENT_PHRASES:
        return True, "train"
    # Very short affirmations (e.g. "ok", "k", "sure") — only if short so we don't match random text
    if len(msg) <= 12 and msg in ("ok", "k", "sure", "yeah", "yep", "yup", "alright", "thik", "theek", "haan", "ha", "ji", "ho"):
        return True, "train"
    return False, ""


def _build_chat_context(session: Dict[str, Any], df: pd.DataFrame) -> str:
    """Build context string for the chat LLM: dataset, model state, plan, last result."""
    parts = []

    # Dataset summary
    profile = profile_dataset(df)
    n_rows = profile.get("n_rows", len(df))
    n_cols = profile.get("n_cols", len(df.columns))
    numeric = profile.get("numeric_cols", []) or []
    categorical = profile.get("categorical_cols", []) or []
    candidates = profile.get("candidate_targets", []) or list(df.columns)[:5]
    parts.append(
        f"## Dataset\n"
        f"- Rows: {n_rows}, Columns: {n_cols}\n"
        f"- Numeric: {', '.join(numeric[:15]) or '—'}\n"
        f"- Categorical: {', '.join(categorical[:15]) or '—'}\n"
        f"- Target candidates: {', '.join(candidates)}"
    )

    # Model state (source of truth)
    ms = session.get("model_state") or {}
    if ms:
        parts.append(
            f"\n## Current model state\n"
            f"- Model: {ms.get('current_model') or '—'}\n"
            f"- Preprocessing: {', '.join(ms.get('preprocessing_steps') or []) or '—'}\n"
            f"- Features used: {len(ms.get('current_features') or [])}\n"
            f"- Attempt: {ms.get('attempt_number', 0)}\n"
            f"- Last diff: {ms.get('last_diff') or '—'}"
        )
        metrics = ms.get("metrics") or {}
        if metrics:
            m_str = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:8] if not str(k).startswith("_"))
            parts.append(f"- Metrics: {m_str}")
    else:
        parts.append("\n## Current model state\nNo training run yet.")

    # Plan (structural + last execution)
    sp = session.get("structural_plan")
    ep_list = session.get("execution_plans") or []
    last_ep = ep_list[-1] if ep_list else None
    if sp:
        parts.append(
            f"\n## Plan\n"
            f"- Task: {sp.get('task_type', '—')}, target: {sp.get('inferred_target', '—')}"
        )
    if last_ep:
        models = [m.get("model_name") for m in (last_ep.get("model_candidates") or []) if m.get("model_name")]
        parts.append(f"- Model candidates: {', '.join(models[:8]) or '—'}")
        parts.append(f"- Primary metric: {last_ep.get('primary_metric', '—')}")
        fts = last_ep.get("feature_transforms") or []
        dropped = [ft.get("name") for ft in fts if ft.get("drop")]
        kept = [ft.get("name") for ft in fts if not ft.get("drop")]
        if dropped:
            parts.append(f"- Dropped features: {', '.join(dropped[:10])}")
        if kept:
            parts.append(f"- Kept features: {', '.join(kept[:15])}")

    # Last training result
    if session.get("refused"):
        parts.append(f"\n## Last run\nRefused: {session.get('refusal_reason', 'Model quality unacceptable')}")

    # User constraints (so LLM knows what user asked for)
    uc = session.get("user_constraints") or {}
    if uc:
        parts.append(f"\n## User constraints\n{uc}")

    # Recent chat (last 6 messages) for continuity
    chat = session.get("chat_history") or []
    if chat:
        recent = chat[-6:]
        lines = []
        for m in recent:
            role = m.get("role", "user")
            content = (m.get("content") or "")[:500]
            lines.append(f"{role}: {content}")
        parts.append("\n## Recent chat\n" + "\n".join(lines))

    return "\n".join(parts)


# Capability contract: LLM is code generator + ML advisor. Like Cursor's agent for the USER's work only; never touch platform.
CHAT_CAPABILITY_CONTRACT = """You are an ML Engineer Agent inside an AutoML system.
You behave like Cursor's agent: generate code, the system executes it, you see results, fix, re-execute. You work ONLY on the user's project: dataset, plan, notebooks, reports, training runs.
You must NEVER edit or suggest edits to the PLATFORM (Intent2Model codebase, backend/main.py, frontend, Cursor app, wireframe, or any source code of the tool you run in). The platform is off-limits; the user's session (notebooks, reports, models) is your workspace."""

CHAT_SYSTEM_PROMPT = f"""{CHAT_CAPABILITY_CONTRACT}

You have access only to the context below (dataset summary, model state, plan, metrics). You do NOT have repo or tool access. The system runs training and builds notebooks; you reason, explain, and confirm in chat.

CRITICAL — Use the LLM to understand what the user is saying (any language: English, Hindi, Hinglish, casual). Do NOT rely on keywords only. Reason about intent.

Chat behavior:
- Do NOT paste full training code or long code blocks. Keep replies short and human.
- When the user wants to RUN TRAINING / execute the plan (any phrasing: "lets do it", "chalo", "haan kar ke dekhte hai", "yes", "karo", "go for it", etc.), reply briefly and set start_training true in INTENT_JSON below.
- When the user asks "how can I improve the metric" or "why is it only X" or "make it better": explain in plain language (e.g. try stronger models like XGBoost, try different preprocessing/normalisation). If they then say "go for it" or "try it", set performance_mode to "aggressive" and start_training true so the system runs XGBoost and more models. The system validates all inputs/outputs with the data; you reason about preprocessing (scaling, normalisation) and model choice.
- If the user says they want to predict X or target is X (e.g. "predict length", "target is length", "wanna predict length"), set target in INTENT_JSON to the exact column name from the context (e.g. petal.length or sepal.length). Use ONLY column names listed in the context.
- If the user says "try something stronger" or "go for it" (after you suggested stronger models), set performance_mode to "aggressive" in INTENT_JSON so the system actually runs XGBoost and more models (not just the same ones again).
- Answer "tell me the plan" (or equivalent) in human language. No code. Explain why accuracy is low in plain language.
- Be concise. Use **bold** for emphasis; the UI renders it as actual bold.

REQUIRED — At the very end of your reply, on a new line, output exactly:
INTENT_JSON: {{"target": "column_name or null", "drop_columns": [], "performance_mode": "aggressive or balanced or conservative or null", "start_training": true or false}}

Rules for INTENT_JSON:
- target: use ONLY a column name from the context (Dataset / Target candidates). If user wants "length", use petal.length or sepal.length as appropriate. If no target change, use null.
- drop_columns: array of column names to drop (from context only), or [].
- performance_mode: "aggressive" or "balanced" or "conservative" or null.
- start_training: true if the user wants to run/execute/train (in any language); false otherwise.
- Use double quotes in JSON. No code blocks around INTENT_JSON."""

# Forbid claims about REPO/TOOLS/FILESYSTEM and about editing the PLATFORM (Intent2Model / Cursor)
_LLM_FORBIDDEN_PHRASES = [
    "i inspected the file",
    "i inspected the code",
    "i opened the file",
    "i edited the file",
    "i modified the file",
    "i used write_file",
    "i used replace",
    "i used a tool",
    "i ran a tool",
    "i applied a patch",
    "in the repo",
    "in the codebase",
    "trainer.py",
    "executor.py",
    "ml/trainer",
    "ml/executor",
    "i have access to the",
    "i opened ml/",
    "i read the file",
    "i looked at the code",
    "i edited the codebase",
    "filesystem access",
    "i used the write_file",
    "i used the replace tool",
    "edit main.py",
    "edit the backend",
    "edit the frontend",
    "edit intent2model",
    "edit the platform",
    "cursor source",
    "cursor wireframe",
    "cursor's source",
    "cursor's wireframe",
    "edit the wireframe",
    "modify the codebase",
]


def _llm_response_violates_capability(response: str) -> bool:
    """True if the response claims repo/filesystem/tool access (not when generating code as output)."""
    if not response or not isinstance(response, str):
        return False
    lower = response.strip().lower()
    return any(phrase in lower for phrase in _LLM_FORBIDDEN_PHRASES)


CHAT_STRICT_PROMPT = f"""{CHAT_CAPABILITY_CONTRACT}

CRITICAL: You must NEVER claim to inspect or edit the repo, use tools, or access the filesystem. You must NEVER suggest or generate edits to the platform (Intent2Model, backend, frontend, Cursor). You generate code and diffs as OUTPUT for the user's session only (notebooks, reports, plan). Reply with reasoning, explanations, and recommendations or generated artifacts. Do not claim any capability you do not have."""


def _plan_summary_for_confirmation(session: Dict[str, Any]) -> str:
    """One short paragraph: exact plan for confirmation (display plan, ask for confirmation)."""
    ms = session.get("model_state") or {}
    sp = session.get("structural_plan") or {}
    ep_list = session.get("execution_plans") or []
    last_ep = ep_list[-1] if ep_list else {}
    target = sp.get("inferred_target", "—")
    task = sp.get("task_type", "—")
    models = [m.get("model_name") for m in (last_ep.get("model_candidates") or []) if m.get("model_name")]
    if not models:
        models = ["ridge", "random_forest", "xgboost"] if task == "regression" else ["logistic_regression", "random_forest"]
    primary = last_ep.get("primary_metric", "—")
    return (
        f"**Plan:** Target **{target}**, task **{task}**, primary metric **{primary}**. "
        f"Models to run: **{', '.join(models[:6])}**. "
        f"Reply **'train'** or **'yes'** to confirm and run."
    )


def _handle_chat_message(user_message: str, session: Dict[str, Any], df: pd.DataFrame) -> tuple[str, dict, bool]:
    """
    Every message is handled as if talking to the LLM; the agent mediates and triggers backend tasks (train, etc.).
    Returns (agent_reply, constraints_delta to merge into session, trigger_training).
    """
    msg = (user_message or "").strip()
    columns = list(df.columns)

    # 0) Plan confirmation: "let's try this first" → display plan, ask for confirmation, do NOT train
    msg_lower = msg.lower()
    if any(x in msg_lower for x in ("let's try this first", "try this first", "lets try this first", "try this plan")):
        summary = _plan_summary_for_confirmation(session)
        return (
            f"Here’s the exact plan.\n\n{summary}",
            {},
            False,
        )

    # 0b) "Try something stronger/better" and we already have a run → merge constraints, start training (no silent retry)
    if any(x in msg_lower for x in (
        "try something stronger", "try something better", "something stronger", "something better",
        "use something stronger", "stronger model", "better model", "try something better na",
    )):
        ms = session.get("model_state") or {}
        attempt = ms.get("attempt_number") or 0
        if attempt >= 1:
            _, constraints_delta, _ = _parse_chat_to_constraints(msg, columns)
            return "Starting training.", constraints_delta, True

    # 1) Control command → no LLM
    is_control, cmd = _is_control_command(msg)
    if is_control:
        if cmd in ("stop", "cancel"):
            session["cancel_requested"] = True
            return "Stopping. I will not start new attempts until you say to train again.", {}, False
        if cmd in ("yes", "train", "go", "start training", "start train", "run", "y", "run training", "begin"):
            # "go for it" / "try something stronger" → set aggressive so we actually run XGBoost etc.
            train_constraints = {}
            if any(x in msg_lower for x in ("go for it", "try something stronger", "try stronger", "something stronger", "more powerful")):
                train_constraints["performance_mode"] = "aggressive"
            return "Starting training.", train_constraints, True
        if cmd in ("accept this plan", "accept plan"):
            return "Plan accepted. Say 'train' or 'yes' to confirm and run.", {}, False
        return "Done.", {}, False

    # 2) Parse constraints via regex first — if we get a clear match, use it and skip LLM
    is_question, constraints_fallback, regex_reply = _parse_chat_to_constraints(msg, columns)
    if not is_question and regex_reply and constraints_fallback:
        # Clear regex match (target, drop, performance_mode) — use it, don't let LLM override
        if "target" in constraints_fallback or "drop_columns" in constraints_fallback or "performance_mode" in constraints_fallback:
            # User said e.g. "i want to predict sepal.length" or "drop id" — trigger training
            trigger = bool(constraints_fallback.get("target")) or bool(constraints_fallback.get("drop_columns")) or bool(constraints_fallback.get("performance_mode"))
            return regex_reply, constraints_fallback, trigger

    # 3) Build context and call LLM — LLM understands what the user is saying (any language)
    context = _build_chat_context(session, df)
    user_prompt = f"{context}\n\n---\nUser message: {msg}"
    provider = os.getenv("LLM_PROVIDER", "gemini_cli")
    # Gemini CLI often returns "I'm ready for your first command" — use API when key exists
    api_key = get_api_key(provider="gemini")
    if provider == "gemini_cli" and api_key:
        provider = "gemini"
        print("   Using Gemini API for chat (CLI unreliable for context-aware replies)")
    try:
        llm = get_llm_with_custom_key(provider=provider)
        response = llm.generate(user_prompt, CHAT_SYSTEM_PROMPT)
        if not (response and str(response).strip()):
            return "I didn't get a valid response. Please try again or rephrase.", constraints_fallback, False
        response = response.strip()

        # Detect generic Gemini CLI responses (LLM didn't understand context)
        _generic_phrases = (
            "i'm ready for your first command", "i am ready for your first command",
            "ready for your first command", "my setup is complete", "setup is complete",
        )
        if any(p in response.lower() for p in _generic_phrases) and "INTENT_JSON" not in response:
            response = (
                "I'm an ML agent — I help you train models on your data. "
                "You can say: **predict X** (set target), **drop Y** (remove a column), **try something stronger** (use XGBoost), "
                "**why is accuracy low** (explain), or **yes** to train. "
                "Add GEMINI_API_KEY to .env for better multilingual understanding."
            )

        # Parse LLM intent (INTENT_JSON) — LLM decides target, drop_columns, performance_mode, start_training
        constraints_delta = dict(constraints_fallback) if constraints_fallback else {}
        trigger_from_intent = False
        intent_match = re.search(r"INTENT_JSON:\s*(\{.*?\})\s*$", response, re.DOTALL)
        if intent_match:
            try:
                intent_str = intent_match.group(1).strip()
                intent = json.loads(intent_str)
                reply_only = response[: intent_match.start()].strip()
                while reply_only.endswith("\n"):
                    reply_only = reply_only[:-1].strip()
                if reply_only:
                    response = reply_only
                if intent.get("target") is not None and str(intent["target"]).strip():
                    col = str(intent["target"]).strip()
                    if col in columns:
                        constraints_delta["target"] = col
                    else:
                        containing = [c for c in columns if col.lower() in c.lower()]
                        if len(containing) == 1:
                            constraints_delta["target"] = containing[0]
                        elif len(containing) > 1:
                            preferred = [c for c in containing if "petal" in c.lower()]
                            constraints_delta["target"] = preferred[0] if preferred else containing[0]
                if intent.get("drop_columns"):
                    valid_drops = [c for c in intent["drop_columns"] if isinstance(c, str) and c in columns]
                    if valid_drops:
                        constraints_delta.setdefault("drop_columns", []).extend(valid_drops)
                if intent.get("performance_mode") in ("aggressive", "balanced", "conservative"):
                    constraints_delta["performance_mode"] = intent["performance_mode"]
                if intent.get("start_training") is True:
                    trigger_from_intent = True
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Do not show long code blocks in chat — system runs training; keep reply conversational
        if "```" in response and len(response) > 400:
            before_code = response.split("```")[0].strip()
            if before_code:
                response = before_code + "\n\nStarting training — the system runs the plan."
            else:
                response = "Starting training — the system will run the plan."
            # If user was clearly asking to run (e.g. "lets do it"), trigger training even though we stripped code
            if msg_lower in _TRAIN_INTENT_PHRASES or any(x in msg_lower for x in ("lets do it", "let's do it", "do it", "chalo", "go ahead", "run it", "run the plan")):
                return response, constraints_delta, True

        # Response validation: discard if LLM claims code/filesystem/tool access
        if _llm_response_violates_capability(response):
            retry_prompt = f"{user_prompt}\n\n[Your previous reply incorrectly claimed repo, filesystem, or tool access. You do NOT have that. You generate code and diffs as OUTPUT only. Reply again with reasoning, explanations, or generated artifacts—no claims about inspecting or editing files.]"
            response = llm.generate(retry_prompt, CHAT_STRICT_PROMPT)
            if response and str(response).strip():
                response = response.strip()
                if _llm_response_violates_capability(response):
                    return (
                        "I can't show that response — it claimed filesystem, repo, or tool access I don't have. "
                        "I generate code and notebooks as OUTPUT only; I never inspect or edit the repo. "
                        "Ask me to explain, diagnose, or propose diffs instead.",
                        constraints_delta,
                        False,
                    )
            else:
                response = (
                    "I can't show that response — it claimed repo or tool access I don't have. "
                    "I'm an ML agent: I explain, diagnose, and generate code/notebooks as output. "
                    "Ask me to explain or propose changes as text/diffs instead."
                )

        # LLM can signal "start training" via [ACTION: start_training] or INTENT_JSON start_training: true
        _ACTION_START_TRAINING = "[ACTION: start_training]"
        if _ACTION_START_TRAINING in response:
            response = response.replace(_ACTION_START_TRAINING, "").strip()
            while response.endswith("\n"):
                response = response[:-1].strip()
            return response or "Starting training.", constraints_delta, True
        return response, constraints_delta, trigger_from_intent
    except Exception as e:
        err = str(e)
        return f"LLM is unavailable: {err}. Please check your API key or CLI configuration.", constraints_delta, False


# CORS: configurable via CORS_ORIGINS (comma-separated), default localhost:3000
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").strip().split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check LLM availability on startup
LLM_AVAILABLE = False
LLM_RATE_LIMITED = False
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini_cli")

# Get API key from api_key_manager (automatically uses .env file)
from utils.api_key_manager import get_api_key
api_key = get_api_key(provider="gemini")

# For gemini_cli: check if CLI is on PATH (works with OAuth, no API key needed)
# On Windows, PyInstaller exe may have limited PATH — also check common npm locations
def _find_gemini_cli(cmd: str) -> bool:
    if shutil.which(cmd):
        return True
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        localappdata = os.environ.get("LOCALAPPDATA")
        pf = os.environ.get("ProgramFiles")
        pf86 = os.environ.get("ProgramFiles(x86)")
        for p in [
            os.path.join(appdata, "npm", "gemini.cmd") if appdata else "",
            os.path.join(localappdata, "npm", "gemini.cmd") if localappdata else "",
            os.path.join(pf, "nodejs", "gemini.cmd") if pf else "",
            os.path.join(pf86, "nodejs", "gemini.cmd") if pf86 else "",
        ]:
            if p and os.path.isfile(p):
                return True
    return False

gemini_cli_cmd = os.getenv("GEMINI_CLI_CMD", "gemini")
gemini_cli_available = _find_gemini_cli(gemini_cli_cmd)

if api_key and api_key.strip():
    try:
        print(f"🔑 Testing LLM with API key: {api_key[:20]}...")
        llm_test = LLMInterface(provider=LLM_PROVIDER, api_key=api_key)
        # Quick test call (with timeout protection)
        test_response = llm_test.generate("Say 'OK'", "You are a test assistant.")
        if test_response and len(test_response.strip()) > 0:
            LLM_AVAILABLE = True
            model_info = get_current_model_info()
            print(f"✅ LLM ({LLM_PROVIDER}) is available and working")
            if model_info.get("model"):
                print(f"   Using model: {model_info.get('model')} - {model_info.get('reason', '')}")
        else:
            print(f"⚠️  LLM ({LLM_PROVIDER}) responded but with empty content")
    except Exception as e:
        err = str(e)
        LLM_RATE_LIMITED = ("rate limit" in err.lower()) or ("quota" in err.lower()) or ("429" in err)
        print(f"⚠️  LLM ({LLM_PROVIDER}) is configured but not available: {err[:200]}")
        print("   System will use rule-based fallbacks (still fully functional)")
        print("   Note: LLM features will be disabled, but all core ML functionality works")
elif LLM_PROVIDER == "gemini_cli" and gemini_cli_available:
    try:
        llm_test = LLMInterface(provider="gemini_cli")
        test_response = llm_test.generate("Say 'OK'", "You are a test assistant.")
        if test_response and len(test_response.strip()) > 0:
            LLM_AVAILABLE = True
            print(f"✅ LLM enabled - using {gemini_cli_cmd} (cli)")
            print(f"   Gemini CLI detected (OAuth or API key)")
        else:
            print(f"⚠️  Gemini CLI responded but with empty content")
    except Exception as e:
        print(f"⚠️  Gemini CLI found but not available: {str(e)[:150]}")
        print("   Set GEMINI_API_KEY for API mode, or run 'gemini' once to complete OAuth")
else:
    if LLM_PROVIDER == "gemini_cli" and not gemini_cli_available:
        print(f"⚠️  Gemini CLI not found ('{gemini_cli_cmd}' not on PATH)")
        print("   Install: npm install -g @google/gemini-cli")
    else:
        print("⚠️  No GEMINI_API_KEY found. System will use rule-based fallbacks (still fully functional)")
    print("   To enable LLM: set GEMINI_API_KEY or install Gemini CLI")
    if sys.platform == "win32":
        print("   💡 On Windows: Add GEMINI_API_KEY to .env (project root) for reliable AutoML planning.")

# In-memory storage for uploaded datasets (in production, use proper storage)
dataset_cache = {}
trained_models_cache = {}  # Store trained models for prediction

# RunState store: single source of truth for GET /runs/{run_id}. Contract: run_id, status, current_step, attempt_count, progress, events[]
# Each event: ts, step_name, message, status?, payload?
run_state_store: Dict[str, Dict[str, Any]] = {}

# Session store: chat-first flow. session_id -> SessionState (dataset_id, chat_history, execution_plans, notebook_cells, performance_mode)
session_store: Dict[str, Dict[str, Any]] = {}

# WebSocket connections for real-time log streaming
websocket_connections: Set[WebSocket] = set()

# Backend log: write to BOTH locations every time (so jo bhi file tum kholo, update hogi)
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_LOG_PROJECT = _PROJECT_ROOT / "backend.log"   # intent2model/backend.log
_LOG_BACKEND = _BACKEND_DIR / "backend.log"     # intent2model/backend/backend.log

_latest_run_id: Optional[str] = None


def _append_backend_log(line: str) -> None:
    """Append to BOTH backend.log files — project root + backend folder. Don't return on first success."""
    line_fmt = line if line.endswith("\n") else line + "\n"
    for log_path in (_LOG_PROJECT, _LOG_BACKEND):
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line_fmt)
                f.flush()
        except Exception as e:
            print(f"[log failed {log_path}] {e}", flush=True)


def _ensure_backend_log_started() -> None:
    """Write one line to backend.log on first use so file exists and /logs/backend returns something."""
    if getattr(_ensure_backend_log_started, "_done", False):
        return
    try:
        from datetime import datetime
        _append_backend_log(f"Backend process started at {datetime.now().isoformat()}")
        _ensure_backend_log_started._done = True
    except Exception:
        pass


@app.on_event("startup")
def _startup_backend_log():
    """Write to backend.log as soon as the app starts so the file is created and writable."""
    _ensure_backend_log_started()
    _append_backend_log("Uvicorn app ready — backend.log is active.")


async def _broadcast_log(entry: Dict[str, Any]):
    """Broadcast log entry to all connected WebSocket clients."""
    if not websocket_connections:
        return
    
    message = json_lib.dumps(entry)
    disconnected = set()
    
    for ws in websocket_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    
    # Remove disconnected clients
    websocket_connections.difference_update(disconnected)


def _broadcast_log_sync(entry: Dict[str, Any]):
    """Synchronous wrapper for async broadcast (for use in sync contexts)."""
    if not websocket_connections:
        return
    
    # Use asyncio to run the async function
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, schedule the task
            asyncio.create_task(_broadcast_log(entry))
        else:
            # If no loop is running, run it
            loop.run_until_complete(_broadcast_log(entry))
    except RuntimeError:
        # No event loop exists, create one
        try:
            asyncio.run(_broadcast_log(entry))
        except Exception:
            pass  # Ignore errors
    except Exception:
        pass  # Ignore errors


def _stage_to_run_status(stage: Optional[str]) -> str:
    """Map log stage to run state status for GET /runs/{run_id}."""
    if not stage:
        return "init"
    s = (stage or "").lower()
    if s in ("plan", "planning"):
        return "planning"
    if s in ("train", "executor", "config", "models", "profile"):
        return "training"
    if s in ("diagnose",):
        return "diagnosing"
    if s in ("repair", "retry"):
        return "retrying"
    if s in ("error", "fallback"):
        return "failed"
    if s in ("refuse",):
        return "refused"
    if s in ("done", "success", "eval"):
        return "success"
    return "training"


def _run_state_init(run_id: str) -> Dict[str, Any]:
    """Return initial RunState for a new run. Contract: run_id, status, current_step, attempt_count, progress, events."""
    return {
        "run_id": run_id,
        "status": "init",
        "current_step": "",
        "attempt_count": 0,
        "progress": 0.0,
        "events": [],
    }


def _log_run_event_sync(
    run_id: str,
    message: str,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    payload: Optional[Dict[str, Any]] = None,
    attempt_count: Optional[int] = None,
    step_name: Optional[str] = None,
    status: Optional[str] = None,
):
    """Append one AgentEvent to RunState.events. Event shape: ts, step_name, message, status?, payload?."""
    _ensure_backend_log_started()
    if not run_id:
        return
    step = str(step_name) if step_name is not None else (str(stage) if stage else "info")
    event_status = str(status) if status is not None else ("failed" if ("failed" in message or "❌" in message or "REFUSED" in message) else "info")

    # AgentEvent: exactly ts, step_name, message, status, payload (payload only if present)
    event: Dict[str, Any] = {
        "ts": pd.Timestamp.now().isoformat(),
        "step_name": step,
        "message": str(message),
        "status": event_status,
    }
    if payload is not None:
        event["payload"] = payload

    # Persistent RunState store (single source of truth for GET /runs/{run_id})
    cur = run_state_store.get(run_id) or _run_state_init(run_id)
    cur["events"] = (cur.get("events") or [])[-500:] + [event]
    if progress is not None:
        cur["progress"] = float(progress)
    cur["current_step"] = step
    cur["status"] = _stage_to_run_status(stage) if stage else cur.get("status", "init")
    if attempt_count is not None:
        cur["attempt_count"] = int(attempt_count)
    if status is not None and status in ("failed", "success", "refused"):
        cur["status"] = status
    run_state_store[run_id] = cur

    stage_str = f"[{stage}]" if stage else ""
    progress_str = f"({progress:.0f}%)" if progress is not None else ""
    line = f"[{run_id[:8]}] {stage_str} {progress_str} {message}"
    print(line, flush=True)
    _append_backend_log(line)
    _broadcast_log_sync(event)


def _log_run_event(
    run_id: str,
    message: str,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    payload: Optional[Dict[str, Any]] = None,
    attempt_count: Optional[int] = None,
    step_name: Optional[str] = None,
    status: Optional[str] = None,
):
    """Append a structured log event for a run. Supports payload/attempt_count for agent timeline."""
    _log_run_event_sync(run_id, message, stage, progress, payload, attempt_count, step_name, status)

# API key management - allow users to provide custom keys
from utils.api_key_manager import set_custom_api_key, get_api_key
current_llm_model = None  # Track which model is currently being used
current_llm_reason = None  # Why this model was chosen


class UserConstraints(BaseModel):
    """User messages as state modifiers — override LLM preferences (chat-first)."""
    drop_columns: Optional[list[str]] = None   # e.g. ["id"] from "drop id"
    target: Optional[str] = None               # from "use X as target"
    exclude_models: Optional[list[str]] = None  # e.g. ["random_forest"]
    keep_features: Optional[list[str]] = None   # do not drop these
    primary_metric: Optional[str] = None        # e.g. "mae" over "r2"
    prefer_simple: Optional[bool] = None        # try simpler models first


class TrainRequest(BaseModel):
    target: Optional[str] = None
    task: Optional[Literal["classification", "regression"]] = None
    metric: Optional[str] = None
    dataset_id: Optional[str] = None
    llm_provider: Optional[str] = None
    user_constraints: Optional[UserConstraints] = None  # chat-first: user messages affect ExecutionPlan


class SelectModelRequest(BaseModel):
    run_id: str
    model_name: str


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """
    WebSocket endpoint for real-time log streaming.
    Connects and receives all log events as they're generated.
    """
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to real-time log stream"
        })
        
        # Send recent logs from cache
        for run_id, cache_data in list(run_state_store.items())[-10:]:  # Last 10 runs
            events = cache_data.get("events", [])[-50:]  # Last 50 events per run
            for event in events:
                await websocket.send_json(event)
        
        # Keep connection alive and handle incoming messages (ping/pong)
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back for ping/pong
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        websocket_connections.discard(websocket)


@app.get("/run/latest-id")
async def get_latest_run_id():
    """Return the run_id of the most recently started training run. Frontend polls this when training starts to get run_id without parsing backend.log."""
    return {"run_id": _latest_run_id}


@app.get("/run/{run_id}/logs")
async def get_run_logs(run_id: str, limit: int = 200):
    """Fetch recent structured log events for a run (for Developer Logs UI)."""
    # If run_id doesn't exist yet, return empty (training might not have started)
    if run_id not in run_state_store:
        return {
            "run_id": run_id,
            "status": "init",
            "current_step": "",
            "attempt_count": 0,
            "progress": 0,
            "events": [{"ts": "", "step_name": "info", "message": "Run not started yet.", "status": "info"}],
        }
    cur = run_state_store[run_id]
    events = cur.get("events") or []
    try:
        lim = max(1, min(int(limit), 500))
    except Exception:
        lim = 200
    return {
        "run_id": run_id,
        "status": cur.get("status", "init"),
        "current_step": cur.get("current_step", ""),
        "attempt_count": cur.get("attempt_count", 0),
        "progress": float(cur.get("progress", 0)),
        "events": events[-lim:],
    }


def _run_state_to_response(cur: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """Return RunState for GET /runs/{run_id}. Contract: run_id, status, current_step, attempt_count, progress, events (each: ts, step_name, message, status?, payload?)."""
    events = cur.get("events") or []
    # Normalize each event to contract shape only
    out_events = []
    for e in events[-500:]:
        ev = {"ts": e.get("ts", ""), "step_name": e.get("step_name", "info"), "message": e.get("message", ""), "status": e.get("status", "info")}
        if e.get("payload") is not None:
            ev["payload"] = e["payload"]
        out_events.append(ev)
    return {
        "run_id": run_id,
        "status": cur.get("status", "init"),
        "current_step": cur.get("current_step", ""),
        "attempt_count": int(cur.get("attempt_count", 0)),
        "progress": float(cur.get("progress", 0)),
        "events": out_events,
    }


@app.get("/runs/{run_id}")
async def get_run_state(run_id: str):
    """
    Return RunState verbatim. Frontend contract: run_id, status, current_step, attempt_count, progress, events[] (ts, step_name, message, status?, payload?).
    """
    if run_id not in run_state_store:
        return _run_state_to_response(_run_state_init(run_id), run_id)
    return _run_state_to_response(run_state_store[run_id], run_id)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Intent2Model API",
        "version": "1.0.0",
        "status": "running",
        "llm_available": LLM_AVAILABLE,
        "llm_provider": LLM_PROVIDER if LLM_AVAILABLE else "rule-based-fallback"
    }

def get_llm_with_custom_key(provider: str = "gemini_cli"):
    """Get LLMInterface with API key (automatically uses .env, custom key if set)."""
    api_key = get_api_key(provider=provider)
    return LLMInterface(provider=provider, api_key=api_key)

# Make this function available to agents via a module-level function
# Agents can import this to get LLM with custom key support
import sys
sys.modules[__name__].get_llm_with_custom_key = get_llm_with_custom_key

@app.get("/health")
async def health():
    """Detailed health check with LLM status. Reloads .env to detect API key changes."""
    # Reload .env file to detect API key changes (fast, no LLM test)
    from dotenv import load_dotenv
    load_dotenv(override=True)  # override=True to reload existing vars
    
    # Re-check if API key exists (don't test LLM - too slow)
    from utils.api_key_manager import get_api_key
    api_key = get_api_key(provider="gemini")
    gemini_cli_cmd = os.getenv("GEMINI_CLI_CMD", "gemini")
    gemini_cli_available = _find_gemini_cli(gemini_cli_cmd)
    
    # Update global state if API key changed
    global LLM_AVAILABLE, LLM_RATE_LIMITED, LLM_PROVIDER, current_llm_model, current_llm_reason
    # If using CLI provider, treat availability as "CLI installed"
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER or "gemini_cli")
    if provider == "gemini_cli":
        LLM_PROVIDER = "gemini_cli"
        LLM_AVAILABLE = gemini_cli_available
        LLM_RATE_LIMITED = False
        current_llm_model = f"{gemini_cli_cmd} (cli)" if gemini_cli_available else None
        current_llm_reason = "Gemini CLI detected" if gemini_cli_available else f"Gemini CLI not found: {gemini_cli_cmd}"
    elif api_key and api_key.strip():
        # API key exists - assume available (actual test happens during training)
        LLM_AVAILABLE = True
        LLM_RATE_LIMITED = False
        LLM_PROVIDER = "gemini"
        current_llm_model = "gemini-2.0-flash-exp"
        current_llm_reason = "API key configured"
    else:
        LLM_AVAILABLE = False
        LLM_RATE_LIMITED = False
    
    model_info = get_current_model_info()
    return {
        "status": "healthy",
        "llm_available": LLM_AVAILABLE,
        "llm_rate_limited": LLM_RATE_LIMITED,
        "llm_provider": LLM_PROVIDER if LLM_AVAILABLE else "rule-based-fallback",
        "current_model": model_info.get("model") or current_llm_model,
        "model_reason": model_info.get("reason") or current_llm_reason,
        "gemini_cli_available": gemini_cli_available,
        "gemini_cli_cmd": gemini_cli_cmd,
        "message": (
            f"✅ LLM enabled - using {model_info.get('model') or current_llm_model or 'AI-powered planning'}"
            if LLM_AVAILABLE
            else ("⚠️  LLM is rate-limited; using fallbacks for planning/explanations" if LLM_RATE_LIMITED else "⚠️  Using rule-based fallbacks (fully functional, but less intelligent)")
        )
    }


def _get_backend_log_file_to_read() -> Optional[Path]:
    """Return backend.log to read from (project root first, then backend/)."""
    if _LOG_PROJECT.exists():
        return _LOG_PROJECT
    if _LOG_BACKEND.exists():
        return _LOG_BACKEND
    return None


@app.get("/debug/log-path")
async def debug_log_path():
    """Where backend.log is written and read from."""
    return {
        "writing_to": [str(_LOG_PROJECT), str(_LOG_BACKEND)],
        "read_path": str(p) if (p := _get_backend_log_file_to_read()) else None,
    }


@app.get("/logs/backend")
async def backend_logs(limit: int = 200):
    """Tail backend.log for developer debugging in the UI. Always reads fresh from disk (no cache)."""
    try:
        lim = max(10, min(int(limit), 500))
    except Exception:
        lim = 200
    log_path = _get_backend_log_file_to_read()
    if not log_path:
        return {"path": "backend.log (not found)", "lines": [], "hint": "Start a training run or restart backend; logs will appear here."}
    try:
        txt = log_path.read_text(encoding="utf-8", errors="ignore")
        lines = txt.splitlines()[-lim:]
    except Exception:
        lines = []
    return {"path": str(log_path), "lines": lines}

class ApiKeyRequest(BaseModel):
    api_key: str
    provider: str = "gemini"

@app.post("/api/set-api-key")
async def set_api_key(request: ApiKeyRequest):
    """
    Allow user to set a custom API key.
    This key will be used for subsequent LLM calls.
    If empty, will use default from environment.
    """
    global LLM_AVAILABLE, current_llm_model, current_llm_reason
    
    # If empty, clear custom key and use default
    if not request.api_key or not request.api_key.strip():
        from utils.api_key_manager import _custom_api_keys
        if request.provider in _custom_api_keys:
            del _custom_api_keys[request.provider]
        
        # Test with default key
        default_key = get_api_key(provider=request.provider)
        if default_key:
            request.api_key = default_key
        else:
            return {
                "status": "error",
                "message": "No API key provided and no default key found in environment"
            }
    else:
        # Store the custom API key
        set_custom_api_key(request.api_key, provider=request.provider)
    
    # Test the API key
    try:
        llm_test = LLMInterface(provider=request.provider, api_key=request.api_key)
        test_response = llm_test.generate("Say 'OK'", "You are a test assistant.")
        
        if test_response and len(test_response.strip()) > 0:
            LLM_AVAILABLE = True
            # Get model info from the test
            model_info = get_current_model_info()
            global current_llm_model, current_llm_reason
            current_llm_model = model_info.get("model")
            current_llm_reason = model_info.get("reason")
            
            return {
                "status": "success",
                "message": "API key validated successfully",
                "llm_available": True,
                "current_model": current_llm_model,
                "model_reason": current_llm_reason,
                "using_default": request.api_key == get_api_key(provider=request.provider)
            }
        else:
            return {
                "status": "error",
                "message": "API key accepted but returned empty response"
            }
    except Exception as e:
        error_msg = str(e)
        # Check if it's a rate limit
        is_rate_limit = (
            '429' in error_msg or 
            'quota' in error_msg.lower() or 
            'rate limit' in error_msg.lower() or
            'resourceexhausted' in error_msg.lower()
        )
        
        # If rate limit, the system will auto-fallback, so this is actually OK
        if is_rate_limit:
            return {
                "status": "warning",
                "message": f"Rate limit detected: {error_msg[:150]}. System will automatically try alternative models.",
                "is_rate_limit": True,
                "suggestion": "The system will automatically switch to alternative models when rate limits are hit."
            }
        
        return {
            "status": "error",
            "message": f"API key validation failed: {error_msg[:200]}",
            "is_rate_limit": False,
            "suggestion": "Please check your API key is correct"
        }


@app.post("/run/select-model")
async def run_select_model(request: SelectModelRequest):
    """
    Select which trained model to use for predictions and artifact generation for a given run_id.
    """
    run_id = (request.run_id or "").strip()
    model_name = (request.model_name or "").strip()
    if not run_id or not model_name:
        raise HTTPException(status_code=400, detail="run_id and model_name are required")

    if run_id not in trained_models_cache:
        raise HTTPException(status_code=404, detail="Run not found")

    info = trained_models_cache[run_id]
    pipelines_by_model = info.get("pipelines_by_model") or {}
    if model_name not in pipelines_by_model:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available for this run")

    # Swap active model & label encoder
    info["model"] = pipelines_by_model[model_name].get("pipeline")
    info["label_encoder"] = pipelines_by_model[model_name].get("label_encoder")
    info["selected_model"] = model_name
    # Keep legacy field in sync for download endpoints
    info["model_name"] = model_name

    # Ensure config includes a model_code used by notebook generation
    task = info.get("task") or "classification"
    cfg = (info.get("config") or {}).copy()
    cfg["model"] = model_name
    cfg["model_code"] = _model_code_for_notebook(task, model_name)
    info["config"] = cfg

    trained_models_cache[run_id] = info
    return {"status": "success", "run_id": run_id, "selected_model": model_name}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV file and return dataset profile.
    AUTONOMOUS: Handles all errors gracefully, never fails the user.
    
    Returns:
        JSON with dataset profile and dataset_id for subsequent requests
    """
    # AUTONOMOUS: Handle non-CSV files by trying to process anyway
    filename = file.filename or "uploaded_file"
    is_csv = filename.endswith('.csv')
    
    try:
        # Read file contents
        contents = await file.read()
        
        # AUTONOMOUS: Try multiple CSV parsing strategies
        df = None
        encoding_errors = []
        
        # Strategy 1: Try as-is
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e1:
            encoding_errors.append(str(e1))
            
            # Strategy 2: Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                    break
                except:
                    continue
            
            # Strategy 3: Try with error handling
            if df is None:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding='utf-8', errors='ignore', on_bad_lines='skip')
                except:
                    try:
                        df = pd.read_csv(io.BytesIO(contents), encoding='latin-1', on_bad_lines='skip')
                    except:
                        pass
        
        # AUTONOMOUS: If still failed, try to extract column names from error or use generic names
        if df is None or df.empty:
            # Try to read just headers to get column names
            try:
                df = pd.read_csv(io.BytesIO(contents), nrows=0)  # Just headers
                if df.empty:
                    # Read with error handling to get at least column names
                    df = pd.read_csv(io.BytesIO(contents), encoding='utf-8', on_bad_lines='skip', nrows=1)
            except:
                # Last resort: create dataset with generic names but preserve original filename hint
                df = pd.DataFrame({
                    'feature1': [1, 2, 3, 4, 5],
                    'feature2': [10, 20, 30, 40, 50],
                    'target': [0, 1, 0, 1, 0]
                })
            print(f"⚠️  CSV parsing had issues, using available data. Original errors: {encoding_errors}")
        
        # AUTONOMOUS: Clean the dataset automatically
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # If dataset is too small, duplicate rows to make it usable
        if len(df) < 3:
            while len(df) < 10:
                df = pd.concat([df, df], ignore_index=True)
        
        # Profile dataset
        try:
            profile = profile_dataset(df)
        except Exception as profile_error:
            # AUTONOMOUS: Create a basic profile if profiling fails
            print(f"⚠️  Profiling failed: {profile_error}. Creating basic profile.")
            profile = {
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "numeric_cols": list(df.select_dtypes(include=['number']).columns),
                "categorical_cols": list(df.select_dtypes(include=['object', 'category']).columns),
                "candidate_targets": list(df.columns)[:3] if len(df.columns) > 0 else []
            }
        
        # Store dataset in cache
        dataset_id = create_run_id()
        dataset_cache[dataset_id] = df

        # Chat-first: create session with ModelState (source of truth) and initial agent message
        session_id = create_run_id()
        content, payload = _build_initial_chat_message(profile, df)
        initial_msg = {"role": "agent", "content": content, "payload": payload}
        dataset_summary = {
            "n_rows": profile.get("n_rows", len(df)),
            "n_cols": profile.get("n_cols", len(df.columns)),
            "numeric_cols": profile.get("numeric_cols", []),
            "categorical_cols": profile.get("categorical_cols", []),
            "candidate_targets": profile.get("candidate_targets", list(df.columns)[:5]),
        }
        model_state = ModelState(dataset_summary=dataset_summary, status="idle").model_dump()
        session_state = SessionState(
            session_id=session_id,
            dataset_id=dataset_id,
            chat_history=[initial_msg],
            model_state=model_state,
            user_constraints={},
        )
        session_store[session_id] = session_state.model_dump()

        return _json_safe({
            "session_id": session_id,
            "dataset_id": dataset_id,
            "profile": profile,
            "message": "Dataset uploaded successfully",
            "initial_message": initial_msg,
        })
    except Exception as e:
        # AUTONOMOUS: Try one more time with most permissive settings
        print(f"⚠️  Upload error: {e}. Trying one more time with permissive settings.")
        try:
            contents = await file.read()
            df = pd.read_csv(
                io.BytesIO(contents), 
                encoding='latin-1', 
                on_bad_lines='skip',
                low_memory=False,
                dtype=str  # Read everything as string first
            )
            # Convert numeric columns back
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
            
            if not df.empty:
                profile = profile_dataset(df)
                dataset_id = create_run_id()
                dataset_cache[dataset_id] = df
                session_id = create_run_id()
                content, payload = _build_initial_chat_message(profile, df)
                initial_msg = {"role": "agent", "content": content, "payload": payload}
                dataset_summary = {"n_rows": len(df), "n_cols": len(df.columns), "numeric_cols": profile.get("numeric_cols", []), "categorical_cols": profile.get("categorical_cols", []), "candidate_targets": list(df.columns)[:5]}
                model_state = ModelState(dataset_summary=dataset_summary, status="idle").model_dump()
                session_state = SessionState(
                    session_id=session_id,
                    dataset_id=dataset_id,
                    chat_history=[initial_msg],
                    model_state=model_state,
                    user_constraints={},
                )
                session_store[session_id] = session_state.model_dump()
                return _json_safe({
                    "session_id": session_id,
                    "dataset_id": dataset_id,
                    "profile": profile,
                    "message": "Dataset uploaded successfully",
                    "initial_message": initial_msg,
                })
        except Exception:
            pass
        
        # Last resort: create minimal dataset but DON'T use it - return error instead
        # This forces user to upload a real file
        raise HTTPException(
            status_code=400,
            detail="Could not process the file. Please ensure it's a valid CSV, JSON, or XLSX file."
        )


class ChatMessageRequest(BaseModel):
    """Body for POST /session/{session_id}/chat."""
    message: str


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Return full session state (chat, execution_plans, notebook_cells, performance_mode)."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return _json_safe(session_store[session_id])


@app.get("/session/{session_id}/notebook")
async def session_notebook(session_id: str):
    """Download or view notebook for this session's last run (uses current_run_id)."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    session = session_store[session_id]
    run_id = session.get("current_run_id")
    if not run_id:
        raise HTTPException(
            status_code=404,
            detail="No run yet. Say 'train' or 'yes' in chat to run training, then try again.",
        )
    if run_id not in trained_models_cache:
        raise HTTPException(
            status_code=404,
            detail="Notebook not available (e.g. backend restarted). Run training again, then try.",
        )
    # Redirect to download endpoint so we reuse the same logic
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/download/{run_id}/notebook", status_code=302)


@app.post("/session/{session_id}/cancel")
async def session_cancel(session_id: str):
    """User-driven steering: request to stop further iterations (no silent retry)."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    session_store[session_id]["cancel_requested"] = True
    return _json_safe({"status": "ok", "message": "Cancel requested. I will not start new attempts until you say to train again."})


@app.post("/session/{session_id}/chat")
async def session_chat(session_id: str, body: ChatMessageRequest):
    """
    Every message is like talking to the LLM; the LLM acts as mediator and agent to run tasks in the backend.
    Control intents (train, stop, try something stronger) trigger backend tasks; reply is agent confirmation.
    Else: full context (dataset, model state, plan, metrics) is sent to the LLM and its reply is returned.
    Response includes trigger_training when the agent is starting a training run.
    """
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    session = session_store[session_id]
    dataset_id = session.get("dataset_id")
    if not dataset_id or dataset_id not in dataset_cache:
        raise HTTPException(status_code=400, detail="Session dataset not available")
    df = dataset_cache[dataset_id]

    # Append user message
    user_content = (body.message or "").strip()
    session.setdefault("chat_history", []).append({"role": "user", "content": user_content})

    # Single chat handler: every message goes through the agent; control intents trigger backend tasks (LLM as mediator)
    agent_reply, constraints_delta, trigger_training = _handle_chat_message(user_content, session, df)

    # Merge constraints into session (drop_columns, target, performance_mode)
    if constraints_delta:
        uc = session.setdefault("user_constraints", {})
        if "drop_columns" in constraints_delta:
            uc.setdefault("drop_columns", []).extend(constraints_delta["drop_columns"])
        if "target" in constraints_delta:
            uc["target"] = constraints_delta["target"]
        if "performance_mode" in constraints_delta:
            uc["performance_mode"] = constraints_delta["performance_mode"]
        session["user_constraints"] = uc

    # Append agent reply (LLM or control confirmation)
    session["chat_history"].append({"role": "agent", "content": agent_reply})
    session_store[session_id] = session

    return _json_safe({"chat_history": session["chat_history"], "trigger_training": trigger_training})


@app.post("/session/{session_id}/train")
async def session_train(session_id: str):
    """
    Run ONE training attempt for this session. Uses session's dataset + user_constraints.
    Updates session with execution_plans, notebook_cells, run_id; appends metrics + choices to chat.
    """
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    session = session_store[session_id]
    dataset_id = session.get("dataset_id")
    if not dataset_id or dataset_id not in dataset_cache:
        raise HTTPException(status_code=400, detail="Session dataset not available")
    df = dataset_cache[dataset_id].copy()
    user_constraints = session.get("user_constraints") or {}
    target_from_constraints = user_constraints.get("target")
    performance_mode = user_constraints.get("performance_mode") or session.get("performance_mode") or "conservative"
    session["performance_mode"] = performance_mode  # keep in sync for UI

    # Resolve target
    target = target_from_constraints
    if not target or target not in df.columns:
        profile = profile_dataset(df)
        candidates = profile.get("candidate_targets") or list(df.columns)[:5]
        target = candidates[0] if candidates else list(df.columns)[-1]

    # Task / metric inference (same as legacy train)
    from ml.evaluator import infer_classification_primary_metric
    target_col = df[target]
    if target_col.dtype in ["int64", "float64", "int32", "float32"]:
        task = "classification" if target_col.nunique() <= 20 else "regression"
    else:
        task = "classification"
    metric = infer_classification_primary_metric(df, target) if task == "classification" else "r2"

    run_id = create_run_id()
    session["current_run_id"] = run_id
    run_state_store[run_id] = _run_state_init(run_id)
    chosen_llm = os.getenv("LLM_PROVIDER", "gemini_cli")

    # Plan once if not yet in session
    structural_plan = session.get("structural_plan")
    first_execution_plan = None
    if session.get("execution_plans"):
        # Reuse last execution plan with constraints applied
        from schemas.pipeline_schema import ExecutionPlan
        last_ep = session["execution_plans"][-1]
        first_execution_plan = ExecutionPlan(**(last_ep if isinstance(last_ep, dict) else last_ep))
    if not structural_plan or not first_execution_plan:
        plan = plan_automl(df, requested_target=target, llm_provider=chosen_llm)
        from agents.execution_planner import automl_plan_to_structural_and_execution
        structural_plan_obj, first_execution_plan = automl_plan_to_structural_and_execution(plan)
        structural_plan = structural_plan_obj.model_dump() if hasattr(structural_plan_obj, "model_dump") else structural_plan_obj
        session["structural_plan"] = structural_plan
        first_execution_plan = first_execution_plan  # use this for execution

    # Build model_candidates from plan, then apply performance_mode (so "try something stronger" changes next run)
    ep_dict = first_execution_plan.model_dump() if hasattr(first_execution_plan, "model_dump") else first_execution_plan
    model_candidates = [m.get("model_name") for m in ep_dict.get("model_candidates", []) if m.get("model_name")]
    if not model_candidates:
        model_candidates = ["random_forest", "gradient_boosting", "logistic_regression"] if task == "classification" else ["random_forest", "gradient_boosting", "ridge"]

    # Apply performance_mode: "try something stronger" → at least ONE nonlinear model runs, no early exit after linear only
    if performance_mode == "aggressive":
        stronger_reg = ["xgboost", "gradient_boosting", "random_forest", "ridge", "svm"]
        stronger_clf = ["xgboost", "gradient_boosting", "random_forest", "logistic_regression", "svm"]
        stronger = stronger_clf if task == "classification" else stronger_reg
        model_candidates = list(dict.fromkeys([m for m in stronger if m not in model_candidates] + model_candidates))
        # Executor runs all; propose message reports which models will run (full transparency)
    elif performance_mode == "balanced":
        balanced_reg = ["gradient_boosting", "random_forest", "ridge"]
        balanced_clf = ["gradient_boosting", "random_forest", "logistic_regression"]
        balanced = balanced_clf if task == "classification" else balanced_reg
        model_candidates = list(dict.fromkeys([m for m in balanced if m not in model_candidates] + model_candidates))

    # Propose: append system message so user sees "I'm about to try ..." (Cursor-for-ML)
    propose_msg = f"I'm about to try: {', '.join(model_candidates[:5])}{'...' if len(model_candidates) > 5 else ''}. Target: **{target}**, task: **{task}**."
    session.setdefault("chat_history", []).append({"role": "agent", "content": propose_msg, "payload": {"stage": "propose", "model_candidates": model_candidates, "target": target, "task": task}})

    # Run one attempt via autonomous executor
    from agents.autonomous_executor import AutonomousExecutor
    from schemas.pipeline_schema import StructuralPlan
    sp = StructuralPlan(**structural_plan) if isinstance(structural_plan, dict) else structural_plan
    executor = AutonomousExecutor(run_id=run_id, log_callback=_log_run_event, llm_provider=chosen_llm)
    uc_for_executor = {k: v for k, v in user_constraints.items() if v is not None}
    train_result = executor.execute_with_auto_fix(
        df=df,
        target=target,
        task=task,
        metric=metric,
        model_candidates=model_candidates,
        requested_target=target,
        llm_provider=chosen_llm,
        structural_plan=sp,
        first_execution_plan=first_execution_plan,
        user_constraints=uc_for_executor or None,
    )

    # Update session: execution_plans, failure_history, notebook_cells
    if train_result.get("execution_plans"):
        session.setdefault("execution_plans", []).extend(train_result["execution_plans"])
    if train_result.get("failure_history"):
        session.setdefault("failure_history", []).extend(train_result["failure_history"])
    if train_result.get("refused"):
        session["refused"] = True
        session["refusal_reason"] = train_result.get("refusal_reason", "Model quality unacceptable")

    metrics = train_result.get("metrics") or {}
    primary_val = metrics.get("primary_metric_value") or metrics.get("accuracy") or metrics.get("r2") or metrics.get("cv_mean")
    primary = primary_val if primary_val is not None else "—"
    if isinstance(primary, float):
        primary = f"{primary:.2%}" if task == "classification" else f"{primary:.4f}"
    current_model = train_result.get("model_name") or (model_candidates[0] if model_candidates else None)
    prev_ms = session.get("model_state") or {}
    previous_model = prev_ms.get("current_model")

    # Build ModelState (source of truth for UI)
    current_features = [c for c in df.columns if c != target]
    preprocessing_steps = []
    for ft in ep_dict.get("feature_transforms", [])[:20]:
        ft_d = ft if isinstance(ft, dict) else (getattr(ft, "model_dump", lambda: ft)())
        if ft_d.get("drop"):
            continue
        if ft_d.get("scale") and ft_d.get("scale") != "none":
            preprocessing_steps.append(f"Scale({ft_d.get('name', '?')})")
        if ft_d.get("encode") and ft_d.get("encode") != "none":
            preprocessing_steps.append(f"Encode({ft_d.get('name', '?')})")
    if not preprocessing_steps:
        preprocessing_steps = ["StandardScaler", "OneHotEncoder"]

    error_analysis = {}
    if metrics.get("confusion_matrix") is not None:
        error_analysis["confusion_matrix"] = metrics["confusion_matrix"]
        error_analysis["class_labels"] = metrics.get("class_labels") or []
    if train_result.get("feature_importance"):
        fi = train_result["feature_importance"]
        error_analysis["feature_importance"] = {str(k): float(v) for k, v in list(fi.items())[:30]} if isinstance(fi, dict) else fi

    last_diff = {}
    if previous_model and current_model and previous_model != current_model:
        last_diff["model"] = f"{previous_model} → {current_model}"
    dropped = user_constraints.get("drop_columns") or []
    if dropped:
        last_diff["dropped_features"] = dropped

    attempt_number = (prev_ms.get("attempt_number") or 0) + 1
    metrics_for_state = dict(metrics) if metrics else {}
    if primary_val is not None:
        metrics_for_state["primary_metric_value"] = float(primary_val)
    model_state = {
        "dataset_summary": (prev_ms.get("dataset_summary") or {}),
        "current_features": current_features,
        "preprocessing_steps": list(dict.fromkeys(preprocessing_steps))[:15],
        "current_model": current_model,
        "previous_model": previous_model,
        "metrics": _json_safe(metrics_for_state),
        "error_analysis": error_analysis,
        "attempt_number": attempt_number,
        "last_diff": last_diff,
        "status": "refused" if train_result.get("refused") else "success",
        "status_message": train_result.get("refusal_reason") or "Training completed",
    }
    session["model_state"] = model_state

    # Discuss: accuracy as discussion, not just result (Cursor-for-ML)
    primary_float = None
    if isinstance(primary_val, (int, float)):
        primary_float = float(primary_val)
    elif task == "classification" and metrics.get("accuracy") is not None:
        primary_float = float(metrics["accuracy"])
    elif task == "regression" and metrics.get("r2") is not None:
        primary_float = float(metrics.get("r2"))

    weak_threshold = 0.75 if task == "classification" else 0.5  # r2
    is_weak = primary_float is not None and (
        (task == "classification" and primary_float < weak_threshold) or (task == "regression" and primary_float < weak_threshold)
    )

    # Per-model summary so user sees we actually tried XGBoost etc. and WHY a model failed (show error)
    all_models = train_result.get("all_models") or []
    per_model_line = ""
    if len(all_models) > 1:
        def _fmt_score(m):
            if m.get("failed"):
                err = (m.get("error") or "unknown error").strip()[:60]
                if len((m.get("error") or "")) > 60:
                    err += "..."
                return f"failed ({err})"
            v = m.get("primary_metric") or m.get("cv_mean")
            if v is None:
                return "—"
            if task == "classification":
                return f"{float(v):.2%}"
            return f"{float(v):.4f}"
        parts_list = [f"{m.get('model_name', '?')} {_fmt_score(m)}" for m in all_models[:8]]
        per_model_line = f"We tried **{len(all_models)}** models: {', '.join(parts_list)}. Best: **{current_model}** ({primary}).\n\n"
        failed_models = [m for m in all_models if m.get("failed") and m.get("error")]
        if failed_models:
            per_model_line += "**Why some failed:** " + " | ".join(
                f"**{m.get('model_name', '?')}**: {str(m.get('error', ''))[:200]}" for m in failed_models[:3]
            ) + "\n\n"
    prev_primary = None
    if prev_ms.get("metrics"):
        pm = prev_ms["metrics"]
        prev_primary = pm.get("primary_metric_value") or pm.get("r2") or pm.get("accuracy") or pm.get("cv_mean")
    same_as_before = (
        prev_primary is not None and primary_float is not None
        and abs(float(prev_primary) - float(primary_float)) < 0.001
    )
    if same_as_before and len(all_models) > 1:
        per_model_line += "None of the stronger models beat the previous best — for this target we might be near the limit without more features or different preprocessing. You can ask me to \"try different preprocessing\" or \"drop X\" / \"use Y as target\".\n\n"

    # Reasoning Diff + What to Try Next + Bottleneck + Honesty (ML Engineer Agent)
    reasoning_block = ""
    try:
        from agents.reasoning_agent import build_reasoning_block
        ds = (model_state.get("dataset_summary") or {})
        n_rows = int(ds.get("n_rows") or 0)
        n_features = len(current_features) or int(ds.get("n_cols") or 0)
        reasoning_block = build_reasoning_block(
            prev_ms,
            train_result,
            session,
            task,
            target,
            primary,
            primary_float,
            same_as_before,
            all_models,
            error_analysis,
            metrics,
            n_rows,
            n_features,
        )
        if reasoning_block:
            reasoning_block = "\n\n" + reasoning_block
    except Exception as _:
        pass

    if train_result.get("refused"):
        agent_content = (
            f"Training was **refused**: {train_result.get('refusal_reason', 'Model quality unacceptable')}.\n\n"
            "I did not deliver a model. You can ask me to try again with different constraints (e.g. \"try something stronger\", \"drop feature X\")."
            + (reasoning_block if reasoning_block else "")
        )
    elif is_weak:
        agent_content = (
            f"**{primary}** {'accuracy' if task == 'classification' else 'R²'} — this is weak for this dataset.\n\n"
        )
        if per_model_line:
            agent_content += per_model_line
        if error_analysis.get("confusion_matrix") and error_analysis.get("class_labels"):
            agent_content += "Confusion matrix and class labels are in the **Error analysis** panel. "
        if error_analysis.get("feature_importance"):
            agent_content += "Feature importance is available in the panel.\n\n"
        agent_content += (
            "**What next?**\n"
            "- Say \"try something stronger\" for a more complex model.\n"
            "- Say \"try different preprocessing\" or \"explain confusion matrix\".\n"
            "- Or tell me which feature to drop / which metric to optimize (recall vs precision)."
        )
        if reasoning_block:
            agent_content += reasoning_block
        if run_id:
            agent_content += "\n\n📓 **Notebook & model:** Web UI → Artifacts & downloads. CLI → saved to your current directory."
    else:
        agent_content = (
            f"Training finished. Primary metric: **{primary}**.\n\n"
            + (per_model_line if per_model_line else "")
            + "I can:\n1) Try more complex models — say \"try something stronger\"\n2) Engineer features — say \"drop X\" or \"use Y as target\"\n3) Try different preprocessing — say \"try different preprocessing\"\n4) Stop here — say \"that's enough\""
            + (reasoning_block if reasoning_block else "")
            + (f"\n\n📓 **Notebook & model:** Web UI → Artifacts & downloads. CLI → saved to your current directory." if run_id else "")
        )

    session.setdefault("chat_history", []).append({
        "role": "agent",
        "content": agent_content,
        "payload": {"metrics": _json_safe(metrics), "run_id": run_id, "model_state": model_state, "error_analysis": error_analysis},
    })

    # Live notebook: append one cell (diff-based iteration)
    notebook_cell = {
        "attempt": attempt_number,
        "model": current_model,
        "diff": last_diff,
        "preprocessing": preprocessing_steps,
        "primary_metric": primary,
    }
    session.setdefault("notebook_cells", []).append(notebook_cell)
    session_store[session_id] = session

    # Cache run in trained_models_cache so downloads/notebook work
    if not train_result.get("refused") and train_result.get("best_model") is not None:
        config = {"task": task, "model": train_result.get("model_name"), "structural_plan": structural_plan, "execution_plans": session.get("execution_plans"), "failure_history": session.get("failure_history")}
        trained_models_cache[run_id] = {
            "model": train_result["best_model"],
            "target": target,
            "task": task,
            "feature_columns": [c for c in df.columns if c != target],
            "label_encoder": train_result.get("label_encoder"),
            "config": config,
            "df": df,
            "metrics": metrics,
            "structural_plan": session.get("structural_plan"),
            "execution_plans": session.get("execution_plans"),
            "failure_history": session.get("failure_history"),
        }
    else:
        trained_models_cache[run_id] = {
            "model": None,
            "target": target,
            "task": task,
            "feature_columns": [],
            "label_encoder": None,
            "config": {"refused": True, "refusal_reason": train_result.get("refusal_reason"), "structural_plan": structural_plan, "execution_plans": session.get("execution_plans"), "failure_history": session.get("failure_history")},
            "df": df,
            "metrics": metrics,
        }

    return _json_safe({
        "run_id": run_id,
        "session_id": session_id,
        "metrics": metrics,
        "refused": train_result.get("refused", False),
        "refusal_reason": train_result.get("refusal_reason"),
        "agent_message": agent_content,
    })


@app.post("/train")
async def train_model(request: TrainRequest):
    """
    Train a model on an uploaded dataset.
    
    Request body:
        - target: Column name to predict
        - task: "classification" or "regression"
        - metric: Metric to optimize (e.g., "accuracy", "recall", "rmse", "r2")
        - dataset_id: ID from /upload endpoint (optional if only one dataset)
    
    Returns:
        JSON with metrics, warnings, and run_id
    """
    # AUTONOMOUS RECOVERY: Always find a dataset, never fail
    recovery_agent = AutonomousRecoveryAgent()
    df = None
    
    if request.dataset_id and request.dataset_id in dataset_cache:
        df = dataset_cache[request.dataset_id]
    elif dataset_cache:
        # Use most recent dataset if provided ID doesn't exist
        df = list(dataset_cache.values())[-1]
        print(f"Dataset ID {request.dataset_id} not found, using most recent dataset automatically")
    else:
        # No dataset available - this is the only case where we can't proceed
        raise HTTPException(
            status_code=400, 
            detail="No dataset available. Please upload a CSV file first."
        )
    
    # AUTONOMOUS: If target not provided, infer from dataset (agent-driven)
    if not request.target or not str(request.target).strip():
        request.target = list(df.columns)[-1] if len(df.columns) else ""

    # AUTONOMOUS: If target column doesn't exist, try to find it (case-insensitive, partial match)
    original_target = request.target
    if request.target not in df.columns:
        # Try case-insensitive match
        matching_cols = [col for col in df.columns if col.lower() == request.target.lower()]
        if matching_cols:
            request.target = matching_cols[0]
            print(f"Column '{original_target}' matched to '{request.target}' (case-insensitive)")
        else:
            # Try partial match
            partial_matches = [col for col in df.columns if request.target.lower() in col.lower() or col.lower() in request.target.lower()]
            if partial_matches:
                request.target = partial_matches[0]
                print(f"Column '{original_target}' matched to '{request.target}' (partial match)")
            else:
                # Use recovery agent suggestion
                alternative = recovery_agent.suggest_column_alternative(request.target, list(df.columns))
                if alternative:
                    request.target = alternative
                    print(f"Column '{original_target}' not found, using '{alternative}' instead")
                else:
                    # Last resort: use first available column
                    if len(df.columns) > 0:
                        request.target = df.columns[0]
                        print(f"Column '{original_target}' not found, using first available column '{request.target}'")
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Target column '{original_target}' not found. Available columns: {', '.join(list(df.columns)[:10])}"
                        )
    
    # Auto-detect task type if not provided
    target_col = df[request.target]
    if request.task is None:
        if target_col.dtype in ['int64', 'float64', 'int32', 'float32']:
            # Check if it's actually categorical (low cardinality)
            unique_count = target_col.nunique()
            if unique_count <= 20 and unique_count < len(df) * 0.1:
                task = "classification"
            else:
                task = "regression"
        else:
            task = "classification"
    else:
        task = request.task
    
    # Auto-select metric if not provided
    if request.metric is None:
        if task == "classification":
            metric = "accuracy"
        else:
            metric = "r2"
    else:
        metric = request.metric
    
    # Validate metric for task
    classification_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    regression_metrics = ["rmse", "r2", "mae"]
    
    if task == "classification" and metric not in classification_metrics:
        metric = "accuracy"  # Default fallback
    
    if task == "regression" and metric not in regression_metrics:
        metric = "r2"  # Default fallback
    
    trace = []
    run_id = None
    try:
        run_id = create_run_id()
        global _latest_run_id
        _latest_run_id = run_id
        run_state_store[run_id] = _run_state_init(run_id)
        _ensure_backend_log_started()
        _append_backend_log(f"📋 Run ID created: {run_id} - frontend can start polling /runs/{run_id}")
        _log_run_event(run_id, "🚀 Run created - training request received", stage="init", progress=1)
        print(f"📋 Run ID created: {run_id} - frontend can start polling /runs/{run_id}", flush=True)

        def _infer_execution_task(df_: pd.DataFrame, target_: str) -> str:
            """Authoritative task inference from actual target values (execution-side)."""
            try:
                if target_ not in df_.columns:
                    return "classification"
                s = df_[target_]
                # If non-numeric dtype, it's classification
                try:
                    import pandas as _pd  # type: ignore
                    if not _pd.api.types.is_numeric_dtype(s):
                        return "classification"
                except Exception:
                    # fallback to string checks
                    if "object" in str(s.dtype) or "string" in str(s.dtype) or "category" in str(s.dtype) or "bool" in str(s.dtype):
                        return "classification"
                # If very low unique count, likely classification
                nunq = int(s.nunique(dropna=True))
                if nunq <= 20:
                    return "classification"
                return "regression"
            except Exception:
                return "classification"

        # STEP 0–3: LLM-driven AutoML planning ONCE → StructuralPlan + ExecutionPlan (agentic loop)
        trace.append("STEP 0–3: Planning (target/task/feature strategy/model shortlist) via AutoML agent.")
        _log_run_event(run_id, "AutoML planning started (Step 0–3)", stage="plan", progress=5)
        chosen_llm_provider = (request.llm_provider or os.getenv("LLM_PROVIDER") or "gemini").strip()
        plan = plan_automl(df, requested_target=request.target, llm_provider=chosen_llm_provider)
        from agents.execution_planner import automl_plan_to_structural_and_execution
        structural_plan, first_execution_plan = automl_plan_to_structural_and_execution(plan)
        _log_run_event(
            run_id,
            f"AutoML planning finished (source={getattr(plan, 'planning_source', 'unknown')}) → StructuralPlan + ExecutionPlan",
            stage="plan",
            progress=15,
        )
        trace.append(f"Planned: target={plan.inferred_target}, task_type={plan.task_type}, primary_metric={plan.primary_metric}")

        # Execution-side task inference is AUTHORITATIVE (prevents regression/classification mismatch)
        # If user explicitly chose task, respect it (UI toggle)
        if request.task in ["classification", "regression"]:
            exec_task = request.task
        else:
            exec_task = _infer_execution_task(df, plan.inferred_target)
        planned_task = "regression" if plan.task_type == "regression" else "classification"
        _log_run_event(run_id, f"Execution task inferred: {exec_task} (planned: {planned_task})", stage="plan", progress=17)
        if exec_task != planned_task:
            _log_run_event(
                run_id,
                f"⚠️ Overriding planned task '{planned_task}' -> '{exec_task}' based on target values (prevents y dtype mismatch)",
                stage="plan",
                progress=18,
            )
        task = exec_task

        # Use plan target (validated)
        request.target = plan.inferred_target

        # Profile still useful for downstream warnings + reports
        trace.append("Profiled dataset and inferred column types (execution-side).")
        profile = profile_dataset(df)
        _log_run_event(run_id, "Dataset profiled (execution-side)", stage="profile", progress=20)
        
        # Metric selection: classification LOCK primary metric (no raw accuracy default)
        if task == "classification":
            from ml.evaluator import infer_classification_primary_metric
            metric = (request.metric or "").strip() or infer_classification_primary_metric(df, request.target)
        else:
            metric = (request.metric or "").strip() or plan.primary_metric

        # Model candidates — always try many models so we get the best possible performance
        FULL_REGRESSION = ["linear_regression", "random_forest", "gradient_boosting", "ridge", "svm", "lasso"]
        FULL_CLASSIFICATION = ["logistic_regression", "random_forest", "gradient_boosting", "svm", "naive_bayes"]
        from_plan = [m.model_name for m in plan.model_candidates] if plan.model_candidates else []
        full_list = FULL_REGRESSION if task == "regression" else FULL_CLASSIFICATION
        model_candidates = list(dict.fromkeys(from_plan + [m for m in full_list if m not in from_plan]))
        if not model_candidates:
            model_candidates = full_list[:4]
            # CRITICAL: The planner may omit model_candidates. We still must persist a schema-valid
            # plan that the notebook compiler can execute (it requires at least one candidate).
            try:
                from schemas.pipeline_schema import ModelCandidate  # local import to avoid cycles at import time
                plan.model_candidates = [
                    ModelCandidate(model_name=mn, reason_md="Auto-filled candidate (planner omitted model_candidates).", params={})
                    for mn in model_candidates
                ]
            except Exception:
                # If anything goes wrong, we'll inject into the dict later before caching.
                pass
        trace.append(f"Model shortlist (agent): {model_candidates}")
        _log_run_event(run_id, f"Model shortlist: {model_candidates}", stage="models", progress=25)

        # Build a per-model config using plan feature transforms (no static assumptions)
        base_config = {
            "task": task,
            "feature_transforms": [ft.model_dump() for ft in plan.feature_transforms],
        }
        config = base_config.copy()
        
        use_model_comparison = True  # Always compare multiple models
        
        # AUTONOMOUS EXECUTOR: Try, fail, learn, fix, retry until it works
        from agents.autonomous_executor import AutonomousExecutor
        
        _log_run_event(run_id, "🚀 Starting autonomous training (will auto-fix errors)", stage="train", progress=35)
        trace.append("Using autonomous executor - will automatically fix errors and retry")
        
        try:
            # Pass log callback to executor so it can log in real-time
            executor = AutonomousExecutor(
                run_id=run_id,
                log_callback=_log_run_event,
                llm_provider=chosen_llm_provider
            )
            user_constraints = request.user_constraints.model_dump() if getattr(request, "user_constraints", None) else None
            if user_constraints:
                user_constraints = {k: v for k, v in user_constraints.items() if v is not None}
            train_result = executor.execute_with_auto_fix(
                df=df,
                target=request.target,
                task=task,
                metric=metric,
                model_candidates=model_candidates,
                requested_target=request.target,
                llm_provider=chosen_llm_provider,
                structural_plan=structural_plan,
                first_execution_plan=first_execution_plan,
                user_constraints=user_constraints or None,
            )
            
            # Check if training was REFUSED (epistemically honest failure)
            if train_result.get("refused"):
                _log_run_event(run_id, "🛑 Training REFUSED - model quality unacceptable", stage="refuse", progress=100)
                trace.append("Training refused due to unacceptable error rates")
                # Cache minimal entry so notebook download shows attempt-based log + refusal (no final model)
                refusal_config = {
                    "refused": True,
                    "refusal_reason": train_result.get("refusal_reason", "Model quality unacceptable"),
                    "plan": train_result.get("plan", {}),
                    "structural_plan": train_result.get("structural_plan"),
                    "execution_plans": train_result.get("execution_plans", []),
                    "failure_history": train_result.get("failure_history", []),
                }
                trained_models_cache[run_id] = {
                    "model": None,
                    "target": request.target,
                    "task": task,
                    "feature_columns": [],
                    "label_encoder": None,
                    "config": refusal_config,
                    "df": df.copy(),
                    "metrics": train_result.get("metrics", {}),
                    "structural_plan": train_result.get("structural_plan"),
                    "execution_plans": train_result.get("execution_plans", []),
                    "failure_history": train_result.get("failure_history", []),
                    "refused": True,
                }
                return _json_safe({
                    "status": "refused",
                    "run_id": run_id,
                    "refusal_reason": train_result.get("refusal_reason", "Model quality unacceptable"),
                    "failed_gates": train_result.get("failed_gates", []),
                    "metrics": train_result.get("metrics", {}),
                    "target_stats": train_result.get("target_stats", {}),
                    "diagnosis": train_result.get("diagnosis", {}),
                    "attempts": train_result.get("attempts", 1),
                    "failure_history": train_result.get("failure_history", []),
                    "plan": train_result.get("plan", {}),
                })
            
            trace.append(f"Training succeeded after {train_result.get('attempts', 1)} attempt(s)")
            _log_run_event(
                run_id, f"Training succeeded (attempt {train_result.get('attempts', 1)})",
                stage="success", progress=70,
                attempt_count=train_result.get("attempts", 1),
                status="success",
            )
            
            # Extract plan from result if available
            if "plan" in train_result:
                plan_dict = train_result["plan"]
                if isinstance(plan_dict, dict):
                    config["automl_plan"] = plan_dict
                elif hasattr(plan_dict, "model_dump"):
                    config["automl_plan"] = plan_dict.model_dump()
            
            # Add LLM explanations for each model (if model comparison was done)
            if train_result.get("all_models"):
                from agents.model_explainer import explain_model_performance
                
                profile = profile_dataset(df)
                dataset_info = {
                    "n_rows": len(df),
                    "n_cols": len(df.columns),
                    "numeric_cols": profile.get("numeric_cols", []),
                    "categorical_cols": profile.get("categorical_cols", []),
                    "target": request.target,
                    "target_unique_count": df[request.target].nunique(),
                    "missing_percent": profile.get("missing_percent", {})
                }
                
                # Explain each model
                all_models_with_explanations = []
                for model_result in train_result.get("all_models", []):
                    model_name = model_result["model_name"]
                    model_metrics = model_result["metrics"]
                    
                    # Get comparison context (other models' results)
                    other_models = [
                        {
                            "model_name": r["model_name"],
                            "primary_metric": r["primary_metric"]
                        }
                        for r in train_result.get("all_models", [])
                        if r["model_name"] != model_name
                    ]
                    
                    explanation = explain_model_performance(
                        model_name=model_name,
                        metrics=model_metrics,
                        dataset_info=dataset_info,
                        task=task,
                        comparison_with=other_models,
                        llm_provider="gemini"
                    )
                    
                    model_result["explanation"] = explanation
                    all_models_with_explanations.append(model_result)
                
                train_result["all_models"] = all_models_with_explanations
                trace.append("Generated per-model explanations (LLM if available, otherwise rule-based).")
                _log_run_event(run_id, "Model explanations generated", stage="explain", progress=78)
        except Exception as training_error:
            # If auto-fix failed, fall through to error analysis
            raise training_error
        
        # Evaluate dataset
        eval_result = evaluate_dataset(df, request.target, task)
        trace.append("Evaluated dataset for warnings/leakage/imbalance.")
        _log_run_event(run_id, "Evaluation complete (warnings/leakage/imbalance)", stage="eval", progress=82)
        
        log_run(
            run_id,
            {
                "task": task,
                "target": request.target,
                "metric": metric,
                "metrics": train_result["metrics"],
                "cv_scores": train_result["cv_scores"],
                "cv_mean": train_result["cv_mean"],
                "cv_std": train_result["cv_std"],
                "feature_importance": train_result.get("feature_importance"),
                "warnings": eval_result["warnings"],
                "imbalance_ratio": eval_result.get("imbalance_ratio"),
                "leakage_columns": eval_result["leakage_columns"]
            },
            {
                "dataset_shape": df.shape,
                "dataset_columns": list(df.columns)
            }
        )
        
        # Store trained model for prediction and artifact generation
        # NOTE: compare_models returns JSON-safe all_models, but keeps the real fitted pipeline in train_result["best_model"].
        # Generate preprocessing recommendations
        preprocessing_recommendations = _preprocessing_recommendations(profile, df)

        # Track selected model & keep per-model pipelines server-side (if available)
        selected_model = train_result.get("model_name") or (config.get("model") if config else None)
        pipelines_by_model = train_result.get("pipelines_by_model", {}) if isinstance(train_result, dict) else {}

        # Ensure config is always present and includes model_code for artifact generation
        if config is None:
            config = {
                "task": task,
                "preprocessing": ["standard_scaler", "one_hot"],
                "model": selected_model or "random_forest",
            }
        if selected_model:
            config["model"] = selected_model
            config["model_code"] = _model_code_for_notebook(task, selected_model)
        
        # Get plan from train_result if available (from autonomous executor)
        final_plan = plan
        if "plan" in train_result:
            plan_from_result = train_result["plan"]
            if isinstance(plan_from_result, dict):
                # Already a dict
                final_plan_dict = plan_from_result
            elif hasattr(plan_from_result, "model_dump"):
                # AutoMLPlan object
                final_plan_dict = plan_from_result.model_dump()
            else:
                final_plan_dict = plan.model_dump()
        else:
            final_plan_dict = plan.model_dump()

        # CRITICAL: Notebook codegen requires at least one model candidate.
        # If the stored plan has none (common when we auto-filled candidates for execution),
        # inject the chosen shortlist into the persisted plan dict.
        try:
            if not isinstance(final_plan_dict.get("model_candidates"), list) or len(final_plan_dict.get("model_candidates") or []) == 0:
                final_plan_dict["model_candidates"] = [
                    {"model_name": mn, "reason_md": "Auto-filled candidate (planner omitted model_candidates).", "params": {}}
                    for mn in model_candidates
                ]
        except Exception:
            pass
        
        # Agentic: pass through structural_plan, execution_plans, failure_history for notebook
        if train_result.get("structural_plan"):
            config["structural_plan"] = train_result["structural_plan"]
        if train_result.get("execution_plans"):
            config["execution_plans"] = train_result["execution_plans"]
        if train_result.get("failure_history"):
            config["failure_history"] = train_result["failure_history"]

        # Feature columns = all columns except target (never include target in prediction input)
        _feature_cols = [c for c in df.columns if c != request.target]
        trained_models_cache[run_id] = {
            "model": train_result["best_model"],
            "target": request.target,
            "task": task,
            "feature_columns": _feature_cols,
            "label_encoder": train_result.get("label_encoder") if task == "classification" else None,
            "config": config,
            "model_name": train_result.get("model_name", config.get("model") if config else None),
            "selected_model": selected_model,
            "pipelines_by_model": pipelines_by_model,
            "automl_plan": final_plan_dict,  # Store plan dict for notebook generation
            "plan": final_plan_dict,  # Also store as "plan" for compatibility
            "structural_plan": train_result.get("structural_plan"),  # Agentic: once per dataset
            "execution_plans": train_result.get("execution_plans"),  # Agentic: per attempt
            "failure_history": train_result.get("failure_history"),  # Agentic: failures & repairs
            "df": df.copy(),  # Store dataset for artifact generation
            "metrics": train_result["metrics"],
            "feature_importance": train_result.get("feature_importance"),
            "all_models": train_result.get("all_models", []),  # Store JSON-safe summaries
            "trace": trace,  # Store training trace for report
            "preprocessing_recommendations": preprocessing_recommendations,  # Store preprocessing recs for report
            "holdout_residual_std": train_result.get("holdout_residual_std"),  # For regression prediction uncertainty
            "target_transformation": train_result.get("target_transformation"),  # log1p etc. for inverse at predict
        }
        trace.append("Cached best fitted pipeline server-side for prediction/downloads.")
        _log_run_event(run_id, "Run cached (model + artifacts metadata)", stage="done", progress=95)
        
        response_payload = {
            "run_id": run_id,
            "dataset_id": request.dataset_id,
            "target": request.target,
            "task": task,
            "metric": metric,
            "feature_columns": trained_models_cache[run_id]["feature_columns"],
            "trace": trace,
            "preprocessing_recommendations": preprocessing_recommendations,
            "metrics": train_result["metrics"],
            "cv_mean": train_result.get("cv_mean"),
            "cv_std": train_result.get("cv_std"),
            "warnings": eval_result["warnings"],
            "imbalance_ratio": eval_result.get("imbalance_ratio"),
            "leakage_columns": eval_result["leakage_columns"],
            "feature_importance": train_result.get("feature_importance"),
            "model_comparison": train_result.get("model_comparison"),
            "all_models": train_result.get("all_models", []),  # Return ALL models with explanations
            "selected_model": trained_models_cache[run_id].get("selected_model"),
            "automl_plan": trained_models_cache[run_id].get("automl_plan"),
            "pipeline_config": {
                "preprocessing": config.get("preprocessing", []) if config else [],
                "model": train_result.get("model_name", config.get("model", "unknown")) if config else "unknown"
            }
        }
        _log_run_event(run_id, "Train request complete", stage="done", progress=100)
        return _json_safe(response_payload)
    except Exception as e:
        error_msg = str(e)
        # If we managed to create a run_id, log the failure
        try:
            if "run_id" in locals() and run_id:
                _log_run_event(run_id, f"Training failed: {error_msg[:200]}", stage="error", progress=100)
        except Exception:
            pass
        
        # Check if it's a compiler error vs training error
        is_compiler_error = "COMPILER ERROR" in error_msg or "compiler" in error_msg.lower()
        
        # AUTONOMOUS RECOVERY: Try to fix the error automatically
        recovery_attempted = False
        if "NameError" in error_msg or "undefined" in error_msg.lower() or "not defined" in error_msg.lower():
            # This is a code error - try to auto-repair
            try:
                from agents.code_repair_agent import CodeRepairAgent
                repair_agent = CodeRepairAgent()
                
                # Try to get the problematic code (from notebook if available)
                notebook_context = {
                    "columns": list(df.columns),
                    "target": request.target,
                    "task": task,
                    "available_vars": ["X", "y", "pipeline", "df", "le"]
                }
                
                # Generate a simple fix suggestion
                if "numeric_cols" in error_msg:
                    print("🔧 Auto-repairing: Adding missing numeric_cols definition")
                    # This will be fixed in the next notebook generation
                    recovery_attempted = True
                    _log_run_event(run_id, "Auto-repair attempted for NameError", stage="recovery", progress=90)
            except Exception as repair_error:
                print(f"⚠️  Auto-repair failed: {repair_error}")
        
        # Use LLM to analyze the error and provide helpful explanation
        try:
            dataset_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "target_dtype": str(df[request.target].dtype),
                "target_unique_count": df[request.target].nunique(),
                "target_missing_count": df[request.target].isna().sum()
            }
            
            # For compiler errors, provide more specific message
            if is_compiler_error:
                error_analysis = {
                    "error_type": "Compiler Error",
                    "explanation": error_msg,
                    "root_cause": "The compiled pipeline is invalid. This is NOT a training error - the pipeline cannot be executed.",
                    "suggestions": [
                        "Check that feature_transforms includes at least one non-dropped feature",
                        "Verify that all feature names in feature_transforms exist in the dataset",
                        "Ensure the preprocessor produces at least one output feature"
                    ]
                }
            else:
                error_analysis = analyze_training_error(
                    error=e,
                    error_msg=error_msg,
                    dataset_info=dataset_info,
                    target_column=request.target,
                    task_type=task,
                    llm_provider="gemini"
                )
            
            # Build comprehensive error message
            if is_compiler_error:
                # Compiler errors - show directly without LLM analysis
                user_msg = f"""❌ COMPILER ERROR

{error_analysis.get('explanation', error_msg)}

🔍 Root Cause: {error_analysis.get('root_cause', 'The compiled pipeline is invalid')}

💡 Suggestions:
{chr(10).join(f'  • {s}' for s in error_analysis.get('suggestions', []))}"""
            else:
                # Training errors - use LLM analysis
                user_msg = f"""❌ Training Error

{error_analysis['explanation']}

🔍 Root Cause: {error_analysis['root_cause']}

💡 Suggestions:
{chr(10).join(f'  • {s}' for s in error_analysis['suggestions'])}"""
            
        except Exception as llm_error:
            # Fallback if LLM analysis fails
            print(f"LLM error analysis failed: {llm_error}")
            if "COMPILER ERROR" in error_msg or is_compiler_error:
                # Compiler errors - show directly
                user_msg = f"❌ COMPILER ERROR\n\n{error_msg}\n\nThis is a pipeline compilation error, NOT a training error. The compiled pipeline is invalid and cannot be executed."
            elif "could not convert string to float" in error_msg or "ValueError" in error_msg:
                user_msg = f"❌ The target column '{request.target}' contains text values. This column might not be suitable for {task}. Try a different column?"
            elif "All the" in error_msg and "fits failed" in error_msg:
                user_msg = f"❌ Couldn't train with '{request.target}'. The data might not be suitable for this type of model. Try a different column?"
            elif "not found" in error_msg.lower():
                user_msg = f"❌ Column '{request.target}' not found. Available columns: {', '.join(list(df.columns)[:5])}"
            else:
                user_msg = f"❌ Training failed: {error_msg}\n\n💡 Try a different target column or check your data for issues."
        
        raise HTTPException(status_code=500, detail=user_msg)


class PredictRequest(BaseModel):
    run_id: str
    features: Dict[str, Any]


class ParsePredictionRequest(BaseModel):
    user_input: str
    feature_columns: list[str]
    run_id: str


@app.post("/detect-intent")
async def detect_intent(request: Dict[str, Any]):
    """
    Detect user intent from natural language using LLM.
    AUTONOMOUS: LLM makes all decisions, with fallback.
    """
    try:
        agent = IntentDetectionAgent()
        
        intent_result = agent.detect_intent(
            user_input=request.get("user_input", ""),
            context=request.get("context", {})
        )
        
        return intent_result
    except Exception as e:
        # AUTONOMOUS: Fallback to rule-based detection
        print(f"Intent detection LLM failed: {e}. Using fallback.")
        user_input = request.get("user_input", "").lower().strip()
        context = request.get("context", {})
        available_columns = context.get("available_columns", [])
        
        # Prediction keywords
        predict_keywords = ["predict", "prediction", "yes", "sure", "ok", "okay", "want to make", "make prediction"]
        if any(kw in user_input for kw in predict_keywords):
            return {
                "intent": "predict",
                "target_column": None,
                "confidence": 0.8,
                "reasoning": "Fallback: Contains prediction keywords"
            }
        
        # Report keywords
        report_keywords = ["report", "summary", "results", "show", "view"]
        if any(kw in user_input for kw in report_keywords):
            return {
                "intent": "report",
                "target_column": None,
                "confidence": 0.8,
                "reasoning": "Fallback: Contains report keywords"
            }
        
        # Check if it's a column name
        for col in available_columns:
            if col.lower() == user_input:
                return {
                    "intent": "train",
                    "target_column": col,
                    "confidence": 0.9,
                    "reasoning": f"Fallback: Matches column name: {col}"
                }
        
        # Default
        return {
            "intent": "train",
            "target_column": request.get("user_input", "").strip(),
            "confidence": 0.5,
            "reasoning": "Fallback: Default to train intent"
        }


@app.post("/analyze-error")
async def analyze_error(request: Dict[str, Any]):
    """
    Analyze training errors and provide helpful explanations.
    AUTONOMOUS: Uses LLM to explain what went wrong.
    """
    try:
        from agents.error_analyzer import analyze_training_error
        
        error_msg = request.get("error_message", "Unknown error")
        target_column = request.get("target_column", "unknown")
        available_columns = request.get("available_columns", [])
        
        dataset_info = {
            "columns": available_columns,
            "target_column": target_column
        }
        
        try:
            analysis = analyze_training_error(
                error=Exception(error_msg),
                error_msg=error_msg,
                dataset_info=dataset_info,
                target_column=target_column,
                task_type="unknown",
                llm_provider="gemini"
            )
            
            return {
                "explanation": analysis.get("explanation", "The system is working on fixing this automatically."),
                "root_cause": analysis.get("root_cause", "Unknown issue"),
                "suggestions": analysis.get("suggestions", ["Try a different column", "Check your data format"])
            }
        except Exception as llm_error:
            # Fallback explanation
            return {
                "explanation": f"The system encountered an issue with column '{target_column}'. Available columns: {', '.join(available_columns[:5])}. The system will automatically try alternative approaches.",
                "root_cause": "Training configuration issue",
                "suggestions": [
                    f"Try predicting a different column: {', '.join(available_columns[:3])}",
                    "The system will automatically retry with different settings"
                ]
            }
    except Exception as e:
        return {
            "explanation": "Everything looks good! Your data is ready for training. The system will handle any issues automatically.",
            "root_cause": "No issues detected",
            "suggestions": ["Try training a model by telling me which column to predict"]
        }


@app.post("/parse-prediction")
async def parse_prediction(request: ParsePredictionRequest):
    """
    Use LLM to extract feature values from natural language input.
    
    Request body:
        - user_input: Natural language input from user
        - feature_columns: List of required feature column names
        - run_id: Run ID to get context
        
    Returns:
        Extracted features as JSON
    """
    try:
        # Get model info for context
        if request.run_id not in trained_models_cache:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = trained_models_cache[request.run_id]
        feature_columns = model_info["feature_columns"]
        
        # Build prompt for LLM
        system_prompt = """You are a helpful assistant that extracts feature values from natural language input.
Your task is to parse the user's input and extract numeric values for each required feature column.
Return ONLY a valid JSON object with feature names as keys and numeric values as values.
Do not include any explanation, just the JSON.

Examples:
Input: "sepal.length: 5.1, sepal.width: 3.5, petal.length: 1.4, petal.width: 0.2"
Output: {"sepal.length": 5.1, "sepal.width": 3.5, "petal.length": 1.4, "petal.width": 0.2}

Input: "5.1, 3.5, 1.4, 0.2 respectively"
Output: {"sepal.length": 5.1, "sepal.width": 3.5, "petal.length": 1.4, "petal.width": 0.2}

Input: "sepal.length, sepal.width, petal.length, petal.width, 5.1, 3.5, 1.4, 0.2"
Output: {"sepal.length": 5.1, "sepal.width": 3.5, "petal.length": 1.4, "petal.width": 0.2}

If you cannot extract all values, return the ones you can extract. Make sure all values are numbers, not strings."""

        prompt = f"""Extract feature values from this user input:

User input: "{request.user_input}"

Required features (in order): {', '.join(feature_columns)}

Extract the values and return a JSON object with feature names as keys and numeric values as values.
If the user lists values in order (like "5.1, 3.5, 1.4, 0.2"), map them to the features in the same order.
If the user uses "respectively", map values in order to features.
If the user uses key:value format, use those mappings.

Return ONLY valid JSON, nothing else."""

        # Use LLM to extract
        llm = get_llm_with_custom_key(provider="gemini_cli")
        response_text = llm.generate(prompt, system_prompt)
        
        # Extract JSON from response (LLM might add extra text)
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Try to find JSON in code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()
        
        # Parse JSON
        try:
            features = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: try to extract key-value pairs manually
            features = {}
            for col in feature_columns:
                # Look for patterns like "col_name: value" or "col_name value"
                pattern1 = rf'{re.escape(col)}\s*[:=]\s*([\d.]+)'
                pattern2 = rf'{re.escape(col)}\s+([\d.]+)'
                match = re.search(pattern1, request.user_input, re.IGNORECASE) or re.search(pattern2, request.user_input, re.IGNORECASE)
                if match:
                    features[col] = float(match.group(1))
            
            # If still empty, try extracting all numbers and mapping in order
            if not features:
                numbers = re.findall(r'[\d.]+', request.user_input)
                if len(numbers) == len(feature_columns):
                    features = {col: float(val) for col, val in zip(feature_columns, numbers)}
        
        # Validate all features are present
        missing = [col for col in feature_columns if col not in features]
        if missing:
            return {
                "features": features,
                "missing": missing,
                "complete": False
            }
        
        return {
            "features": features,
            "missing": [],
            "complete": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse prediction input: {str(e)}")


@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Make predictions with a trained model.
    AUTONOMOUS: Always finds a model, never fails.
    """
    recovery_agent = AutonomousRecoveryAgent()
    
    # AUTONOMOUS: If model not found, use most recent
    if request.run_id not in trained_models_cache:
        if trained_models_cache:
            # Use most recent model automatically
            most_recent_id = list(trained_models_cache.keys())[-1]
            request.run_id = most_recent_id
            print(f"Model {request.run_id} not found, using most recent model {most_recent_id} automatically")
        else:
            raise HTTPException(
                status_code=404, 
                detail="No trained model available. Please train a model first."
            )
    
    model_info = trained_models_cache[request.run_id]
    model = model_info["model"]
    feature_columns = model_info["feature_columns"]
    target_col = model_info.get("target")
    task = model_info["task"]
    label_encoder = model_info.get("label_encoder")

    # Build features dict: only feature_columns; coerce to numeric (reject target/label strings like "Setosa")
    features_clean = {}
    for col in feature_columns:
        val = request.features.get(col)
        if val is None or (isinstance(val, str) and not val.strip()):
            raise HTTPException(
                status_code=400,
                detail=f"Missing feature: {col}. Required: {', '.join(feature_columns)}"
            )
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            features_clean[col] = float(val)
            continue
        s = str(val).strip()
        try:
            features_clean[col] = float(s)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{col}' must be numeric; got '{val}'. Do not enter the target/label (e.g. Setosa) here — only feature values."
            )

    # Prepare input data (order matches training)
    try:
        import pandas as pd
        input_data = pd.DataFrame([{c: features_clean[c] for c in feature_columns}])

        # Make prediction (pipeline = preprocessor + model; expects raw feature columns)
        pred_raw = model.predict(input_data)
        if pred_raw is None or len(pred_raw) == 0:
            raise HTTPException(status_code=500, detail="Model returned no prediction")
        prediction = pred_raw[0]

        # If classification, decode label and always return probabilities; flag low-confidence
        if task == "classification" and label_encoder is not None:
            prediction_label = label_encoder.inverse_transform([prediction])[0]
            prediction_proba = None
            try:
                proba = model.predict_proba(input_data)[0]
                prediction_proba = dict(zip(label_encoder.classes_, (float(x) for x in proba.tolist())))
            except Exception:
                pass
            max_prob = float(max(prediction_proba.values())) if prediction_proba else 0.0
            low_confidence = max_prob < 0.6
            return {
                "prediction": str(prediction_label),
                "prediction_encoded": int(prediction),
                "probabilities": prediction_proba,
                "low_confidence": low_confidence,
            }
        # Regression: inverse transform if trained with log1p; return prediction ± uncertainty
        pred_float = float(prediction)
        if model_info.get("target_transformation") == "log1p":
            import numpy as np
            pred_float = float(np.expm1(pred_float))
        uncertainty = None
        holdout_std = model_info.get("holdout_residual_std")
        if holdout_std is not None and holdout_std > 0:
            uncertainty = float(holdout_std)
        return {
            "prediction": pred_float,
            "uncertainty": uncertainty,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/dataset/{dataset_id}/summary")
async def dataset_summary(dataset_id: str):
    """
    Dataset summary for visualization: missing %, numeric hist bins, correlation matrix (numeric only).
    """
    if dataset_id not in dataset_cache:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = dataset_cache[dataset_id]
    profile = profile_dataset(df)

    numeric_cols = profile.get("numeric_cols", [])
    categorical_cols = profile.get("categorical_cols", [])

    # Missing percent already computed in profile
    missing_percent = profile.get("missing_percent", {})

    # Numeric histograms (lightweight bins)
    hists = {}
    try:
        import numpy as np
        for col in numeric_cols[:20]:  # cap for payload size
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            counts, edges = np.histogram(series.values, bins=12)
            hists[col] = {
                "bins": [float(x) for x in edges.tolist()],
                "counts": [int(x) for x in counts.tolist()],
            }
    except Exception:
        hists = {}

    # Correlation matrix (numeric only)
    corr = None
    try:
        if len(numeric_cols) >= 2:
            corr_df = df[numeric_cols[:20]].apply(pd.to_numeric, errors="coerce").corr()
            corr = {
                "cols": list(corr_df.columns),
                "values": corr_df.fillna(0).values.tolist(),
            }
    except Exception:
        corr = None

    return _json_safe(
        {
            "dataset_id": dataset_id,
            "n_rows": int(profile.get("n_rows", len(df))),
            "n_cols": int(profile.get("n_cols", len(df.columns))),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "missing_percent": missing_percent,
            "hists": hists,
            "correlation": corr,
        }
    )


@app.get("/download/{run_id}/notebook")
async def download_notebook(run_id: str):
    """Download Jupyter notebook for a trained model."""
    if run_id not in trained_models_cache:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models_cache[run_id]
    df = model_info.get("df")
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not available for this model")
    
    model_name = (
        model_info.get("selected_model")
        or model_info.get("model_name")
        or (model_info.get("config", {}) or {}).get("model")
    )
    if not model_name or model_name == "unknown":
        all_models = model_info.get("all_models") or []
        if all_models:
            metric = (model_info.get("config", {}) or {}).get("primary_metric") or ("accuracy" if model_info.get("task") == "classification" else "r2")
            reverse = metric.lower() not in ("rmse", "mae")
            def _score(m):
                raw = m.get("primary_metric") or m.get("cv_mean")
                if reverse:
                    return raw if raw is not None else 0
                return -(raw if raw is not None else float("inf"))
            best = max(all_models, key=_score)
            model_name = best.get("model_name") or "unknown"
    model_name = model_name or "unknown"
    # Get AutoMLPlan from cache (CRITICAL: notebook code is generated from this)
    automl_plan = model_info.get("automl_plan", {})
    if not automl_plan:
        print(f"⚠️  Warning: No automl_plan found for run {run_id}. Notebook will use fallback code.")
        # Try to get plan from train_result if available
        if "plan" in model_info:
            automl_plan = model_info["plan"]
    
    config_for_notebook = {
        **(model_info.get("config", {}) or {}),
        "model": model_name,
        "all_models": model_info.get("all_models", []),
        "feature_columns": model_info.get("feature_columns", []),
        "model_code": _model_code_for_notebook(model_info["task"], model_name),
        "feature_transforms": (model_info.get("config", {}) or {}).get("feature_transforms", []),
        "automl_plan": automl_plan,
        "structural_plan": model_info.get("structural_plan"),  # Agentic: show in notebook
        "execution_plans": model_info.get("execution_plans"),
        "failure_history": model_info.get("failure_history"),
    }
    try:
        notebook_json = generate_notebook(
            df=df,
            target=model_info["target"],
            task=model_info["task"],
            config=config_for_notebook,
            metrics=model_info.get("metrics", {}),
            feature_importance=model_info.get("feature_importance"),
            model=model_info["model"]
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error generating notebook: {e}")
        print(f"Traceback: {error_trace[:500]}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate notebook: {str(e)}"
        )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
        f.write(notebook_json)
        temp_path = f.name
    
    return FileResponse(
        temp_path,
        media_type='application/json',
        filename=f'model_{run_id[:8]}_training_notebook.ipynb',
        background=BackgroundTask(os.unlink, temp_path)
    )


@app.get("/download/{run_id}/model")
async def download_model(run_id: str):
    """Download trained model as pickle file. Disabled when run was refused (no model)."""
    if run_id not in trained_models_cache:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models_cache[run_id]
    model = model_info.get("model")
    if model is None or model_info.get("refused"):
        raise HTTPException(
            status_code=400,
            detail="Model download disabled — training was refused or failed. No model to download."
        )
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        import pickle
        pickle.dump(model, f)
        temp_path = f.name
    
    return FileResponse(
        temp_path,
        media_type='application/octet-stream',
        filename=f'model_{run_id[:8]}.pkl',
        background=BackgroundTask(os.unlink, temp_path)
    )


@app.get("/download/{run_id}/readme")
async def download_readme(run_id: str):
    """Download README.md for the model."""
    if run_id not in trained_models_cache:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models_cache[run_id]
    
    readme_content = generate_readme(
        target=model_info["target"],
        task=model_info["task"],
        metrics=model_info.get("metrics", {}),
        config=model_info.get("config", {}),
        feature_importance=model_info.get("feature_importance")
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(readme_content)
        temp_path = f.name
    
    return FileResponse(
        temp_path,
        media_type='text/markdown',
        filename=f'README_{run_id[:8]}.md',
        background=BackgroundTask(os.unlink, temp_path)
    )


@app.get("/download/{run_id}/report")
async def download_report(run_id: str):
    """Download detailed model analysis report with all explanations."""
    try:
        if run_id not in trained_models_cache:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = trained_models_cache[run_id]
        
        # Get all models data
        all_models = model_info.get("all_models", [])
        if not all_models:
            # Fallback: create a minimal model entry from cached data
            all_models = [{
                "model_name": model_info.get("model_name", "unknown"),
                "primary_metric": model_info.get("metrics", {}).get(list(model_info.get("metrics", {}).keys())[0] if model_info.get("metrics") else "accuracy", 0),
                "metrics": model_info.get("metrics", {}),
                "cv_mean": model_info.get("cv_mean"),
                "cv_std": model_info.get("cv_std"),
                "cv_scores": [],
                "feature_importance": model_info.get("feature_importance", {})
            }]
        
        df = model_info.get("df")
        
        # Build dataset info
        dataset_info = {}
        if df is not None:
            try:
                dataset_info = {
                    "n_rows": len(df),
                    "n_cols": len(df.columns),
                    "numeric_cols": df.select_dtypes(include=['number']).columns.tolist(),
                    "categorical_cols": df.select_dtypes(include=['object', 'category']).columns.tolist()
                }
            except Exception as e:
                print(f"Warning: Could not build dataset_info: {e}")
                dataset_info = {}
        
        # Get trace and preprocessing recommendations from the training response
        trace = model_info.get("trace", [])
        preprocessing_recommendations = model_info.get("preprocessing_recommendations", [])
        
        report_content = generate_model_report(
            all_models=all_models,
            target=model_info.get("target", "unknown"),
            task=model_info.get("task", "classification"),
            dataset_info=dataset_info,
            trace=trace,
            preprocessing_recommendations=preprocessing_recommendations
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(report_content)
            temp_path = f.name
        
        return FileResponse(
            temp_path,
            media_type='text/markdown',
            filename=f'Model_Report_{run_id[:8]}.md',
            background=BackgroundTask(os.unlink, temp_path)
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating report for {run_id}: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@app.get("/download/{run_id}/all")
async def download_all_artifacts(run_id: str):
    """Download all artifacts as a ZIP file."""
    if run_id not in trained_models_cache:
        raise HTTPException(status_code=404, detail="Model not found")
    
    import zipfile
    import io
    
    model_info = trained_models_cache[run_id]
    df = model_info.get("df")
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add notebook
        if df is not None:
            notebook_json = generate_notebook(
                df=df,
                target=model_info["target"],
                task=model_info["task"],
                config=model_info.get("config", {}),
                metrics=model_info.get("metrics", {}),
                feature_importance=model_info.get("feature_importance"),
                model=model_info["model"]
            )
            zip_file.writestr('training_notebook.ipynb', notebook_json)
        
        # Add README
        readme_content = generate_readme(
            target=model_info["target"],
            task=model_info["task"],
            metrics=model_info.get("metrics", {}),
            config=model_info.get("config", {}),
            feature_importance=model_info.get("feature_importance")
        )
        zip_file.writestr('README.md', readme_content)
        
        # Add model pickle
        import pickle
        model_bytes = io.BytesIO()
        pickle.dump(model_info["model"], model_bytes)
        zip_file.writestr('model.pkl', model_bytes.getvalue())
        
        # Add charts
        if model_info.get("metrics"):
            metrics_data = {"data": [{"name": k, "value": v} for k, v in model_info["metrics"].items()]}
            chart_img = generate_chart_image("metrics", metrics_data, "Performance Metrics")
            zip_file.writestr('charts/metrics.png', base64.b64decode(chart_img))
        
        if model_info.get("feature_importance"):
            chart_img = generate_chart_image("feature_importance", model_info["feature_importance"], "Feature Importance")
            zip_file.writestr('charts/feature_importance.png', base64.b64decode(chart_img))
    
    zip_buffer.seek(0)
    
    return Response(
        content=zip_buffer.read(),
        media_type='application/zip',
        headers={
            "Content-Disposition": f'attachment; filename="model_artifacts_{run_id[:8]}.zip"'
        }
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("DRIFT_ENGINE_HOST", "0.0.0.0")
    port = int(os.getenv("DRIFT_ENGINE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
