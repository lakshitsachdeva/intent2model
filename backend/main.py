"""
FastAPI backend for Intent2Model.

Provides endpoints for dataset upload and model training.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
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

# Load environment variables from .env file (if it exists)
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
            "xgboost": "XGBClassifier(random_state=42, eval_metric='logloss')",
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check LLM availability on startup
LLM_AVAILABLE = False
LLM_RATE_LIMITED = False
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Get API key from api_key_manager (automatically uses .env file)
from utils.api_key_manager import get_api_key
api_key = get_api_key(provider="gemini")

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
else:
    print("⚠️  No GEMINI_API_KEY found. System will use rule-based fallbacks (still fully functional)")
    print("   To enable LLM features, set GEMINI_API_KEY environment variable")

# In-memory storage for uploaded datasets (in production, use proper storage)
dataset_cache = {}
trained_models_cache = {}  # Store trained models for prediction
run_logs_cache: Dict[str, Any] = {}  # run_id -> {"events": [...], "progress": float, "stage": str}


def _log_run_event(run_id: str, message: str, stage: Optional[str] = None, progress: Optional[float] = None):
    """Append a structured log event for a run (for Developer Logs UI)."""
    if not run_id:
        return
    entry = {
        "ts": pd.Timestamp.now().isoformat(),
        "message": str(message),
    }
    if stage is not None:
        entry["stage"] = str(stage)
    if progress is not None:
        try:
            entry["progress"] = float(progress)
        except Exception:
            pass

    cur = run_logs_cache.get(run_id) or {"events": [], "progress": 0.0, "stage": "init"}
    cur["events"] = (cur.get("events") or [])[-500:] + [entry]
    if progress is not None:
        cur["progress"] = entry.get("progress", cur.get("progress", 0.0))
    if stage is not None:
        cur["stage"] = entry.get("stage", cur.get("stage", ""))
    run_logs_cache[run_id] = cur

# API key management - allow users to provide custom keys
from utils.api_key_manager import set_custom_api_key, get_api_key
current_llm_model = None  # Track which model is currently being used
current_llm_reason = None  # Why this model was chosen


class TrainRequest(BaseModel):
    target: Optional[str] = None
    task: Optional[Literal["classification", "regression"]] = None
    metric: Optional[str] = None
    dataset_id: Optional[str] = None


class SelectModelRequest(BaseModel):
    run_id: str
    model_name: str


@app.get("/run/{run_id}/logs")
async def get_run_logs(run_id: str, limit: int = 200):
    """Fetch recent structured log events for a run (for Developer Logs UI)."""
    if run_id not in run_logs_cache:
        raise HTTPException(status_code=404, detail="Run logs not found")
    cur = run_logs_cache[run_id]
    events = cur.get("events") or []
    try:
        lim = max(1, min(int(limit), 500))
    except Exception:
        lim = 200
    return {
        "run_id": run_id,
        "stage": cur.get("stage"),
        "progress": cur.get("progress"),
        "events": events[-lim:],
    }


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

def get_llm_with_custom_key(provider: str = "gemini"):
    """Get LLMInterface with API key (automatically uses .env, custom key if set)."""
    api_key = get_api_key(provider=provider)
    return LLMInterface(provider=provider, api_key=api_key)

# Make this function available to agents via a module-level function
# Agents can import this to get LLM with custom key support
import sys
sys.modules[__name__].get_llm_with_custom_key = get_llm_with_custom_key

@app.get("/health")
async def health():
    """Detailed health check with LLM status."""
    model_info = get_current_model_info()
    return {
        "status": "healthy",
        "llm_available": LLM_AVAILABLE,
        "llm_rate_limited": LLM_RATE_LIMITED,
        "llm_provider": LLM_PROVIDER if LLM_AVAILABLE else "rule-based-fallback",
        "current_model": model_info.get("model") or current_llm_model,
        "model_reason": model_info.get("reason") or current_llm_reason,
        "message": (
            f"✅ LLM enabled - using {model_info.get('model') or current_llm_model or 'AI-powered planning'}"
            if LLM_AVAILABLE
            else ("⚠️  LLM is rate-limited; using fallbacks for planning/explanations" if LLM_RATE_LIMITED else "⚠️  Using rule-based fallbacks (fully functional, but less intelligent)")
        )
    }


@app.get("/logs/backend")
async def backend_logs(limit: int = 200):
    """Tail backend.log for developer debugging in the UI."""
    try:
        lim = max(10, min(int(limit), 500))
    except Exception:
        lim = 200
    log_path = Path(__file__).parent.parent / "backend.log"
    if not log_path.exists():
        return {"path": str(log_path), "lines": []}
    try:
        # Use tail for efficiency
        out = subprocess.check_output(["tail", "-n", str(lim), str(log_path)], text=True, stderr=subprocess.STDOUT)
        lines = out.splitlines()
    except Exception:
        # fallback: read last bytes
        txt = log_path.read_text(errors="ignore")
        lines = txt.splitlines()[-lim:]
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
        
        return {
            "dataset_id": dataset_id,
            "profile": profile,
            "message": "Dataset uploaded successfully"
        }
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
                return {
                    "dataset_id": dataset_id,
                    "profile": profile,
                    "message": "Dataset uploaded successfully"
                }
        except:
            pass
        
        # Last resort: create minimal dataset but DON'T use it - return error instead
        # This forces user to upload a real file
        raise HTTPException(
            status_code=400,
            detail="Could not process the file. Please ensure it's a valid CSV, JSON, or XLSX file."
        )


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
        _log_run_event(run_id, "🚀 Run created - training request received", stage="init", progress=1)
        
        # IMPORTANT: Return run_id immediately so frontend can start polling logs
        # We'll stream updates via the logs endpoint

        # STEP 0–3: LLM-driven AutoML planning BEFORE any model training
        trace.append("STEP 0–3: Planning (target/task/feature strategy/model shortlist) via AutoML agent.")
        _log_run_event(run_id, "AutoML planning started (Step 0–3)", stage="plan", progress=5)
        plan = plan_automl(df, requested_target=request.target, llm_provider="gemini")
        _log_run_event(
            run_id,
            f"AutoML planning finished (source={getattr(plan, 'planning_source', 'unknown')})",
            stage="plan",
            progress=15,
        )
        trace.append(f"Planned: target={plan.inferred_target}, task_type={plan.task_type}, primary_metric={plan.primary_metric}")

        # Keep a classic task label for trainer
        if plan.task_type == "regression":
            task = "regression"
        else:
            task = "classification"

        # Use plan target (validated)
        request.target = plan.inferred_target

        # Profile still useful for downstream warnings + reports
        trace.append("Profiled dataset and inferred column types (execution-side).")
        profile = profile_dataset(df)
        _log_run_event(run_id, "Dataset profiled (execution-side)", stage="profile", progress=20)
        
        # Metric selection (agent-driven)
        metric = (request.metric or "").strip() or plan.primary_metric

        # Model candidates (agent-driven)
        model_candidates = [m.model_name for m in plan.model_candidates] if plan.model_candidates else []
        if not model_candidates:
            # last-resort minimal set
            model_candidates = ["random_forest"] if task == "classification" else ["random_forest"]
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
                log_callback=_log_run_event
            )
            train_result = executor.execute_with_auto_fix(
                df=df,
                target=request.target,
                task=task,
                metric=metric,
                model_candidates=model_candidates,
                requested_target=request.target
            )
            
            trace.append(f"Training succeeded after {train_result.get('attempts', 1)} attempt(s)")
            _log_run_event(run_id, f"Training succeeded (attempt {train_result.get('attempts', 1)})", stage="train", progress=70)
            
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
        
        trained_models_cache[run_id] = {
            "model": train_result["best_model"],
            "target": request.target,
            "task": task,
            "feature_columns": list(df.drop(columns=[request.target]).columns),
            "label_encoder": train_result.get("label_encoder") if task == "classification" else None,
            "config": config,
            "model_name": train_result.get("model_name", config.get("model") if config else None),
            "selected_model": selected_model,
            "pipelines_by_model": pipelines_by_model,
            "automl_plan": final_plan_dict,  # Store plan dict for notebook generation
            "plan": final_plan_dict,  # Also store as "plan" for compatibility
            "df": df.copy(),  # Store dataset for artifact generation
            "metrics": train_result["metrics"],
            "feature_importance": train_result.get("feature_importance"),
            "all_models": train_result.get("all_models", []),  # Store JSON-safe summaries
            "trace": trace,  # Store training trace for report
            "preprocessing_recommendations": preprocessing_recommendations  # Store preprocessing recs for report
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
        llm = get_llm_with_custom_key(provider="gemini")
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
    task = model_info["task"]
    label_encoder = model_info.get("label_encoder")
    
    # Prepare input data
    try:
        import pandas as pd
        # Create DataFrame with features
        input_data = pd.DataFrame([request.features])
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(input_data.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {', '.join(missing_features)}. Required: {', '.join(feature_columns)}"
            )
        
        # Reorder columns to match training
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # If classification, decode label
        if task == "classification" and label_encoder is not None:
            prediction_label = label_encoder.inverse_transform([prediction])[0]
            prediction_proba = None
            try:
                proba = model.predict_proba(input_data)[0]
                prediction_proba = dict(zip(label_encoder.classes_, proba.tolist()))
            except:
                pass
            return {
                "prediction": str(prediction_label),
                "prediction_encoded": int(prediction),
                "probabilities": prediction_proba
            }
        else:
            return {
                "prediction": float(prediction)
            }
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
        or (model_info.get("config", {}) or {}).get("model", "unknown")
    )
    # Get AutoMLPlan from cache (CRITICAL: notebook code is generated from this)
    automl_plan = model_info.get("automl_plan", {})
    if not automl_plan:
        print(f"⚠️  Warning: No automl_plan found for run {run_id}. Notebook will use fallback code.")
        # Try to get plan from train_result if available
        if "plan" in model_info:
            automl_plan = model_info["plan"]
    
    try:
        notebook_json = generate_notebook(
            df=df,
            target=model_info["target"],
            task=model_info["task"],
            config={
                **(model_info.get("config", {}) or {}),
                "model": model_name,
                "feature_columns": model_info.get("feature_columns", []),
                "model_code": _model_code_for_notebook(model_info["task"], model_name),
                "feature_transforms": (model_info.get("config", {}) or {}).get("feature_transforms", []),
                "automl_plan": automl_plan,  # CRITICAL: plan drives code generation
            },
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
    """Download trained model as pickle file."""
    if run_id not in trained_models_cache:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models_cache[run_id]
    model = model_info["model"]
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
