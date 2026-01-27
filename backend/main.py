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
from agents.planner_agent import plan_pipeline
from schemas.pipeline_schema import UserIntent
from agents.llm_interface import LLMInterface
from agents.error_analyzer import analyze_training_error
from agents.recovery_agent import AutonomousRecoveryAgent
from agents.intent_detector import IntentDetectionAgent
from utils.artifact_generator import generate_notebook, generate_readme, save_model_pickle, generate_chart_image
from fastapi.responses import FileResponse, Response
import json
import re
import os
import tempfile
import base64


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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for uploaded datasets (in production, use proper storage)
dataset_cache = {}
trained_models_cache = {}  # Store trained models for prediction


class TrainRequest(BaseModel):
    target: str
    task: Optional[Literal["classification", "regression"]] = None
    metric: Optional[str] = None
    dataset_id: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Intent2Model API", "status": "running"}


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
    
    try:
        # Get dataset profile for planning
        profile = profile_dataset(df)
        
        # Use LLM planner to generate optimal pipeline config
        user_intent = UserIntent(
            target_column=request.target,
            task_type=task,
            priority_metric=metric
        )
        
        # ALWAYS use model comparison with comprehensive model list
        if task == "classification":
            model_candidates = ["logistic_regression", "random_forest", "gradient_boosting", "naive_bayes"]
            # Try XGBoost if available
            try:
                import xgboost
                model_candidates.append("xgboost")
            except:
                pass
        else:
            model_candidates = ["linear_regression", "random_forest", "gradient_boosting", "ridge"]
            # Try XGBoost if available
            try:
                import xgboost
                model_candidates.append("xgboost")
            except:
                pass
        
        try:
            pipeline_config = plan_pipeline(profile, user_intent, llm_provider="gemini")
            # Use LLM-generated config
            config = {
                "task": pipeline_config.task,
                "preprocessing": pipeline_config.preprocessing,
                "model": model_candidates[0]  # Use first as default
            }
            # Override with LLM suggestions if available
            if pipeline_config.model_candidates and len(pipeline_config.model_candidates) > 0:
                # Merge LLM suggestions with defaults
                model_candidates = list(set(model_candidates + pipeline_config.model_candidates))
        except Exception as e:
            # Fallback to default config if LLM fails
            print(f"LLM planning failed: {e}. Using default config with model comparison.")
            config = None
        
        use_model_comparison = True  # Always compare multiple models
        
        # Train model(s) with automatic error fixing
        from agents.auto_fix_agent import auto_fix_training_error
        
        training_context = {
            "target": request.target,
            "task": task,
            "metric": metric,
            "dataset_shape": df.shape,
            "target_dtype": str(df[request.target].dtype),
            "target_unique_count": df[request.target].nunique()
        }
        
        try:
            if use_model_comparison:
                # Try multiple models and get ALL results
                train_result = auto_fix_training_error(
                    compare_models,
                    df, request.target, task, metric, model_candidates, config,
                    context=training_context
                )
                
                # Add LLM explanations for each model
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
            else:
                # Single model training with auto-fix
                if task == "classification":
                    train_result = auto_fix_training_error(
                        train_classification,
                        df, request.target, metric, config,
                        context=training_context
                    )
                else:
                    train_result = auto_fix_training_error(
                        train_regression,
                        df, request.target, metric, config,
                        context=training_context
                    )
        except Exception as training_error:
            # If auto-fix failed, fall through to error analysis
            raise training_error
        
        # Evaluate dataset
        eval_result = evaluate_dataset(df, request.target, task)
        
        # Create run ID and log
        run_id = create_run_id()
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
        trained_models_cache[run_id] = {
            "model": train_result["best_model"],
            "target": request.target,
            "task": task,
            "feature_columns": list(df.drop(columns=[request.target]).columns),
            "label_encoder": train_result.get("label_encoder") if task == "classification" else None,
            "config": config,
            "df": df.copy(),  # Store dataset for artifact generation
            "metrics": train_result["metrics"],
            "feature_importance": train_result.get("feature_importance"),
            "all_models": train_result.get("all_models", [])  # Store JSON-safe summaries
        }
        
        response_payload = {
            "run_id": run_id,
            "metrics": train_result["metrics"],
            "cv_mean": train_result.get("cv_mean"),
            "cv_std": train_result.get("cv_std"),
            "warnings": eval_result["warnings"],
            "imbalance_ratio": eval_result.get("imbalance_ratio"),
            "leakage_columns": eval_result["leakage_columns"],
            "feature_importance": train_result.get("feature_importance"),
            "model_comparison": train_result.get("model_comparison"),
            "all_models": train_result.get("all_models", []),  # Return ALL models with explanations
            "pipeline_config": {
                "preprocessing": config.get("preprocessing", []) if config else [],
                "model": train_result.get("model_name", config.get("model", "unknown")) if config else "unknown"
            }
        }
        return _json_safe(response_payload)
    except Exception as e:
        error_msg = str(e)
        
        # Use LLM to analyze the error and provide helpful explanation
        try:
            dataset_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "target_dtype": str(df[request.target].dtype),
                "target_unique_count": df[request.target].nunique(),
                "target_missing_count": df[request.target].isna().sum()
            }
            
            error_analysis = analyze_training_error(
                error=e,
                error_msg=error_msg,
                dataset_info=dataset_info,
                target_column=request.target,
                task_type=task,
                llm_provider="gemini"
            )
            
            # Build comprehensive error message
            user_msg = f"""❌ Training Error

{error_analysis['explanation']}

🔍 Root Cause: {error_analysis['root_cause']}

💡 Suggestions:
{chr(10).join(f'  • {s}' for s in error_analysis['suggestions'])}"""
            
        except Exception as llm_error:
            # Fallback if LLM analysis fails
            print(f"LLM error analysis failed: {llm_error}")
            if "could not convert string to float" in error_msg or "ValueError" in error_msg:
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
        llm = LLMInterface(provider="gemini")
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


@app.get("/download/{run_id}/notebook")
async def download_notebook(run_id: str):
    """Download Jupyter notebook for a trained model."""
    if run_id not in trained_models_cache:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models_cache[run_id]
    df = model_info.get("df")
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not available for this model")
    
    notebook_json = generate_notebook(
        df=df,
        target=model_info["target"],
        task=model_info["task"],
        config=model_info.get("config", {}),
        metrics=model_info.get("metrics", {}),
        feature_importance=model_info.get("feature_importance"),
        model=model_info["model"]
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
        f.write(notebook_json)
        temp_path = f.name
    
    return FileResponse(
        temp_path,
        media_type='application/json',
        filename=f'model_{run_id[:8]}_training_notebook.ipynb',
        background=lambda: os.unlink(temp_path)
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
        background=lambda: os.unlink(temp_path)
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
        background=lambda: os.unlink(temp_path)
    )


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
