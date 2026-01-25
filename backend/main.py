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


app = FastAPI(title="Intent2Model API", version="1.0.0")

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
    
    Returns:
        JSON with dataset profile and dataset_id for subsequent requests
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Profile dataset
        profile = profile_dataset(df)
        
        # Store dataset in cache (in production, use proper storage)
        dataset_id = create_run_id()
        dataset_cache[dataset_id] = df
        
        return {
            "dataset_id": dataset_id,
            "profile": profile,
            "message": "Dataset uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")


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
    # Get dataset
    if request.dataset_id:
        if request.dataset_id not in dataset_cache:
            raise HTTPException(status_code=404, detail="Dataset not found")
        df = dataset_cache[request.dataset_id]
    else:
        # Use most recent dataset if no ID provided
        if not dataset_cache:
            raise HTTPException(status_code=400, detail="No dataset available. Please upload a dataset first.")
        df = list(dataset_cache.values())[-1]
    
    # Validate target column
    if request.target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{request.target}' not found in dataset. Available columns: {list(df.columns)}"
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
        
        try:
            pipeline_config = plan_pipeline(profile, user_intent, llm_provider="gemini")
            # Use LLM-generated config
            config = {
                "task": pipeline_config.task,
                "preprocessing": pipeline_config.preprocessing,
                "model": pipeline_config.model_candidates[0] if pipeline_config.model_candidates else "random_forest"
            }
            use_model_comparison = len(pipeline_config.model_candidates) > 1
        except Exception as e:
            # Fallback to default config if LLM fails
            print(f"LLM planning failed: {e}. Using default config.")
            config = None
            use_model_comparison = False
        
        # Train model(s)
        if use_model_comparison and pipeline_config.model_candidates:
            # Try multiple models and pick the best
            train_result = compare_models(df, request.target, task, metric, pipeline_config.model_candidates, config)
        else:
            # Single model training
            if task == "classification":
                train_result = train_classification(df, request.target, metric, config)
            else:
                train_result = train_regression(df, request.target, metric, config)
        
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
        
        # Store trained model for prediction
        trained_models_cache[run_id] = {
            "model": train_result["best_model"],
            "target": request.target,
            "task": task,
            "feature_columns": list(df.drop(columns=[request.target]).columns),
            "label_encoder": train_result.get("label_encoder") if task == "classification" else None
        }
        
        return {
            "run_id": run_id,
            "metrics": train_result["metrics"],
            "cv_mean": train_result["cv_mean"],
            "cv_std": train_result["cv_std"],
            "warnings": eval_result["warnings"],
            "imbalance_ratio": eval_result.get("imbalance_ratio"),
            "leakage_columns": eval_result["leakage_columns"],
            "feature_importance": train_result.get("feature_importance"),
            "model_comparison": train_result.get("model_comparison"),
            "pipeline_config": {
                "preprocessing": config.get("preprocessing", []) if config else [],
                "model": train_result.get("model_name", config.get("model", "unknown")) if config else "unknown"
            }
        }
    except Exception as e:
        error_msg = str(e)
        # Convert technical errors to user-friendly messages
        if "could not convert string to float" in error_msg or "ValueError" in error_msg:
            user_msg = f"the target column '{request.target}' contains text values. make sure it's a valid column name."
        elif "All the" in error_msg and "fits failed" in error_msg:
            user_msg = f"couldn't train with '{request.target}'. the data might not be suitable for this type of model. try a different column?"
        elif "not found" in error_msg.lower():
            user_msg = f"column '{request.target}' not found. available columns: {', '.join(list(df.columns)[:5])}"
        else:
            user_msg = f"something went wrong training the model. try a different column or check your data."
        raise HTTPException(status_code=500, detail=user_msg)


class PredictRequest(BaseModel):
    run_id: str
    features: Dict[str, Any]


@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Make predictions with a trained model.
    
    Request body:
        - run_id: Run ID from training
        - features: Dictionary of feature values
        
    Returns:
        Prediction result
    """
    if request.run_id not in trained_models_cache:
        raise HTTPException(status_code=404, detail="Model not found. Train a model first.")
    
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
