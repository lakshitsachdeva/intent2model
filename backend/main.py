"""
FastAPI backend for Intent2Model.

Provides endpoints for dataset upload and model training.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
import pandas as pd
import io
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.profiler import profile_dataset
from ml.trainer import train_classification, train_regression
from ml.evaluator import evaluate_dataset
from utils.logging import create_run_id, log_run


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
        # Train model
        if task == "classification":
            train_result = train_classification(df, request.target, metric)
        else:
            train_result = train_regression(df, request.target, metric)
        
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
        
        return {
            "run_id": run_id,
            "metrics": train_result["metrics"],
            "cv_mean": train_result["cv_mean"],
            "cv_std": train_result["cv_std"],
            "warnings": eval_result["warnings"],
            "imbalance_ratio": eval_result.get("imbalance_ratio"),
            "leakage_columns": eval_result["leakage_columns"],
            "feature_importance": train_result.get("feature_importance")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
