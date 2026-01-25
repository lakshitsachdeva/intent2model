"""
Model trainer for Intent2Model.

Handles cross-validation, model training, and metric calculation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, List, Any, Optional
import warnings

from .pipeline_builder import build_pipeline
from .profiler import profile_dataset


def train_classification(
    df: pd.DataFrame,
    target: str,
    metric: str = "accuracy",
    config: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Train classification models with cross-validation.
    
    Args:
        df: Input DataFrame
        target: Target column name
        metric: Metric to optimize ("accuracy", "precision", "recall", "f1", "roc_auc")
        config: Pipeline configuration dict (optional, will use defaults if not provided)
        
    Returns:
        Dictionary with:
        - best_model: Fitted pipeline
        - metrics: Dictionary of all metrics
        - cv_scores: List of cross-validation scores
        - feature_importance: Dictionary of feature importances (if available)
    """
    if config is None:
        config = {
            "task": "classification",
            "preprocessing": ["standard_scaler", "one_hot"],
            "model": "random_forest"
        }
    
    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]
    
    # Get column types
    profile = profile_dataset(df)
    numeric_cols = profile["numeric_cols"]
    categorical_cols = profile["categorical_cols"]
    
    # Remove target from column lists if present
    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)
    
    # Build pipeline
    pipeline = build_pipeline(config, numeric_cols, categorical_cols)
    
    # Select scoring metric
    scoring_map = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "roc_auc": "roc_auc_ovr" if len(np.unique(y)) > 2 else "roc_auc"
    }
    
    cv_scoring = scoring_map.get(metric, "accuracy")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=cv_scoring, return_train_score=True
    )
    cv_scores = cv_results["test_score"].tolist()
    
    # Train final model on full data
    pipeline.fit(X, y)
    
    # Calculate all metrics on full training set
    y_pred = pipeline.predict(X)
    y_pred_proba = None
    try:
        y_pred_proba = pipeline.predict_proba(X)
    except:
        pass
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y, y_pred, average="macro", zero_division=0),
    }
    
    # Add ROC AUC if binary or multiclass with probabilities
    if y_pred_proba is not None:
        try:
            if len(np.unique(y)) == 2:
                metrics["roc_auc"] = roc_auc_score(y, y_pred_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y, y_pred_proba, multi_class="ovr", average="macro")
        except:
            metrics["roc_auc"] = None
    
    # Get feature importance if available
    feature_importance = None
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            # Get feature names after preprocessing
            preprocessor = pipeline.named_steps["preprocessor"]
            try:
                # Try to get feature names from ColumnTransformer
                feature_names = []
                for name, transformer, cols in preprocessor.transformers_:
                    if hasattr(transformer, "get_feature_names_out"):
                        feature_names.extend(transformer.get_feature_names_out(cols))
                    elif hasattr(transformer, "named_steps"):
                        # Handle Pipeline within ColumnTransformer
                        for step_name, step_transformer in transformer.named_steps.items():
                            if hasattr(step_transformer, "get_feature_names_out"):
                                feature_names.extend(step_transformer.get_feature_names_out(cols))
                                break
                        else:
                            feature_names.extend([f"{name}_{i}" for i in range(len(cols))])
                    else:
                        feature_names.extend([f"{name}_{col}" for col in cols])
                
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            except:
                # Fallback: use indices
                feature_importance = {
                    f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)
                }
    except:
        pass
    
    return {
        "best_model": pipeline,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "feature_importance": feature_importance
    }


def train_regression(
    df: pd.DataFrame,
    target: str,
    metric: str = "rmse",
    config: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Train regression models with cross-validation.
    
    Args:
        df: Input DataFrame
        target: Target column name
        metric: Metric to optimize ("rmse", "r2", "mae")
        config: Pipeline configuration dict (optional, will use defaults if not provided)
        
    Returns:
        Dictionary with:
        - best_model: Fitted pipeline
        - metrics: Dictionary of all metrics
        - cv_scores: List of cross-validation scores
        - feature_importance: Dictionary of feature importances (if available)
    """
    if config is None:
        config = {
            "task": "regression",
            "preprocessing": ["standard_scaler", "one_hot"],
            "model": "random_forest"
        }
    
    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]
    
    # Get column types
    profile = profile_dataset(df)
    numeric_cols = profile["numeric_cols"]
    categorical_cols = profile["categorical_cols"]
    
    # Remove target from column lists if present
    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)
    
    # Build pipeline
    pipeline = build_pipeline(config, numeric_cols, categorical_cols)
    
    # Select scoring metric (note: sklearn uses negative MSE for RMSE)
    scoring_map = {
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
        "mae": "neg_mean_absolute_error"
    }
    
    cv_scoring = scoring_map.get(metric, "r2")
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=cv_scoring, return_train_score=True
    )
    cv_scores = cv_results["test_score"].tolist()
    
    # For negative scores (RMSE, MAE), convert to positive
    if metric in ["rmse", "mae"]:
        cv_scores = [-s for s in cv_scores]
    
    # Train final model on full data
    pipeline.fit(X, y)
    
    # Calculate all metrics on full training set
    y_pred = pipeline.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    metrics = {
        "rmse": np.sqrt(mse),
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "mse": mse
    }
    
    # Get feature importance if available
    feature_importance = None
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            # Get feature names after preprocessing
            preprocessor = pipeline.named_steps["preprocessor"]
            try:
                # Try to get feature names from ColumnTransformer
                feature_names = []
                for name, transformer, cols in preprocessor.transformers_:
                    if hasattr(transformer, "get_feature_names_out"):
                        feature_names.extend(transformer.get_feature_names_out(cols))
                    elif hasattr(transformer, "named_steps"):
                        # Handle Pipeline within ColumnTransformer
                        for step_name, step_transformer in transformer.named_steps.items():
                            if hasattr(step_transformer, "get_feature_names_out"):
                                feature_names.extend(step_transformer.get_feature_names_out(cols))
                                break
                        else:
                            feature_names.extend([f"{name}_{i}" for i in range(len(cols))])
                    else:
                        feature_names.extend([f"{name}_{col}" for col in cols])
                
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            except:
                # Fallback: use indices
                feature_importance = {
                    f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)
                }
    except:
        pass
    
    return {
        "best_model": pipeline,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "feature_importance": feature_importance
    }
