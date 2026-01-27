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


def _safe_n_splits_classification(y: np.ndarray, desired: int = 5) -> int:
    """
    Pick a safe number of CV folds for classification.
    StratifiedKFold requires n_splits <= min_class_count and <= n_samples.
    """
    try:
        y_arr = np.asarray(y)
        n_samples = int(len(y_arr))
        if n_samples < 2:
            return 1
        # class counts
        _, counts = np.unique(y_arr, return_counts=True)
        min_class = int(counts.min()) if len(counts) else 1
        n_splits = min(desired, n_samples, min_class)
        return max(1, n_splits)
    except Exception:
        # fall back conservatively
        return 2


def _safe_n_splits_regression(n_samples: int, desired: int = 5) -> int:
    """Pick a safe number of CV folds for regression (KFold requires n_splits <= n_samples)."""
    try:
        n = int(n_samples)
        if n < 2:
            return 1
        return max(1, min(desired, n))
    except Exception:
        return 2


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
    
    # Check if target is categorical (for classification) - handle both object and numeric categorical
    le = None
    unique_count = y.nunique()
    total_count = len(y)
    
    # Determine if it's categorical: object dtype OR numeric with low cardinality
    is_categorical = (
        y.dtype == 'object' or 
        y.dtype.name == 'category' or
        (y.dtype in ['int64', 'float64', 'int32', 'float32'] and unique_count <= 20 and unique_count < total_count * 0.1)
    )
    
    if is_categorical:
        # Encode target labels - convert to string first to handle any type
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        y_classes = le.classes_
    else:
        y_encoded = y.values if hasattr(y, 'values') else y
        y_classes = None
    
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
        "roc_auc": "roc_auc_ovr" if len(np.unique(y_encoded)) > 2 else "roc_auc"
    }
    
    cv_scoring = scoring_map.get(metric, "accuracy")
    
    # Cross-validation
    n_splits = _safe_n_splits_classification(y_encoded, desired=5)
    cv_scores: List[float] = []
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, X, y_encoded, cv=cv, scoring=cv_scoring, return_train_score=True, error_score='raise'
        )
        cv_scores = cv_results["test_score"].tolist()
    else:
        # Too few samples per class; skip CV and train on full data
        cv_scores = []
    
    # Train final model on full data
    pipeline.fit(X, y_encoded)
    
    # Calculate all metrics on full training set
    y_pred = pipeline.predict(X)
    y_pred_proba = None
    try:
        y_pred_proba = pipeline.predict_proba(X)
    except:
        pass
    
    metrics = {
        "accuracy": accuracy_score(y_encoded, y_pred),
        "precision": precision_score(y_encoded, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_encoded, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_encoded, y_pred, average="macro", zero_division=0),
    }
    
    # Add ROC AUC if binary or multiclass with probabilities
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_encoded)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_encoded, y_pred_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_encoded, y_pred_proba, multi_class="ovr", average="macro")
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
        "cv_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_std": float(np.std(cv_scores)) if cv_scores else None,
        "feature_importance": feature_importance,
        "label_encoder": le  # Return encoder for prediction
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
    n_splits = _safe_n_splits_regression(len(y), desired=5)
    cv_scores: List[float] = []
    if n_splits >= 2:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring=cv_scoring, return_train_score=True, error_score='raise'
        )
        cv_scores = cv_results["test_score"].tolist()
    else:
        cv_scores = []
    
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
        "cv_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_std": float(np.std(cv_scores)) if cv_scores else None,
        "feature_importance": feature_importance,
        "label_encoder": None  # Regression doesn't need label encoder
    }


def compare_models(
    df: pd.DataFrame,
    target: str,
    task: str,
    metric: str,
    model_candidates: List[str],
    base_config: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Try multiple models and return ALL results with detailed comparison.
    
    Args:
        df: Input DataFrame
        target: Target column name
        task: "classification" or "regression"
        metric: Metric to optimize
        model_candidates: List of model names to try
        base_config: Base pipeline configuration
        
    Returns:
        Dictionary with:
        - best_model: Best model result (for backward compatibility)
        - all_models: List of ALL model results with full details
        - model_comparison: Comparison summary
    """
    results = []
    
    for model_name in model_candidates:
        try:
            if base_config:
                config = base_config.copy()
            else:
                config = {
                    "task": task,
                    "preprocessing": ["standard_scaler", "one_hot"],
                    "model": model_name
                }
            config["model"] = model_name
            
            if task == "classification":
                result = train_classification(df, target, metric, config)
            else:
                result = train_regression(df, target, metric, config)
            
            # Get the primary metric value
            primary_metric_value = result["metrics"].get(metric, result["cv_mean"])
            result["model_name"] = model_name
            result["primary_metric"] = primary_metric_value
            results.append(result)
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue
    
    if not results:
        raise ValueError("All models failed to train")
    
    # Sort by primary metric (higher is better for most metrics, except rmse/mae)
    reverse = metric not in ["rmse", "mae"]
    results.sort(key=lambda x: x["primary_metric"], reverse=reverse)
    
    best_result = results[0].copy()

    def _to_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _json_safe_model_result(r: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip non-JSON-serializable objects (sklearn pipelines, label encoders, numpy types).
        Keep only what the UI needs.
        """
        metrics = {k: _to_float(v) for k, v in (r.get("metrics") or {}).items()}
        cv_scores = [float(s) for s in (r.get("cv_scores") or [])]
        feature_importance = r.get("feature_importance")
        if isinstance(feature_importance, dict):
            # feature importance may have numpy floats
            feature_importance = {str(k): _to_float(v) for k, v in feature_importance.items()}
        else:
            feature_importance = None

        return {
            "model_name": r.get("model_name"),
            "primary_metric": _to_float(r.get("primary_metric")),
            "metrics": metrics,
            "cv_scores": cv_scores,
            "cv_mean": _to_float(r.get("cv_mean")),
            "cv_std": _to_float(r.get("cv_std")),
            "feature_importance": feature_importance,
        }

    all_models_safe = [_json_safe_model_result(r) for r in results]

    # Build comprehensive comparison (JSON-safe)
    comparison = {
        "tried_models": [r.get("model_name") for r in all_models_safe],
        "best_model": best_result.get("model_name"),
        "all_results": [
            {
                "model": r.get("model_name"),
                "metric": r.get("primary_metric"),
            }
            for r in all_models_safe
        ],
        "all_models": all_models_safe,
    }

    # Keep best_model pipeline in best_result for server-side caching,
    # but return JSON-safe all_models for UI.
    best_result["model_comparison"] = comparison
    best_result["all_models"] = all_models_safe

    return best_result
