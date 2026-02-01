"""
Model trainer for Intent2Model.

Handles cross-validation, model training, and metric calculation.
Auto-installs missing packages (e.g. xgboost) on ImportError when error message suggests pip install.
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import re
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)
from typing import Dict, List, Any, Optional, Tuple, Callable
import warnings


def _try_pip_install_from_error(error_msg: str) -> Tuple[bool, str]:
    """
    If error_msg suggests 'pip install X', run it once. Returns (installed, package_name).
    Used so the system can auto-install xgboost etc. instead of asking the user.
    """
    err = (error_msg or "").strip().lower()
    if "pip install" not in err:
        return False, ""
    # Match "pip install xgboost" or "Install with: pip install xgboost"
    match = re.search(r"pip\s+install\s+(\S+)", err, re.IGNORECASE)
    if not match:
        return False, ""
    pkg = match.group(1).strip()
    if not pkg or len(pkg) > 50:
        return False, ""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print(f"✅ Auto-installed {pkg} (pip install {pkg})")
            return True, pkg
    except Exception as e:
        print(f"⚠️ Auto-install failed for {pkg}: {e}")
    return False, pkg

from .pipeline_builder import build_pipeline, get_model_complexity
from .profiler import profile_dataset

# Opinionated model search order: simple first, complex last (anti-overfitting)
REGRESSION_MODEL_ORDER = [
    "ridge", "lasso", "linear_regression",  # Linear first
    "gradient_boosting", "random_forest",   # Then GB (shallow), RF (depth-limited)
    "svm", "xgboost",                       # Complex last
]
CLASSIFICATION_MODEL_ORDER = [
    "logistic_regression", "naive_bayes",   # Linear/simple first
    "gradient_boosting", "random_forest",
    "svm", "xgboost",
]

COMPLEXITY_PENALTY = 0.05  # effective_score = cv_score - COMPLEXITY_PENALTY * model_complexity
HOLDOUT_FRACTION = 0.08    # 5-10% never-touched holdout
HOLDOUT_BASELINE_MAX_RATIO = 0.75  # Fail if model_MAE > 0.75 * baseline_MAE


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
    # Confusion matrix for error analysis (labels = class indices; classes from le.classes_ for display)
    cm = confusion_matrix(y_encoded, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    if le is not None and hasattr(le, "classes_"):
        metrics["class_labels"] = le.classes_.tolist()
    
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
        "label_encoder": le,
        "confusion_matrix": metrics.get("confusion_matrix"),
        "class_labels": metrics.get("class_labels"),
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
    
    # Holdout: never-touched 5-10% for sanity check after CV
    n_holdout = max(5, int(len(df) * HOLDOUT_FRACTION))
    n_holdout = min(n_holdout, len(df) - 10)  # keep enough for train
    if n_holdout >= 5 and len(df) - n_holdout >= 10:
        df_train_val, df_holdout = train_test_split(
            df, test_size=n_holdout, random_state=42, shuffle=True
        )
    else:
        df_train_val = df
        df_holdout = None

    # Prepare data (train_val only for CV and fit)
    X = df_train_val.drop(columns=[target])
    y = df_train_val[target].copy()
    target_transformation = (config or {}).get("target_transformation")
    if target_transformation == "log1p":
        try:
            y_min = float(y.min())
            if y_min > -0.99:
                y = np.log1p(y)
        except Exception:
            target_transformation = None

    # Get column types
    profile = profile_dataset(df_train_val)
    numeric_cols = profile["numeric_cols"]
    categorical_cols = profile["categorical_cols"]
    
    # Remove target from column lists if present
    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)
    
    # Validate feature_transforms before building pipeline
    from agents.pipeline_validator import validate_feature_transforms
    feature_transforms = config.get("feature_transforms")
    if feature_transforms:
        validation = validate_feature_transforms(
            feature_transforms=feature_transforms,
            available_columns=list(X.columns),
            target=target
        )
        if not validation["valid"]:
            error_msg = "COMPILER ERROR - Invalid feature_transforms:\n"
            for err in validation["errors"]:
                error_msg += f"  - {err}\n"
            raise RuntimeError(error_msg)
        print(f"✅ Feature transforms validated: {validation.get('kept_count', 0)} features kept, {validation.get('dropped_count', 0)} dropped")
    
    # Build pipeline
    pipeline = build_pipeline(config, numeric_cols, categorical_cols)
    
    # COMPILER INVARIANT CHECKS - Fail loudly before training
    from agents.pipeline_validator import validate_pipeline_before_training
    try:
        diagnostic = validate_pipeline_before_training(
            pipeline=pipeline,
            X=X,
            y=y,
            config=config,
            feature_transforms=feature_transforms
        )
        print(f"✅ Pipeline validation passed: {diagnostic.get('output_features', 'unknown')} output features")
    except RuntimeError as e:
        # Re-raise with clear compiler error message
        raise RuntimeError(f"COMPILER ERROR: {str(e)}")
    
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
    
    # Train final model on full train_val data
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
    
    # EPISTEMICALLY HONEST ERROR GATING for regression
    from agents.error_gating import (
        compute_target_stats,
        compute_normalized_metrics,
        check_regression_error_gates,
        detect_variance_fit_illusion,
        check_holdout_baseline_sanity,
    )
    
    target_stats = compute_target_stats(y)
    normalized_metrics = compute_normalized_metrics(y, y_pred, target_stats)
    
    # Merge normalized metrics into main metrics dict
    metrics.update(normalized_metrics)
    metrics["target_mean"] = target_stats["mean"]
    metrics["target_std"] = target_stats["std"]
    metrics["target_IQR"] = target_stats["IQR"]
    
    # Holdout sanity: model must beat naive baseline (median predictor)
    holdout_residual_std = None
    if df_holdout is not None and len(df_holdout) >= 5:
        X_holdout = df_holdout.drop(columns=[target])
        y_holdout_orig = df_holdout[target].values
        try:
            pred_holdout = pipeline.predict(X_holdout)
            if target_transformation == "log1p":
                pred_holdout_orig = np.expm1(pred_holdout)
            else:
                pred_holdout_orig = pred_holdout
            holdout_mae = float(np.mean(np.abs(y_holdout_orig - pred_holdout_orig)))
            baseline_pred = float(np.median(df_train_val[target].values))
            baseline_mae = float(np.mean(np.abs(y_holdout_orig - baseline_pred)))
            sanity_passed, sanity_msg = check_holdout_baseline_sanity(
                holdout_mae, baseline_mae, max_ratio=HOLDOUT_BASELINE_MAX_RATIO
            )
            if not sanity_passed:
                raise RuntimeError(
                    f"HOLDOUT SANITY FAILED: {sanity_msg} "
                    f"(holdout_mae={holdout_mae:.4f}, baseline_mae={baseline_mae:.4f})"
                )
            holdout_residual_std = float(np.std(y_holdout_orig - pred_holdout_orig))
            if np.isnan(holdout_residual_std) or holdout_residual_std <= 0:
                holdout_residual_std = target_stats.get("std") or 1.0
        except RuntimeError:
            raise
        except Exception as e:
            # If holdout predict fails (e.g. column mismatch), do not fail training but skip uncertainty
            pass
    
    # Check error gates
    gates_passed, failed_gates = check_regression_error_gates(metrics, target_stats)
    
    if not gates_passed:
        # Training failed error gates - raise exception with structured info
        error_msg = "TRAINING REFUSED: Absolute error unacceptable.\n"
        error_msg += "Failed gates:\n"
        for gate in failed_gates:
            error_msg += f"  - {gate}\n"
        error_msg += f"\nMetrics:\n"
        error_msg += f"  - RMSE: {metrics['RMSE']:.4f}\n"
        error_msg += f"  - normalized_RMSE: {metrics['normalized_RMSE']:.4f}\n"
        error_msg += f"  - MAE: {metrics['MAE']:.4f}\n"
        error_msg += f"  - normalized_MAE: {metrics['normalized_MAE']:.4f}\n"
        error_msg += f"  - R²: {metrics['R²']:.4f}\n"
        
        # Check for variance-fit illusion
        is_illusion, illusion_desc = detect_variance_fit_illusion(metrics, target_stats)
        if is_illusion:
            error_msg += f"\n⚠️ {illusion_desc}\n"
        
        # Store failure info in metrics for diagnosis
        metrics["_failed_gates"] = failed_gates
        metrics["_target_stats"] = target_stats
        metrics["_is_variance_fit_illusion"] = is_illusion
        
        raise RuntimeError(error_msg)
    
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
    
    out = {
        "best_model": pipeline,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "cv_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_std": float(np.std(cv_scores)) if cv_scores else None,
        "feature_importance": feature_importance,
        "label_encoder": None,  # Regression doesn't need label encoder
    }
    if holdout_residual_std is not None:
        out["holdout_residual_std"] = holdout_residual_std
    if target_transformation:
        out["target_transformation"] = target_transformation
    return out


def compare_models(
    df: pd.DataFrame,
    target: str,
    task: str,
    metric: str,
    model_candidates: List[str],
    base_config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict:
    """
    Try multiple models and return ALL results with detailed comparison.
    On ImportError with "pip install X", auto-installs and retries once.
    progress_callback(message, current_index, total) for live activity (e.g. "Training model 2/5: xgboost").
    
    Args:
        df: Input DataFrame
        target: Target column name
        task: "classification" or "regression"
        metric: Metric to optimize
        model_candidates: List of model names to try
        base_config: Base pipeline configuration
        progress_callback: Optional (message, current, total) for live logging
        
    Returns:
        Dictionary with:
        - best_model: Best model result (for backward compatibility)
        - all_models: List of ALL model results with full details
        - model_comparison: Comparison summary
    """
    # CRITICAL: Define numeric_cols and categorical_cols BEFORE using them
    profile = profile_dataset(df)
    numeric_cols = list(profile["numeric_cols"])
    categorical_cols = list(profile["categorical_cols"])
    
    # Remove target from column lists if present
    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)
    
    # Opinionated model search order: simple first, complex must earn place
    order_list = REGRESSION_MODEL_ORDER if task == "regression" else CLASSIFICATION_MODEL_ORDER
    order_rank = {m: i for i, m in enumerate(order_list)}
    def _model_sort_key(name: str) -> int:
        return order_rank.get(name, 999)
    model_candidates = sorted(model_candidates, key=_model_sort_key)
    
    # Prepare X for validation
    X = df.drop(columns=[target])
    y = df[target]
    
    results = []
    total = len(model_candidates)
    
    for idx, model_name in enumerate(model_candidates):
        retried = False
        while True:
            try:
                if progress_callback:
                    progress_callback(f"Training model {idx + 1}/{total}: {model_name}...", idx + 1, total)
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
                
                # Primary metric for ranking/table: use CV mean when available (generalization), else in-sample
                cv_mean = result.get("cv_mean")
                primary_metric_value = (cv_mean if cv_mean is not None else result["metrics"].get(metric))
                if primary_metric_value is None:
                    primary_metric_value = result["metrics"].get(metric, 0)
                # Complexity penalty: effective_score = cv_score - 0.05 * complexity (complex must earn place)
                complexity = get_model_complexity(model_name)
                penalty = COMPLEXITY_PENALTY * complexity
                # For "higher is better" metrics (r2, accuracy, f1), subtract penalty
                # For "lower is better" (rmse, mae), add penalty so effective "score" is worse
                reverse = metric not in ["rmse", "mae"]
                if reverse:
                    effective = (primary_metric_value - penalty) if primary_metric_value is not None else -float("inf")
                else:
                    effective = (primary_metric_value + penalty) if primary_metric_value is not None else float("inf")
                result["model_name"] = model_name
                result["primary_metric"] = primary_metric_value
                result["effective_score"] = effective
                result["complexity"] = complexity
                results.append(result)
                break
            except Exception as e:
                err_str = str(e)
                # Auto-install on ImportError when error suggests pip install (e.g. xgboost)
                if isinstance(e, ImportError) and not retried and "pip install" in err_str.lower():
                    installed, pkg = _try_pip_install_from_error(err_str)
                    if installed and progress_callback:
                        progress_callback(f"Installed {pkg}; retrying {model_name}...", idx + 1, total)
                    if installed:
                        retried = True
                        continue
                print(f"Model {model_name} failed: {e}")
                reverse = metric not in ["rmse", "mae"]
                results.append({
                    "model_name": model_name,
                    "primary_metric": None,
                    "failed": True,
                    "error": err_str[:200],
                    "effective_score": -float("inf") if reverse else float("inf"),
                })
                break
    
    if not results:
        # Check if it's a compiler error vs training error
        error_summary = []
        # Ensure numeric_cols and categorical_cols are defined (they should be from above)
        if 'numeric_cols' not in locals() or 'categorical_cols' not in locals():
            # Fallback: define them here if somehow not defined
            profile = profile_dataset(df)
            numeric_cols = list(profile["numeric_cols"])
            categorical_cols = list(profile["categorical_cols"])
            if target in numeric_cols:
                numeric_cols.remove(target)
            if target in categorical_cols:
                categorical_cols.remove(target)
            X = df.drop(columns=[target])
            y = df[target]
        
        for model_name in model_candidates:
            # Try to build and validate pipeline for this model
            try:
                test_config = base_config.copy() if base_config else {}
                test_config["model"] = model_name
                test_pipeline = build_pipeline(test_config, numeric_cols, categorical_cols)
                
                from agents.pipeline_validator import validate_pipeline_before_training
                validate_pipeline_before_training(
                    pipeline=test_pipeline,
                    X=X,
                    y=y,
                    config=test_config,
                    feature_transforms=test_config.get("feature_transforms")
                )
            except RuntimeError as e:
                error_summary.append(f"  - {model_name}: COMPILER ERROR - {str(e)[:200]}")
            except Exception as e:
                error_summary.append(f"  - {model_name}: Training error - {str(e)[:200]}")
        
        if error_summary:
            error_msg = "All models failed. Errors:\n" + "\n".join(error_summary)
        else:
            error_msg = (
                "All models failed to train. "
                "This is likely a COMPILER ERROR (invalid pipeline) rather than a training error. "
                "Check that feature_transforms produces at least one output feature."
            )
        raise ValueError(error_msg)
    
    # Best model = highest raw primary_metric (so e.g. SVM 0.81 beats Ridge 0.78), not effective_score
    reverse = metric not in ["rmse", "mae"]
    valid_results = [r for r in results if not r.get("failed") and r.get("primary_metric") is not None]
    if not valid_results:
        valid_results = [r for r in results if not r.get("failed")]
    if not valid_results:
        failed_names = [r.get("model_name") for r in results if r.get("failed")]
        raise ValueError(
            f"All models failed to train. Failed: {', '.join(failed_names or ['all'])}. "
            "Check logs (e.g. XGBoost: pip install xgboost)."
        )
    def _best_key(r: Dict[str, Any]) -> float:
        v = r.get("primary_metric")
        if v is None:
            return -float("inf") if reverse else float("inf")
        return float(v)
    best_result = max(valid_results, key=_best_key).copy()

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
        Keep only what the UI needs. Handles failed results (no metrics, failed=True).
        """
        if r.get("failed"):
            return {
                "model_name": r.get("model_name"),
                "primary_metric": None,
                "failed": True,
                "error": r.get("error", "Unknown error"),
            }
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
    # Keep fitted pipelines/encoders server-side (NOT JSON-serializable)
    pipelines_by_model = {
        str(r.get("model_name")): {
            "pipeline": r.get("best_model"),
            "label_encoder": r.get("label_encoder"),
            "cv_mean": r.get("cv_mean"),
            "cv_std": r.get("cv_std"),
            "metrics": r.get("metrics"),
            "feature_importance": r.get("feature_importance"),
        }
        for r in results
        if r.get("model_name") and r.get("best_model") is not None
    }

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
    best_result["pipelines_by_model"] = pipelines_by_model

    return best_result
