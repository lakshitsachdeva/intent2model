"""
Pipeline builder for Intent2Model.

Converts pipeline configuration dictionaries to sklearn Pipeline objects.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from typing import Dict, List, Any, Optional


def build_pipeline(
    config: Dict[str, Any],
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Pipeline:
    """
    Build sklearn Pipeline from configuration.
    
    Args:
        config: Dictionary with keys:
            - task: "classification" or "regression"
            - preprocessing: List of preprocessing steps (e.g., ["standard_scaler", "one_hot"])
            - model: Model name (e.g., "random_forest", "logistic_regression")
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        sklearn Pipeline object
    """
    task = config.get("task", "classification")
    preprocessing = config.get("preprocessing", [])
    model_name = config.get("model", "random_forest")
    
    # Build preprocessing transformers
    transformers = []
    
    # Numeric preprocessing
    numeric_steps = []
    if "imputer" in preprocessing:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    if "standard_scaler" in preprocessing:
        numeric_steps.append(("scaler", StandardScaler()))
    
    if numeric_steps and numeric_cols:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
    elif numeric_cols:
        # Default: just impute if no preprocessing specified
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_cols)
        )
    
    # Categorical preprocessing
    categorical_steps = []
    if "imputer" in preprocessing:
        categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    if "one_hot" in preprocessing:
        categorical_steps.append(("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")))
    
    if categorical_steps and categorical_cols:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_cols))
    elif categorical_cols:
        # Default: one-hot encode if no preprocessing specified
        transformers.append(
            ("categorical", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
        )
    
    # Create column transformer
    if transformers:
        preprocessor = ColumnTransformer(transformers, remainder="drop")
    else:
        # Fallback: identity transformer
        from sklearn.preprocessing import FunctionTransformer
        preprocessor = FunctionTransformer(lambda x: x)
    
    # Build model
    if task == "classification":
        model = _get_classification_model(model_name)
    else:
        model = _get_regression_model(model_name)
    
    # Combine into pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return pipeline


def _get_classification_model(model_name: str):
    """Get classification model by name."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": None  # Will be handled separately if available
    }
    
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(random_state=42, eval_metric="logloss")
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    if model_name not in models:
        raise ValueError(f"Unknown classification model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]


def _get_regression_model(model_name: str):
    """Get regression model by name."""
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgboost": None  # Will be handled separately if available
    }
    
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=42)
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    if model_name not in models:
        raise ValueError(f"Unknown regression model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]
