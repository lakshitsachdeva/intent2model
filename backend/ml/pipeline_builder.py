"""
Pipeline builder for Intent2Model.

Converts pipeline configuration dictionaries to sklearn Pipeline objects.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from typing import Dict, List, Any, Optional

# Model complexity for anti-overfitting penalty (effective_score = cv_score - 0.05 * complexity)
# Linear = 0, Ridge/Lasso = 1, GB = 2, RF = 3, XGB/SVM/NN = 4
MODEL_COMPLEXITY: Dict[str, int] = {
    # Regression
    "linear_regression": 0,
    "ridge": 1,
    "lasso": 1,
    "elastic_net": 1,
    "svm": 4,
    "gradient_boosting": 2,
    "random_forest": 3,
    "xgboost": 4,
    # Classification
    "logistic_regression": 0,
    "naive_bayes": 0,
    "svm": 4,
    "gradient_boosting": 2,
    "random_forest": 3,
    "xgboost": 4,
}


def get_model_complexity(model_name: str) -> int:
    """Return complexity tier (0=simplest, 4=most complex) for penalty."""
    return MODEL_COMPLEXITY.get(model_name, 3)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Simple frequency encoder for high-cardinality categoricals (train-time frequency, test-time unknown -> 0).
    Produces a single numeric column per input column.
    """

    def __init__(self):
        self.maps_: Dict[int, Dict[str, float]] = {}

    def fit(self, X, y=None):
        import numpy as np
        X_arr = X
        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        X_arr = np.asarray(X_arr, dtype=object)
        self.maps_ = {}
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            # normalize to string, keep None
            col_s = [("" if v is None else str(v)) for v in col]
            counts: Dict[str, int] = {}
            for v in col_s:
                counts[v] = counts.get(v, 0) + 1
            n = float(max(1, len(col_s)))
            self.maps_[j] = {k: c / n for k, c in counts.items()}
        return self

    def transform(self, X):
        import numpy as np
        X_arr = X
        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        X_arr = np.asarray(X_arr, dtype=object)
        out = np.zeros((X_arr.shape[0], X_arr.shape[1]), dtype=float)
        for j in range(X_arr.shape[1]):
            m = self.maps_.get(j, {})
            col = X_arr[:, j]
            col_s = [("" if v is None else str(v)) for v in col]
            out[:, j] = [float(m.get(v, 0.0)) for v in col_s]
        return out


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
    feature_transforms = config.get("feature_transforms")
    
    # Build preprocessing transformers
    transformers = []

    # If we have per-feature transforms, use them (agent-driven, dataset-specific).
    if isinstance(feature_transforms, list) and len(feature_transforms) > 0:
        # Split columns into groups according to plan
        num_cols_scale: List[str] = []
        num_cols_noscale: List[str] = []
        cat_cols_onehot: List[str] = []
        cat_cols_ordinal: List[str] = []
        cat_cols_freq: List[str] = []

        impute_num_mean: List[str] = []
        impute_num_median: List[str] = []
        impute_cat_mf: List[str] = []
        impute_cat_const: List[str] = []
        dropped: set[str] = set()

        for ft in feature_transforms:
            try:
                name = ft.get("name")
                if not name:
                    continue
                if ft.get("drop") is True:
                    dropped.add(name)
                    continue

                kind = ft.get("kind")
                encode = ft.get("encode", "none")
                scale = ft.get("scale", "none")
                impute = ft.get("impute", "none")

                is_numeric = name in numeric_cols
                is_categorical = name in categorical_cols

                # imputation buckets
                if is_numeric:
                    if impute == "mean":
                        impute_num_mean.append(name)
                    elif impute == "median":
                        impute_num_median.append(name)
                else:
                    if impute == "most_frequent":
                        impute_cat_mf.append(name)
                    elif impute == "constant":
                        impute_cat_const.append(name)

                # encoding/scaling buckets
                if is_numeric:
                    if scale == "standard":
                        num_cols_scale.append(name)
                    else:
                        num_cols_noscale.append(name)
                elif is_categorical:
                    if encode == "one_hot":
                        cat_cols_onehot.append(name)
                    elif encode == "ordinal":
                        cat_cols_ordinal.append(name)
                    elif encode == "frequency":
                        cat_cols_freq.append(name)
                    else:
                        # treat as ordinal-safe by default if agent says none
                        cat_cols_ordinal.append(name)
                else:
                    # unknown: pass-through
                    num_cols_noscale.append(name)
            except Exception:
                continue

        # Build numeric pipelines
        if num_cols_scale:
            steps = []
            # impute
            steps.append(("imputer", SimpleImputer(strategy="median")))
            steps.append(("scaler", StandardScaler()))
            transformers.append(("num_scaled", Pipeline(steps), num_cols_scale))

        if num_cols_noscale:
            steps = []
            steps.append(("imputer", SimpleImputer(strategy="median")))
            transformers.append(("num", Pipeline(steps), num_cols_noscale))

        # Categorical: onehot
        if cat_cols_onehot:
            steps = []
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            # Use min_frequency to avoid blowups on high cardinality if present in sklearn version
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            steps.append(("onehot", ohe))
            transformers.append(("cat_onehot", Pipeline(steps), cat_cols_onehot))

        # Categorical: ordinal
        if cat_cols_ordinal:
            steps = []
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            steps.append(("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
            transformers.append(("cat_ordinal", Pipeline(steps), cat_cols_ordinal))

        # Categorical: frequency encoding
        if cat_cols_freq:
            steps = []
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            steps.append(("freq", FrequencyEncoder()))
            transformers.append(("cat_freq", Pipeline(steps), cat_cols_freq))

        preprocessor = ColumnTransformer(transformers, remainder="drop")
        
        # CRITICAL: Validate preprocessor has at least one transformer
        if not transformers:
            raise RuntimeError(
                "COMPILER ERROR: Preprocessor has NO transformers. "
                "This means all features were dropped or no feature_transforms were provided. "
                "Check that feature_transforms includes at least one non-dropped feature."
            )

        # Build model
        if task == "classification":
            model = _get_classification_model(model_name)
        else:
            model = _get_regression_model(model_name)

        return Pipeline([("preprocessor", preprocessor), ("model", model)])
    
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
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        "xgboost": None,  # Will be handled separately if available
        "svm": None,  # Will be handled separately
        "naive_bayes": None,  # Will be handled separately
        "gradient_boosting": None  # Will be handled separately
    }
    
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
            # mlogloss works for both binary and multi-class; logloss can fail on multi-class (e.g. Iris)
            return XGBClassifier(random_state=42, eval_metric="mlogloss")
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    if model_name == "svm":
        try:
            from sklearn.svm import SVC
            return SVC(random_state=42, probability=True)
        except ImportError:
            raise ImportError("SVM requires sklearn")
    
    if model_name == "naive_bayes":
        try:
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()
        except ImportError:
            raise ImportError("Naive Bayes requires sklearn")
    
    if model_name == "gradient_boosting":
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            # Shallow by default (anti-overfitting)
            return GradientBoostingClassifier(random_state=42, max_depth=3, n_estimators=100)
        except ImportError:
            raise ImportError("Gradient Boosting requires sklearn")
    
    if model_name not in models:
        raise ValueError(f"Unknown classification model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]


def _get_regression_model(model_name: str):
    """Get regression model by name."""
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        "xgboost": None,  # Will be handled separately if available
        "svm": None,  # Will be handled separately
        "gradient_boosting": None,  # Will be handled separately
        "ridge": None,  # Will be handled separately
        "lasso": None  # Will be handled separately
    }
    
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=42)
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    if model_name == "svm":
        try:
            from sklearn.svm import SVR
            return SVR()
        except ImportError:
            raise ImportError("SVM requires sklearn")
    
    if model_name == "gradient_boosting":
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            # Shallow by default (anti-overfitting)
            return GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=100)
        except ImportError:
            raise ImportError("Gradient Boosting requires sklearn")
    
    if model_name == "ridge":
        try:
            from sklearn.linear_model import Ridge
            return Ridge(random_state=42)
        except ImportError:
            raise ImportError("Ridge requires sklearn")
    
    if model_name == "lasso":
        try:
            from sklearn.linear_model import Lasso
            return Lasso(random_state=42)
        except ImportError:
            raise ImportError("Lasso requires sklearn")
    
    if model_name not in models:
        raise ValueError(f"Unknown regression model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]
