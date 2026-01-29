"""
Plan → Code Compiler

This is the EXECUTION NERVOUS SYSTEM that converts AutoMLPlan (reasoning) 
into executable sklearn code.

CRITICAL RULE: If code does NOT reference AutoMLPlan fields, it is a BUG.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from schemas.pipeline_schema import AutoMLPlan, FeatureTransform, ModelCandidate


def compile_preprocessing_code(plan: AutoMLPlan, df: pd.DataFrame) -> str:
    """
    Compile preprocessing code from AutoMLPlan.feature_transforms.
    
    NEVER uses select_dtypes.
    NEVER applies StandardScaler unless plan explicitly says so.
    Groups features by plan semantics, not dtype.
    """
    import pandas as pd
    
    lines = []
    lines.append("# Preprocessing compiled from AutoMLPlan\n")
    lines.append("# Each feature transform is based on plan.feature_transforms\n")
    lines.append("\n")
    
    # Validate plan has feature_transforms
    if not plan.feature_transforms:
        lines.append("# ⚠️ WARNING: plan.feature_transforms is empty! Using fallback.\n")
        lines.append("# This should not happen - the plan is incomplete.\n")
        lines.append("\n")
    
    # Group features by transform strategy (from plan, not dtype)
    dropped_features = []
    numeric_scale = []
    numeric_noscale = []
    numeric_impute_mean = []
    numeric_impute_median = []
    categorical_onehot = []
    categorical_ordinal = []
    categorical_frequency = []
    categorical_impute_mf = []
    categorical_impute_const = []
    
    for ft in plan.feature_transforms:
        if ft.drop:
            dropped_features.append(ft.name)
            continue
        
        # Numeric features
        if ft.kind in ["continuous", "ordinal", "count"]:
            if ft.scale == "standard":
                numeric_scale.append(ft.name)
            else:
                numeric_noscale.append(ft.name)
            
            if ft.impute == "mean":
                numeric_impute_mean.append(ft.name)
            elif ft.impute == "median":
                numeric_impute_median.append(ft.name)
        
        # Categorical features
        elif ft.kind in ["binary", "nominal"]:
            if ft.encode == "one_hot":
                categorical_onehot.append(ft.name)
            elif ft.encode == "ordinal":
                categorical_ordinal.append(ft.name)
            elif ft.encode == "frequency":
                categorical_frequency.append(ft.name)
            
            if ft.impute == "most_frequent":
                categorical_impute_mf.append(ft.name)
            elif ft.impute == "constant":
                categorical_impute_const.append(ft.name)
    
    # Build transformers list
    lines.append("from sklearn.pipeline import Pipeline\n")
    lines.append("from sklearn.compose import ColumnTransformer\n")
    lines.append("from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder\n")
    lines.append("from sklearn.impute import SimpleImputer\n")
    lines.append("\n")
    lines.append("transformers = []\n")
    lines.append("\n")
    
    # Numeric: scaled
    if numeric_scale:
        lines.append(f"# Numeric features requiring scaling (from plan): {numeric_scale}\n")
        impute_strategy = "median" if numeric_impute_median else ("mean" if numeric_impute_mean else "median")
        lines.append(f"num_scaled_cols = {numeric_scale}\n")
        lines.append(f"if num_scaled_cols:\n")
        lines.append(f"    transformers.append((\n")
        lines.append(f"        'num_scaled',\n")
        lines.append(f"        Pipeline([\n")
        if impute_strategy != "none":
            lines.append(f"            ('imputer', SimpleImputer(strategy='{impute_strategy}')),\n")
        lines.append(f"            ('scaler', StandardScaler())\n")
        lines.append(f"        ]),\n")
        lines.append(f"        num_scaled_cols\n")
        lines.append(f"    ))\n")
        lines.append("\n")
    
    # Numeric: no scaling
    if numeric_noscale:
        lines.append(f"# Numeric features without scaling (from plan): {numeric_noscale}\n")
        impute_strategy = "median" if numeric_impute_median else ("mean" if numeric_impute_mean else "none")
        lines.append(f"num_plain_cols = {numeric_noscale}\n")
        lines.append(f"if num_plain_cols:\n")
        lines.append(f"    transformers.append((\n")
        lines.append(f"        'num_plain',\n")
        if impute_strategy != "none":
            lines.append(f"        Pipeline([('imputer', SimpleImputer(strategy='{impute_strategy}'))]),\n")
        else:
            lines.append(f"        'passthrough',\n")
        lines.append(f"        num_plain_cols\n")
        lines.append(f"    ))\n")
        lines.append("\n")
    
    # Categorical: one-hot
    if categorical_onehot:
        lines.append(f"# Categorical features: one-hot encoding (from plan): {categorical_onehot}\n")
        lines.append(f"cat_onehot_cols = {categorical_onehot}\n")
        lines.append(f"if cat_onehot_cols:\n")
        lines.append(f"    steps = []\n")
        if categorical_impute_mf:
            lines.append(f"    steps.append(('imputer', SimpleImputer(strategy='most_frequent')))\n")
        lines.append(f"    try:\n")
        lines.append(f"        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=5)\n")
        lines.append(f"    except TypeError:\n")
        lines.append(f"        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n")
        lines.append(f"    steps.append(('onehot', ohe))\n")
        lines.append(f"    transformers.append(('cat_onehot', Pipeline(steps), cat_onehot_cols))\n")
        lines.append("\n")
    
    # Categorical: ordinal
    if categorical_ordinal:
        lines.append(f"# Categorical features: ordinal encoding (from plan): {categorical_ordinal}\n")
        lines.append(f"cat_ordinal_cols = {categorical_ordinal}\n")
        lines.append(f"if cat_ordinal_cols:\n")
        lines.append(f"    steps = []\n")
        if categorical_impute_mf:
            lines.append(f"    steps.append(('imputer', SimpleImputer(strategy='most_frequent')))\n")
        lines.append(f"    steps.append(('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))\n")
        lines.append(f"    transformers.append(('cat_ordinal', Pipeline(steps), cat_ordinal_cols))\n")
        lines.append("\n")
    
    # Categorical: frequency
    if categorical_frequency:
        lines.append(f"# Categorical features: frequency encoding (from plan): {categorical_frequency}\n")
        lines.append(f"cat_freq_cols = {categorical_frequency}\n")
        lines.append(f"if cat_freq_cols:\n")
        lines.append(f"    from sklearn.base import BaseEstimator, TransformerMixin\n")
        lines.append(f"    class FrequencyEncoder(BaseEstimator, TransformerMixin):\n")
        lines.append(f"        def fit(self, X, y=None):\n")
        lines.append(f"            import numpy as np\n")
        lines.append(f"            X = np.asarray(X, dtype=object)\n")
        lines.append(f"            self.maps_ = []\n")
        lines.append(f"            for j in range(X.shape[1]):\n")
        lines.append(f"                col = [\"\" if v is None else str(v) for v in X[:, j]]\n")
        lines.append(f"                counts = {{}}\n")
        lines.append(f"                for v in col:\n")
        lines.append(f"                    counts[v] = counts.get(v, 0) + 1\n")
        lines.append(f"                n = float(max(1, len(col)))\n")
        lines.append(f"                self.maps_.append({{k: c / n for k, c in counts.items()}})\n")
        lines.append(f"            return self\n")
        lines.append(f"        def transform(self, X):\n")
        lines.append(f"            import numpy as np\n")
        lines.append(f"            X = np.asarray(X, dtype=object)\n")
        lines.append(f"            out = np.zeros((X.shape[0], X.shape[1]), dtype=float)\n")
        lines.append(f"            for j in range(X.shape[1]):\n")
        lines.append(f"                m = self.maps_[j] if j < len(self.maps_) else {{}}\n")
        lines.append(f"                col = [\"\" if v is None else str(v) for v in X[:, j]]\n")
        lines.append(f"                out[:, j] = [float(m.get(v, 0.0)) for v in col]\n")
        lines.append(f"            return out\n")
        lines.append(f"    steps = []\n")
        if categorical_impute_mf:
            lines.append(f"    steps.append(('imputer', SimpleImputer(strategy='most_frequent')))\n")
        lines.append(f"    steps.append(('freq', FrequencyEncoder()))\n")
        lines.append(f"    transformers.append(('cat_freq', Pipeline(steps), cat_freq_cols))\n")
        lines.append("\n")
    
    # Create preprocessor
    lines.append("# Create preprocessor from plan-driven transformers\n")
    if dropped_features:
        lines.append(f"# Dropped features (from plan): {dropped_features}\n")
    
    # CRITICAL: If no transformers were generated from plan.feature_transforms, add a RUNTIME fallback.
    # (Do NOT reference a Python variable named `transformers` in this compiler function.)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    target = plan.inferred_target
    numeric_cols = [c for c in numeric_cols if c != target and c not in dropped_features]
    categorical_cols = [c for c in categorical_cols if c != target and c not in dropped_features]

    lines.append("\n")
    lines.append("if len(transformers) == 0:\n")
    lines.append("    # ⚠️ WARNING: No transformers generated from plan.feature_transforms! Using runtime fallback.\n")
    if numeric_cols:
        lines.append(f"    numeric_cols = {numeric_cols}\n")
        lines.append("    transformers.append(('num_scaled', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols))\n")
    if categorical_cols:
        lines.append(f"    categorical_cols = {categorical_cols}\n")
        lines.append("    try:\n")
        lines.append("        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=5)\n")
        lines.append("    except TypeError:\n")
        lines.append("        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n")
        lines.append("    transformers.append(('cat_onehot', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', ohe)]), categorical_cols))\n")
    
    lines.append("preprocessor = ColumnTransformer(transformers, remainder='drop')\n")
    lines.append("\n")
    
    return "".join(lines)


def compile_model_code(plan: AutoMLPlan, selected_model_name: Optional[str] = None) -> str:
    """
    Compile model instantiation code. MUST use the trained run's selected model when provided.
    
    - When selected_model_name is provided (best model from training), use it for code gen
      even if not in plan.model_candidates — so notebook matches what was actually trained.
    - Otherwise use plan.model_candidates[0].
    """
    lines = []
    lines.append("# Model compiled from AutoMLPlan.model_candidates\n")
    lines.append("\n")
    
    task = plan.task_type
    is_classification = "classification" in task
    model_map = (
        {
            "logistic_regression": "LogisticRegression",
            "random_forest": "RandomForestClassifier",
            "xgboost": "XGBClassifier",
            "svm": "SVC",
            "naive_bayes": "GaussianNB",
            "gradient_boosting": "GradientBoostingClassifier",
        }
        if is_classification
        else {
            "linear_regression": "LinearRegression",
            "random_forest": "RandomForestRegressor",
            "xgboost": "XGBRegressor",
            "svm": "SVR",
            "gradient_boosting": "GradientBoostingRegressor",
            "ridge": "Ridge",
            "lasso": "Lasso",
        }
    )
    
    model_candidate = None
    if selected_model_name:
        for mc in (plan.model_candidates or []):
            if (getattr(mc, "model_name", None) or (mc if isinstance(mc, dict) else {}).get("model_name")) == selected_model_name:
                model_candidate = mc
                break
        if not model_candidate and selected_model_name in model_map:
            model_name = selected_model_name
            params = {}
            reason_md = "Selected as best model from training."
        elif model_candidate:
            mc = model_candidate
            model_name = getattr(mc, "model_name", None) or (mc.get("model_name") if isinstance(mc, dict) else None)
            params = getattr(mc, "params", None) or (mc.get("params") if isinstance(mc, dict) else {}) or {}
            reason_md = getattr(mc, "reason_md", None) or (mc.get("reason_md") if isinstance(mc, dict) else None) or "Selected based on plan."
        else:
            model_name = None
            params = {}
            reason_md = ""
    elif plan.model_candidates:
        mc = plan.model_candidates[0]
        model_name = getattr(mc, "model_name", None) or (mc.get("model_name") if isinstance(mc, dict) else None)
        params = getattr(mc, "params", None) or (mc.get("params") if isinstance(mc, dict) else {}) or {}
        reason_md = getattr(mc, "reason_md", None) or (mc.get("reason_md") if isinstance(mc, dict) else None) or "From plan."
    else:
        model_name = None
        params = {}
        reason_md = ""
    
    if not model_name or model_name not in model_map:
        raise ValueError(f"No model candidate found in plan (or selected_model_name='{selected_model_name}' not supported).")
    
    lines.append(f"# Selected model: {model_name} (from plan.model_candidates)\n")
    lines.append(f"# Reason: {reason_md}\n")
    lines.append("\n")
    
    sklearn_class = model_map.get(model_name)
    if not sklearn_class:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Build import
    if model_name == "xgboost":
        lines.append("from xgboost import XGBClassifier, XGBRegressor\n")
    elif model_name == "svm":
        lines.append("from sklearn.svm import SVC, SVR\n")
    elif model_name == "naive_bayes":
        lines.append("from sklearn.naive_bayes import GaussianNB\n")
    elif model_name == "gradient_boosting":
        lines.append("from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor\n")
    elif model_name == "ridge":
        lines.append("from sklearn.linear_model import Ridge\n")
    elif model_name == "lasso":
        lines.append("from sklearn.linear_model import Lasso\n")
    elif model_name == "logistic_regression":
        lines.append("from sklearn.linear_model import LogisticRegression\n")
    elif model_name == "random_forest":
        lines.append("from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n")
    elif model_name == "linear_regression":
        lines.append("from sklearn.linear_model import LinearRegression\n")
    
    lines.append("\n")
    
    # Build params string
    param_strs = []
    for k, v in params.items():
        if isinstance(v, str):
            param_strs.append(f"{k}='{v}'")
        else:
            param_strs.append(f"{k}={v}")
    
    # Add default random_state if not present
    if "random_state" not in params and model_name not in ["linear_regression", "ridge", "lasso", "naive_bayes"]:
        param_strs.append("random_state=42")
    
    params_code = ", ".join(param_strs) if param_strs else ""
    
    lines.append(f"model = {sklearn_class}({params_code})\n")
    lines.append("\n")
    
    return "".join(lines)


def compile_metrics_code(plan: AutoMLPlan) -> str:
    """
    Compile metrics code from AutoMLPlan.primary_metric and plan.additional_metrics.
    
    NEVER hardcodes accuracy, R², RMSE.
    Adapts to task_type and class imbalance.
    """
    lines = []
    lines.append("# Metrics compiled from AutoMLPlan\n")
    lines.append(f"# Primary metric: {plan.primary_metric}\n")
    lines.append(f"# Additional metrics: {plan.additional_metrics}\n")
    lines.append("\n")
    
    task = plan.task_type
    is_classification = "classification" in task
    
    lines.append("from sklearn.metrics import (\n")
    
    if is_classification:
        metric_imports = []
        all_metrics = [plan.primary_metric] + plan.additional_metrics
        if "accuracy" in all_metrics:
            metric_imports.append("    accuracy_score")
        if "precision" in all_metrics:
            metric_imports.append("    precision_score")
        if "recall" in all_metrics:
            metric_imports.append("    recall_score")
        if "f1" in all_metrics or "f1_score" in all_metrics:
            metric_imports.append("    f1_score")
        if "roc_auc" in all_metrics:
            metric_imports.append("    roc_auc_score")
        if "classification_report" not in metric_imports:
            metric_imports.append("    classification_report")
        
        lines.append(",\n".join(metric_imports) + "\n")
    else:
        metric_imports = []
        all_metrics = [plan.primary_metric] + plan.additional_metrics
        if "rmse" in all_metrics or "mse" in all_metrics:
            metric_imports.append("    mean_squared_error")
        if "mae" in all_metrics:
            metric_imports.append("    mean_absolute_error")
        if "r2" in all_metrics or "r2_score" in all_metrics:
            metric_imports.append("    r2_score")
        
        lines.append(",\n".join(metric_imports) + "\n")
    
    lines.append(")\n")
    lines.append("\n")
    
    return "".join(lines)


def compile_pipeline_code(plan: AutoMLPlan) -> str:
    """Compile final pipeline assembly code."""
    lines = []
    lines.append("# Assemble pipeline from plan-driven components\n")
    lines.append("pipeline = Pipeline([\n")
    lines.append("    ('preprocessor', preprocessor),\n")
    lines.append("    ('model', model)\n")
    lines.append("])\n")
    lines.append("\n")
    return "".join(lines)


def validate_plan_for_execution(plan: AutoMLPlan) -> None:
    """
    Execution guard: validate plan confidence before generating code.
    
    Raises RuntimeError if plan is not confident enough.
    """
    if plan.plan_quality == "fallback_low_confidence":
        raise RuntimeError(
            f"Cannot execute low-confidence fallback plan. "
            f"LLM planning failed: {plan.planning_error or 'Unknown error'}. "
            f"Please check LLM configuration or dataset."
        )
    
    if plan.target_confidence < 0.7:
        raise RuntimeError(
            f"Target inference confidence too low ({plan.target_confidence:.2f}). "
            f"Alternative targets: {plan.alternative_targets}. "
            f"Please specify target explicitly or verify dataset."
        )
    
    if plan.task_confidence < 0.7:
        raise RuntimeError(
            f"Task type inference confidence too low ({plan.task_confidence:.2f}). "
            f"Please specify task type explicitly."
        )
    
    # Don't raise errors - allow fallback code generation
    # The compiler will handle empty feature_transforms gracefully
    if not plan.feature_transforms:
        print("⚠️  Warning: Plan has no feature_transforms. Compiler will use fallback logic.")
    
    if not plan.model_candidates:
        print("⚠️  Warning: Plan has no model_candidates. Cannot generate model code.")
        raise RuntimeError("Plan has no model_candidates. Cannot generate model code.")
