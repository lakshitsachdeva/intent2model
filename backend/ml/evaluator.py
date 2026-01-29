"""
Model evaluator for Intent2Model.

Detects potential issues: class imbalance, data leakage, small datasets.
Aggressive feature pruning (variance, missing, cardinality, weak univariate).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Aggressive feature pruning (mandatory before training)
# ---------------------------------------------------------------------------

def prune_features_aggressive(
    df: pd.DataFrame,
    target: str,
    task: str,
    *,
    missing_ratio_threshold: float = 0.40,
    cardinality_ratio_threshold: float = 0.80,
    variance_near_zero_threshold: float = 1e-10,
    regression_weak_fraction_drop: float = 0.35,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop bad/weak features before model training. Deterministic, no LLM.

    Drops:
    - Variance ≈ 0 (constant columns)
    - Missing ratio > missing_ratio_threshold (default 40%)
    - Cardinality > cardinality_ratio_threshold * n_rows (ID-like)

    For regression only:
    - Univariate correlation (or mutual info) with target; drop bottom
      regression_weak_fraction_drop (default 35%) weakest.

    Returns:
        (df_pruned, list of dropped column names)
    """
    if target not in df.columns:
        return df.copy(), []
    n_rows = len(df)
    if n_rows < 2:
        return df.copy(), []

    feature_cols = [c for c in df.columns if c != target]
    dropped: List[str] = []

    # 1) Variance ≈ 0
    for col in feature_cols[:]:
        if col in dropped:
            continue
        try:
            s = df[col].dropna()
            if len(s) < 2:
                dropped.append(col)
                feature_cols = [c for c in feature_cols if c != col]
                continue
            if s.dtype in ["int64", "float64", "int32", "float32"]:
                if s.var() is None or s.var() <= variance_near_zero_threshold:
                    dropped.append(col)
                    feature_cols = [c for c in feature_cols if c != col]
            else:
                if s.nunique() <= 1:
                    dropped.append(col)
                    feature_cols = [c for c in feature_cols if c != col]
        except Exception:
            continue

    # 2) Missing ratio > threshold
    for col in list(df.columns):
        if col == target or col in dropped:
            continue
        missing_ratio = df[col].isna().sum() / n_rows
        if missing_ratio > missing_ratio_threshold:
            dropped.append(col)
            if col in feature_cols:
                feature_cols = [c for c in feature_cols if c != col]

    # 3) Cardinality > 0.8 * n_rows (ID-like)
    for col in list(df.columns):
        if col == target or col in dropped:
            continue
        n_unique = df[col].nunique()
        if n_unique >= cardinality_ratio_threshold * n_rows:
            dropped.append(col)
            if col in feature_cols:
                feature_cols = [c for c in feature_cols if c != col]

    # 4) Regression: drop bottom 30–40% weakest by univariate correlation
    if task == "regression" and feature_cols:
        y = df[target]
        if pd.api.types.is_numeric_dtype(y):
            strengths: List[Tuple[str, float]] = []
            for col in feature_cols:
                try:
                    x = df[col]
                    if pd.api.types.is_numeric_dtype(x):
                        corr = abs(x.corr(y))
                        if pd.isna(corr):
                            corr = 0.0
                        strengths.append((col, float(corr)))
                    else:
                        # Categorical: use mutual info or simple encoding correlation
                        try:
                            from sklearn.feature_selection import mutual_info_regression
                            enc = pd.get_dummies(df[[col]], drop_first=True)
                            if enc.shape[1] == 0:
                                strengths.append((col, 0.0))
                            else:
                                mi = mutual_info_regression(enc, y, random_state=42)
                                strengths.append((col, float(np.mean(np.abs(mi))) if len(mi) else 0.0))
                        except Exception:
                            strengths.append((col, 0.0))
                except Exception:
                    strengths.append((col, 0.0))
            strengths.sort(key=lambda t: t[1])
            n_drop = max(0, int(len(strengths) * regression_weak_fraction_drop))
            # Never drop so many that we have zero features left
            n_keep = len(strengths) - n_drop
            if n_keep < 1 and strengths:
                n_drop = len(strengths) - 1
            for i in range(n_drop):
                col = strengths[i][0]
                if col not in dropped:
                    dropped.append(col)
                    feature_cols = [c for c in feature_cols if c != col]

    kept = [c for c in df.columns if c not in dropped]
    if target not in kept:
        kept.append(target)
    df_pruned = df[kept].copy()
    return df_pruned, dropped


def evaluate_dataset(
    df: pd.DataFrame,
    target: str,
    task: str = "classification"
) -> Dict:
    """
    Evaluate dataset for potential issues.
    
    Args:
        df: Input DataFrame
        target: Target column name
        task: "classification" or "regression"
        
    Returns:
        Dictionary with:
        - warnings: List of warning messages
        - imbalance_ratio: Ratio of minority to majority class (classification only)
        - leakage_columns: List of columns that might cause data leakage
    """
    warnings_list = []
    imbalance_ratio = None
    leakage_columns = []
    
    # Check dataset size
    n_rows = len(df)
    n_cols = len(df.columns)
    
    if n_rows < 50:
        warnings_list.append(f"Dataset is very small ({n_rows} rows). Model may not generalize well.")
    elif n_rows < 100:
        warnings_list.append(f"Dataset is small ({n_rows} rows). Consider collecting more data.")
    
    if n_rows < n_cols * 5:
        warnings_list.append(f"More features ({n_cols}) than recommended for dataset size ({n_rows} rows). Risk of overfitting.")
    
    # Check for data leakage
    leakage_columns = _detect_leakage(df, target)
    if leakage_columns:
        warnings_list.append(f"Potential data leakage detected in columns: {', '.join(leakage_columns)}")
    
    # Check for class imbalance (classification only)
    if task == "classification":
        imbalance_ratio = _check_imbalance(df[target])
        if imbalance_ratio is not None and imbalance_ratio < 0.1:
            warnings_list.append(
                f"Severe class imbalance detected (ratio: {imbalance_ratio:.3f}). "
                "Consider using class weights or resampling techniques."
            )
        elif imbalance_ratio is not None and imbalance_ratio < 0.3:
            warnings_list.append(
                f"Moderate class imbalance detected (ratio: {imbalance_ratio:.3f}). "
                "Consider using class weights."
            )
    
    # Check for high missing values
    missing_threshold = 0.5
    for col in df.columns:
        if col == target:
            continue
        missing_pct = df[col].isna().sum() / len(df)
        if missing_pct > missing_threshold:
            warnings_list.append(
                f"Column '{col}' has {missing_pct*100:.1f}% missing values. "
                "Consider removing or imputing."
            )
    
    # Check for constant columns (no variance)
    for col in df.columns:
        if col == target:
            continue
        if df[col].nunique() <= 1:
            warnings_list.append(f"Column '{col}' has no variance (constant values). Consider removing.")
    
    return {
        "warnings": warnings_list,
        "imbalance_ratio": imbalance_ratio,
        "leakage_columns": leakage_columns
    }


def infer_classification_primary_metric(
    df: pd.DataFrame,
    target: str,
    *,
    risk_sensitive_keywords: Optional[List[str]] = None,
) -> str:
    """
    Infer primary metric for classification. No raw accuracy default.
    - Imbalanced → recall_macro or roc_auc (PR-AUC if binary and imbalanced)
    - Risk-sensitive (target name hints) → recall_macro
    - Balanced → f1_macro

    Returns one of: "recall", "f1", "roc_auc", "precision".
    LOCK this across retries; planner must not change it mid-run.
    """
    if target not in df.columns:
        return "f1"
    y = df[target]
    value_counts = y.value_counts()
    n_classes = len(value_counts)
    if n_classes < 2:
        return "f1"
    minority_count = value_counts.min()
    majority_count = value_counts.max()
    imbalance_ratio = minority_count / majority_count if majority_count else 0

    # Risk-sensitive: column name hints (fraud, churn, defect, failure, death, etc.)
    risk_sensitive_keywords = risk_sensitive_keywords or [
        "fraud", "churn", "defect", "failure", "death", "default", "attrit", "cancel"
    ]
    target_lower = target.lower()
    if any(kw in target_lower for kw in risk_sensitive_keywords):
        return "recall"

    # Imbalanced: prefer recall or roc_auc
    if imbalance_ratio < 0.1:
        return "recall"
    if imbalance_ratio < 0.3:
        return "roc_auc" if n_classes == 2 else "recall"

    # Balanced
    return "f1"


def _check_imbalance(y: pd.Series) -> Optional[float]:
    """
    Check class imbalance ratio.
    
    Returns ratio of minority class to majority class.
    Returns None if not applicable (regression or single class).
    """
    value_counts = y.value_counts()
    
    if len(value_counts) < 2:
        return None
    
    minority_count = value_counts.min()
    majority_count = value_counts.max()
    
    return minority_count / majority_count


def _detect_leakage(df: pd.DataFrame, target: str) -> List[str]:
    """
    Detect potential data leakage columns.
    
    Heuristics:
    1. Columns with "id", "index", "key" in name (except target)
    2. Columns that are perfect predictors (correlation = 1.0 or unique per row)
    3. Columns with "target", "label", "outcome" in name (except target itself)
    4. Columns that are duplicates of target
    """
    leakage_cols = []
    
    # Check for ID/index columns
    id_keywords = ["id", "index", "key", "uuid", "guid"]
    for col in df.columns:
        if col == target:
            continue
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in id_keywords):
            # Check if it's actually an ID (high cardinality, unique per row)
            if df[col].nunique() == len(df):
                leakage_cols.append(col)
    
    # Check for perfect correlation with target (numeric only)
    if df[target].dtype in ['int64', 'float64', 'int32', 'float32']:
        for col in df.columns:
            if col == target:
                continue
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                correlation = abs(df[col].corr(df[target]))
                if not pd.isna(correlation) and correlation > 0.99:
                    leakage_cols.append(col)
    
    # Check for columns that are duplicates of target
    for col in df.columns:
        if col == target:
            continue
        if df[col].equals(df[target]):
            leakage_cols.append(col)
    
    # Check for target-like column names
    target_keywords = ["target", "label", "outcome", "y_true", "actual"]
    for col in df.columns:
        if col == target:
            continue
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in target_keywords):
            leakage_cols.append(col)
    
    # Remove duplicates
    return list(set(leakage_cols))
