"""
Model evaluator for Intent2Model.

Detects potential issues: class imbalance, data leakage, small datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


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
