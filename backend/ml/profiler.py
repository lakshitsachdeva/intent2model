"""
Dataset profiler for Intent2Model.

Analyzes datasets to detect column types, missing values, and candidate target columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def profile_dataset(df: pd.DataFrame) -> Dict:
    """
    Analyze dataset and return comprehensive profile.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        Dictionary containing:
        - n_rows: Number of rows
        - n_cols: Number of columns
        - numeric_cols: List of numeric column names
        - categorical_cols: List of categorical column names
        - missing_percent: Dictionary mapping column names to missing percentage
        - unique_counts: Dictionary mapping column names to unique value counts
        - candidate_targets: List of potential target column names
    """
    profile = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "numeric_cols": [],
        "categorical_cols": [],
        "missing_percent": {},
        "unique_counts": {},
        "candidate_targets": []
    }
    
    # Detect column types
    for col in df.columns:
        # Check if numeric
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            profile["numeric_cols"].append(col)
        else:
            profile["categorical_cols"].append(col)
        
        # Calculate missing percentage
        missing_count = df[col].isna().sum()
        profile["missing_percent"][col] = (missing_count / len(df)) * 100
        
        # Count unique values
        profile["unique_counts"][col] = df[col].nunique()
    
    # Find candidate target columns
    profile["candidate_targets"] = _find_candidate_targets(df, profile)
    
    return profile


def _find_candidate_targets(df: pd.DataFrame, profile: Dict) -> List[str]:
    """
    Identify candidate target columns using heuristics.
    
    Heuristics:
    1. Column name hints: ["target", "label", "class", "score", "result"]
    2. High variance numeric columns
    3. Low cardinality categorical columns (2-20 unique values)
    
    Args:
        df: Input DataFrame
        profile: Profile dictionary with column type info
        
    Returns:
        List of candidate target column names
    """
    candidates = []
    
    # Check column name hints (case-insensitive)
    name_hints = ["target", "label", "class", "score", "result", "y", "outcome"]
    for col in df.columns:
        col_lower = col.lower()
        if any(hint in col_lower for hint in name_hints):
            candidates.append(col)
    
    # Check numeric columns for high variance (potential regression targets)
    for col in profile["numeric_cols"]:
        if col in candidates:
            continue
        
        # Skip if too many missing values
        if profile["missing_percent"][col] > 50:
            continue
        
        # Calculate coefficient of variation (std/mean)
        col_data = df[col].dropna()
        if len(col_data) > 0 and col_data.std() > 0:
            cv = col_data.std() / abs(col_data.mean()) if col_data.mean() != 0 else col_data.std()
            # High variance suggests it might be a target
            if cv > 0.1:  # Threshold for "high variance"
                candidates.append(col)
    
    # Check categorical columns for low cardinality (potential classification targets)
    for col in profile["categorical_cols"]:
        if col in candidates:
            continue
        
        # Skip if too many missing values
        if profile["missing_percent"][col] > 50:
            continue
        
        unique_count = profile["unique_counts"][col]
        # Low cardinality (2-20) suggests classification target
        if 2 <= unique_count <= 20:
            candidates.append(col)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for col in candidates:
        if col not in seen:
            seen.add(col)
            unique_candidates.append(col)
    
    return unique_candidates
