"""
Error Gating - Epistemically Honest Failure Detection.

This module implements hard failure gates that prevent the system
from claiming success when predictions are unusable.

High R² does NOT override failure.
Variance explanation ≠ usable prediction.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats


def compute_target_stats(y: pd.Series) -> Dict[str, float]:
    """
    Compute comprehensive target statistics for error gating.
    
    Returns:
        Dictionary with: mean, std, IQR, MAD, skew, min, max, median
    """
    y_arr = np.asarray(y, dtype=float)
    y_arr = y_arr[~np.isnan(y_arr)]  # Remove NaNs
    
    if len(y_arr) == 0:
        return {
            "mean": 0.0,
            "std": 1.0,
            "IQR": 1.0,
            "MAD": 1.0,
            "skew": 0.0,
            "min": 0.0,
            "max": 1.0,
            "median": 0.0,
        }
    
    q25, q50, q75 = np.percentile(y_arr, [25, 50, 75])
    iqr = q75 - q25
    
    # Median Absolute Deviation (robust to outliers)
    mad = np.median(np.abs(y_arr - q50))
    if mad == 0:
        mad = np.std(y_arr)  # Fallback to std if MAD is zero
    
    # Skewness
    try:
        skew = float(stats.skew(y_arr))
    except:
        skew = 0.0
    
    return {
        "mean": float(np.mean(y_arr)),
        "std": float(np.std(y_arr)) if len(y_arr) > 1 else 1.0,
        "IQR": float(iqr) if iqr > 0 else 1.0,
        "MAD": float(mad),
        "skew": skew,
        "min": float(np.min(y_arr)),
        "max": float(np.max(y_arr)),
        "median": float(q50),
    }


def compute_normalized_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_stats: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute normalized error metrics for regression.
    
    Returns:
        Dictionary with: RMSE, MAE, normalized_RMSE, normalized_MAE, R²
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    
    # Remove NaNs
    mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    
    if len(y_true_arr) == 0:
        return {
            "RMSE": float('inf'),
            "MAE": float('inf'),
            "normalized_RMSE": float('inf'),
            "normalized_MAE": float('inf'),
            "R²": -float('inf'),
        }
    
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)
    
    if target_stats is None:
        target_stats = compute_target_stats(y_true)
    
    target_std = target_stats["std"]
    target_iqr = target_stats["IQR"]
    
    # Normalized metrics
    normalized_rmse = rmse / target_std if target_std > 0 else float('inf')
    normalized_mae = mae / target_iqr if target_iqr > 0 else float('inf')
    
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "normalized_RMSE": float(normalized_rmse),
        "normalized_MAE": float(normalized_mae),
        "R²": float(r2),
    }


def check_regression_error_gates(
    metrics: Dict[str, float],
    target_stats: Dict[str, float]
) -> Tuple[bool, List[str]]:
    """
    Check mandatory failure gates for regression tasks.
    
    Returns:
        (passed: bool, failed_gates: List[str])
    
    Failure rules:
    - If RMSE > 0.5 * target_std → FAIL
    - If MAE > 0.5 * target_IQR → FAIL
    - If RMSE / target_mean > 0.25 → FAIL (if target_mean != 0)
    """
    failed_gates = []
    
    rmse = metrics.get("RMSE", float('inf'))
    mae = metrics.get("MAE", float('inf'))
    normalized_rmse = metrics.get("normalized_RMSE", float('inf'))
    normalized_mae = metrics.get("normalized_MAE", float('inf'))
    
    target_std = target_stats.get("std", 1.0)
    target_iqr = target_stats.get("IQR", 1.0)
    target_mean = target_stats.get("mean", 0.0)
    
    # Gate 1: RMSE > 0.5 * target_std
    if normalized_rmse > 0.5:
        failed_gates.append(f"normalized_RMSE ({normalized_rmse:.3f}) > 0.5 * target_std")
    
    # Gate 2: MAE > 0.5 * target_IQR
    if normalized_mae > 0.5:
        failed_gates.append(f"normalized_MAE ({normalized_mae:.3f}) > 0.5 * target_IQR")
    
    # Gate 3: RMSE / target_mean > 0.25 (if target_mean is non-zero)
    if abs(target_mean) > 1e-6:
        relative_rmse = rmse / abs(target_mean)
        if relative_rmse > 0.25:
            failed_gates.append(f"RMSE ({rmse:.3f}) / target_mean ({target_mean:.3f}) = {relative_rmse:.3f} > 0.25")
    
    passed = len(failed_gates) == 0
    return passed, failed_gates


def detect_target_transformation_need(target_stats: Dict[str, float]) -> Optional[str]:
    """
    Detect if target transformation is needed based on skewness and outliers.
    
    Returns:
        Suggested transformation: "log", "quantile", "robust", or None
    """
    skew = target_stats.get("skew", 0.0)
    min_val = target_stats.get("min", 0.0)
    max_val = target_stats.get("max", 1.0)
    
    # High positive skew (> 2) suggests log transform
    if skew > 2.0 and min_val > 0:
        return "log"
    
    # Extreme skew (> 3) or very wide range suggests quantile transform
    if abs(skew) > 3.0:
        return "quantile"
    
    # Moderate skew with outliers suggests robust scaling
    if abs(skew) > 1.5:
        return "robust"
    
    return None


def analyze_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Tuple[bool, str]:
    """
    Analyze residuals for heteroscedastic failure.
    
    Returns:
        (is_heteroscedastic: bool, description: str)
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    
    # Remove NaNs
    mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    
    if len(y_true_arr) < 10:
        return False, "Insufficient data for residual analysis"
    
    residuals = y_true_arr - y_pred_arr
    
    # Split into quantiles
    q25_idx = int(len(y_pred_arr) * 0.25)
    q75_idx = int(len(y_pred_arr) * 0.75)
    
    sorted_indices = np.argsort(y_pred_arr)
    lower_residuals = residuals[sorted_indices[:q25_idx]]
    upper_residuals = residuals[sorted_indices[q75_idx:]]
    
    lower_std = np.std(lower_residuals) if len(lower_residuals) > 1 else 0.0
    upper_std = np.std(upper_residuals) if len(upper_residuals) > 1 else 0.0
    
    # Check if variance explodes in upper or lower quantiles
    if lower_std > 0 and upper_std > 0:
        variance_ratio = max(lower_std, upper_std) / min(lower_std, upper_std)
        if variance_ratio > 2.0:
            return True, f"Heteroscedastic failure: residual variance ratio = {variance_ratio:.2f}"
    
    return False, "Residuals appear homoscedastic"


def detect_variance_fit_illusion(
    metrics: Dict[str, float],
    target_stats: Dict[str, float]
) -> Tuple[bool, str]:
    """
    Detect RandomForest variance-fit illusion.
    
    If R² is high but normalized errors are high, the model is
    explaining variance but not making usable predictions.
    
    Returns:
        (is_illusion: bool, description: str)
    """
    r2 = metrics.get("R²", -float('inf'))
    normalized_rmse = metrics.get("normalized_RMSE", float('inf'))
    normalized_mae = metrics.get("normalized_MAE", float('inf'))
    
    # High R² but high normalized errors = illusion
    if r2 > 0.7 and (normalized_rmse > 0.5 or normalized_mae > 0.5):
        return True, f"Variance-fit illusion: R² = {r2:.3f} but normalized_RMSE = {normalized_rmse:.3f}, normalized_MAE = {normalized_mae:.3f}"
    
    return False, "No variance-fit illusion detected"
