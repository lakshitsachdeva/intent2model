"""
Pipeline Validator - Compiler Invariant Checks

Validates compiled pipelines before training to ensure they are safe and executable.
Fails loudly with precise error messages.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def validate_pipeline_before_training(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    feature_transforms: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Validate pipeline invariants before training.
    
    Raises RuntimeError with precise message if validation fails.
    Returns diagnostic info if validation passes.
    """
    errors = []
    warnings = []
    
    # 1. Check X has at least ONE feature
    if X.empty:
        errors.append("X is empty - no features available for training")
    elif X.shape[1] == 0:
        errors.append("X has 0 columns - all features were dropped")
    
    # 2. Check feature_transforms don't drop all features
    if feature_transforms:
        dropped_count = sum(1 for ft in feature_transforms if ft.get("drop", False))
        remaining_count = len(feature_transforms) - dropped_count
        if remaining_count == 0:
            errors.append(
                f"Plan drops ALL features ({dropped_count} dropped, 0 remaining). "
                "This would result in an empty feature matrix. "
                "Please revise the plan to keep at least one feature."
            )
        elif remaining_count < X.shape[1]:
            warnings.append(
                f"Plan drops {dropped_count} features, leaving {remaining_count} features. "
                f"Original dataset had {X.shape[1]} features."
            )
    
    # 3. Check for NaNs in X (after preprocessing, but we can check input)
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        nan_cols = nan_counts[nan_counts > 0].to_dict()
        warnings.append(
            f"Input features contain NaNs: {nan_cols}. "
            "Ensure imputation is configured in feature_transforms."
        )
    
    # 4. Validate preprocessor exists and has transformers
    if "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        if isinstance(preprocessor, ColumnTransformer):
            transformers = preprocessor.transformers
            if not transformers:
                errors.append(
                    "Preprocessor has NO transformers. "
                    "This means no features will be processed. "
                    "Check that feature_transforms includes at least one non-dropped feature."
                )
            else:
                # Check that transformers have columns
                empty_transformers = []
                for name, transformer, cols in transformers:
                    if not cols or len(cols) == 0:
                        empty_transformers.append(name)
                if empty_transformers:
                    warnings.append(
                        f"Some transformers have no columns: {empty_transformers}. "
                        "They will be skipped during transformation."
                    )
        else:
            warnings.append(f"Preprocessor is not a ColumnTransformer: {type(preprocessor)}")
    else:
        errors.append("Pipeline missing 'preprocessor' step")
    
    # 5. Validate model exists
    if "model" not in pipeline.named_steps:
        errors.append("Pipeline missing 'model' step")
    
    # 6. Check y alignment
    if len(y) != len(X):
        errors.append(
            f"Target y length ({len(y)}) does not match features X length ({len(X)})"
        )
    
    # 7. Check y has no NaNs
    if y.isna().sum() > 0:
        errors.append(
            f"Target y contains {y.isna().sum()} NaN values. "
            "Target must be complete (no missing values)."
        )
    
    # 8. Try to infer output feature count (dry run if possible)
    try:
        # Get preprocessor
        if "preprocessor" in pipeline.named_steps:
            preprocessor = pipeline.named_steps["preprocessor"]
            if isinstance(preprocessor, ColumnTransformer):
                # Try to get feature names (if available)
                try:
                    # Fit on a small sample to check output shape
                    sample_X = X.head(10)
                    preprocessor.fit(sample_X)
                    if hasattr(preprocessor, "transform"):
                        X_transformed = preprocessor.transform(sample_X)
                        if X_transformed.shape[1] == 0:
                            errors.append(
                                "Preprocessor produces 0 output features. "
                                "This means all features were dropped or filtered out."
                            )
                        else:
                            warnings.append(
                                f"Preprocessor produces {X_transformed.shape[1]} output features "
                                f"from {X.shape[1]} input features."
                            )
                except Exception as e:
                    warnings.append(f"Could not validate preprocessor output shape: {e}")
    except Exception as e:
        warnings.append(f"Could not perform dry-run validation: {e}")
    
    # Fail loudly if critical errors
    if errors:
        error_msg = "COMPILER VALIDATION FAILED:\n\n"
        error_msg += "Critical Errors:\n"
        for i, err in enumerate(errors, 1):
            error_msg += f"  {i}. {err}\n"
        
        if warnings:
            error_msg += "\nWarnings:\n"
            for i, warn in enumerate(warnings, 1):
                error_msg += f"  {i}. {warn}\n"
        
        error_msg += "\nThis error occurred during pipeline compilation, NOT during model training."
        error_msg += "\nThe compiled pipeline is invalid and cannot be executed."
        
        raise RuntimeError(error_msg)
    
    # Return diagnostic info
    diagnostic = {
        "valid": True,
        "warnings": warnings,
        "input_features": X.shape[1],
        "input_samples": X.shape[0],
        "target_samples": len(y),
    }
    
    # Try to get output feature count
    try:
        if "preprocessor" in pipeline.named_steps:
            preprocessor = pipeline.named_steps["preprocessor"]
            if isinstance(preprocessor, ColumnTransformer):
                sample_X = X.head(10)
                preprocessor.fit(sample_X)
                X_transformed = preprocessor.transform(sample_X)
                diagnostic["output_features"] = X_transformed.shape[1]
    except Exception:
        pass
    
    return diagnostic


def validate_feature_transforms(
    feature_transforms: List[Dict[str, Any]],
    available_columns: List[str],
    target: str
) -> Dict[str, Any]:
    """
    Validate feature_transforms before compilation.
    
    Checks:
    - All feature names exist in dataset
    - At least one feature is not dropped
    - Target is marked for dropping
    """
    errors = []
    warnings = []
    
    if not feature_transforms:
        errors.append("feature_transforms is empty - no preprocessing decisions available")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    # Check all names exist
    missing_features = []
    dropped_features = []
    kept_features = []
    
    for ft in feature_transforms:
        name = ft.get("name")
        if not name:
            warnings.append("Found feature_transform without 'name' field - skipping")
            continue
        
        if name not in available_columns:
            missing_features.append(name)
            continue
        
        if ft.get("drop", False):
            dropped_features.append(name)
        else:
            kept_features.append(name)
    
    if missing_features:
        errors.append(
            f"feature_transforms references features not in dataset: {missing_features}"
        )
    
    # Check target is dropped
    if target not in dropped_features:
        warnings.append(
            f"Target '{target}' is not marked for dropping in feature_transforms. "
            "This may cause leakage."
        )
    
    # Check at least one feature is kept
    if len(kept_features) == 0:
        errors.append(
            f"All {len(dropped_features)} features are marked for dropping. "
            "At least one feature must be kept for training."
        )
    
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "kept_features": kept_features,
        "dropped_features": dropped_features,
        "kept_count": len(kept_features),
        "dropped_count": len(dropped_features),
    }
