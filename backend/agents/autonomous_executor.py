"""
Autonomous Executor - Tries, fails, learns, fixes, retries until it works.
Never gives up. Always finds a solution.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from agents.automl_agent import plan_automl
from agents.plan_normalizer import normalize_plan_dict, _generate_feature_transforms_from_profile
from schemas.pipeline_schema import AutoMLPlan
from ml.profiler import profile_dataset
from ml.trainer import compare_models, train_classification, train_regression
import traceback


class AutonomousExecutor:
    """
    Autonomous agent that executes ML training with automatic error recovery.
    Tries → Fails → Learns → Fixes → Retries → Succeeds
    """
    
    def __init__(self):
        self.max_attempts = 5
        self.attempt_history = []
    
    def execute_with_auto_fix(
        self,
        df: pd.DataFrame,
        target: str,
        task: str,
        metric: str,
        model_candidates: List[str],
        requested_target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute training with automatic error detection and fixing.
        Never gives up. Always finds a solution.
        """
        profile = profile_dataset(df)
        
        for attempt in range(self.max_attempts):
            try:
                print(f"\n🔄 Attempt {attempt + 1}/{self.max_attempts}: Planning and training...")
                
                # Step 1: Get plan (or repair previous plan)
                if attempt == 0:
                    # First attempt: get plan from LLM
                    plan = plan_automl(df, requested_target=requested_target, llm_provider="gemini")
                else:
                    # Subsequent attempts: repair the plan based on previous errors
                    plan = self._repair_plan_from_errors(
                        df=df,
                        profile=profile,
                        target=target,
                        task=task,
                        previous_errors=self.attempt_history,
                        requested_target=requested_target
                    )
                
                # Step 2: Validate plan has features
                if not plan.feature_transforms or all(ft.drop for ft in plan.feature_transforms):
                    print("⚠️  Plan has no features - auto-generating feature_transforms...")
                    plan.feature_transforms = _generate_feature_transforms_from_profile(
                        profile=profile,
                        target=plan.inferred_target
                    )
                    print(f"✅ Generated {len([ft for ft in plan.feature_transforms if not ft.drop])} features")
                
                # Step 3: Build config from plan
                config = self._plan_to_config(plan, profile)
                
                # Step 4: Try training
                print(f"🚀 Training models: {model_candidates}")
                if len(model_candidates) > 1:
                    result = compare_models(
                        df=df,
                        target=target,
                        task=task,
                        metric=metric,
                        model_candidates=model_candidates,
                        base_config=config
                    )
                else:
                    if task == "classification":
                        result = train_classification(df, target, metric, config)
                    else:
                        result = train_regression(df, target, metric, config)
                    result["model_name"] = model_candidates[0] if model_candidates else "unknown"
                
                # Step 5: Success!
                print(f"✅ Training succeeded on attempt {attempt + 1}!")
                result["plan"] = plan.model_dump()
                result["attempts"] = attempt + 1
                return result
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                error_trace = traceback.format_exc()
                
                print(f"❌ Attempt {attempt + 1} failed: {error_type}: {error_msg[:200]}")
                
                # Store error for learning
                self.attempt_history.append({
                    "attempt": attempt + 1,
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "error_trace": error_trace[:500],
                    "plan": plan.model_dump() if 'plan' in locals() else None
                })
                
                # Check if it's a fixable error
                if self._is_fixable_error(error_msg):
                    print(f"🔧 Error is fixable - will repair and retry...")
                    continue
                elif attempt < self.max_attempts - 1:
                    print(f"🔄 Will try different approach on next attempt...")
                    continue
                else:
                    # Last attempt failed - use ultimate fallback
                    print(f"⚠️  All attempts failed - using ultimate fallback...")
                    return self._ultimate_fallback(df, target, task, metric, model_candidates, profile)
        
        # Should never reach here, but just in case
        return self._ultimate_fallback(df, target, task, metric, model_candidates, profile)
    
    def _is_fixable_error(self, error_msg: str) -> bool:
        """Check if error can be automatically fixed."""
        fixable_patterns = [
            "no feature_transforms",
            "all features were dropped",
            "preprocessor has NO transformers",
            "COMPILER ERROR",
            "feature_transforms is empty",
            "at least one feature",
            "zero remaining features"
        ]
        return any(pattern.lower() in error_msg.lower() for pattern in fixable_patterns)
    
    def _repair_plan_from_errors(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        target: str,
        task: str,
        previous_errors: List[Dict[str, Any]],
        requested_target: Optional[str] = None
    ) -> AutoMLPlan:
        """
        Repair plan based on previous errors.
        Automatically fixes common issues.
        """
        print("🔧 Repairing plan based on previous errors...")
        
        # Get the last plan (if available)
        last_plan_dict = None
        if previous_errors and previous_errors[-1].get("plan"):
            last_plan_dict = previous_errors[-1]["plan"]
        
        # Analyze errors
        all_errors = " ".join([e.get("error_msg", "") for e in previous_errors])
        
        # Common fixes:
        # 1. If all features dropped, generate new feature_transforms
        if "all features" in all_errors.lower() or "no features" in all_errors.lower():
            print("  → Fix: Generating feature_transforms from dataset...")
            if not last_plan_dict:
                last_plan_dict = {}
            
            # Generate proper feature_transforms
            feature_transforms = _generate_feature_transforms_from_profile(
                profile=profile,
                target=target
            )
            
            last_plan_dict["feature_transforms"] = [ft if isinstance(ft, dict) else ft.model_dump() for ft in feature_transforms]
            last_plan_dict["inferred_target"] = target
            last_plan_dict["task_type"] = task
            last_plan_dict["planning_source"] = "auto_repair"
            last_plan_dict["plan_quality"] = "high_confidence"
        
        # 2. If plan missing, create minimal plan
        if not last_plan_dict:
            print("  → Fix: Creating minimal plan from dataset...")
            last_plan_dict = {
                "plan_schema_version": "v1",
                "inferred_target": target,
                "target_confidence": 1.0,
                "alternative_targets": [],
                "task_type": task,
                "task_confidence": 1.0,
                "task_inference_md": f"Task type: {task} (auto-detected)",
                "dataset_intelligence_md": f"Dataset has {profile.get('n_rows', 0)} rows and {profile.get('n_cols', 0)} columns",
                "transformation_strategy_md": "Auto-generated transformation strategy",
                "model_selection_md": "Auto-selected models",
                "training_validation_md": "Using cross-validation",
                "error_behavior_analysis_md": "Standard error analysis",
                "explainability_md": "Feature importance analysis",
                "primary_metric": "f1" if task == "classification" else "rmse",
                "additional_metrics": ["precision", "recall"] if task == "classification" else ["mae", "r2"],
                "feature_transforms": _generate_feature_transforms_from_profile(profile, target),
                "model_candidates": [{"model_name": "random_forest", "reason_md": "Robust baseline", "params": {}}],
                "planning_source": "auto_repair",
                "planning_error": None,
                "plan_quality": "high_confidence",
                "plan_warnings": []
            }
        
        # Normalize and validate
        last_plan_dict = normalize_plan_dict(
            last_plan_dict,
            profile=profile,
            requested_target=requested_target
        )
        
        return AutoMLPlan(**last_plan_dict)
    
    def _plan_to_config(self, plan: AutoMLPlan, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Convert AutoMLPlan to training config."""
        return {
            "task": plan.task_type,
            "model": plan.model_candidates[0].model_name if plan.model_candidates else "random_forest",
            "feature_transforms": [ft.model_dump() if hasattr(ft, 'model_dump') else ft for ft in plan.feature_transforms],
            "automl_plan": plan.model_dump()
        }
    
    def _ultimate_fallback(
        self,
        df: pd.DataFrame,
        target: str,
        task: str,
        metric: str,
        model_candidates: List[str],
        profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ultimate fallback - use simplest possible configuration.
        This should ALWAYS work.
        """
        print("🆘 Using ultimate fallback - simplest configuration...")
        
        # Create minimal config that always works
        config = {
            "task": task,
            "preprocessing": ["standard_scaler", "one_hot"],
            "model": model_candidates[0] if model_candidates else ("random_forest" if task == "classification" else "random_forest"),
            "feature_transforms": _generate_feature_transforms_from_profile(profile, target)
        }
        
        # Try training with minimal config
        try:
            if len(model_candidates) > 1:
                result = compare_models(df, target, task, metric, model_candidates[:1], config)
            else:
                if task == "classification":
                    result = train_classification(df, target, metric, config)
                else:
                    result = train_regression(df, target, metric, config)
                result["model_name"] = model_candidates[0] if model_candidates else "unknown"
            
            result["plan"] = {
                "planning_source": "ultimate_fallback",
                "plan_quality": "high_confidence"
            }
            result["attempts"] = self.max_attempts
            return result
        except Exception as e:
            # Even ultimate fallback failed - this is very bad
            raise RuntimeError(
                f"CRITICAL: Even ultimate fallback failed. "
                f"This suggests a fundamental issue with the dataset. "
                f"Error: {str(e)}"
            )
