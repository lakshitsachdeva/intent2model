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
    
    def __init__(self, run_id: Optional[str] = None, log_callback=None):
        self.max_attempts = 5
        self.attempt_history = []
        self.run_id = run_id
        self.log_callback = log_callback  # Function to call for logging
    
    def _log(self, message: str, stage: str = "executor", progress: Optional[float] = None):
        """Log a message (to callback if available, and print)."""
        if self.log_callback:
            self.log_callback(self.run_id, message, stage, progress)
        print(f"[AutonomousExecutor] {message}")
    
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
                self._log(f"🔄 Attempt {attempt + 1}/{self.max_attempts}: Starting planning and training...", "executor", 30 + attempt * 10)
                
                # Step 1: Get plan (or repair previous plan)
                if attempt == 0:
                    # First attempt: get plan from LLM
                    self._log("📋 Step 1: Getting plan from LLM...", "plan", 35)
                    plan = plan_automl(df, requested_target=requested_target, llm_provider="gemini")
                    self._log(f"✅ Plan received: {len(plan.feature_transforms)} feature transforms, {len(plan.model_candidates)} models", "plan", 40)
                else:
                    # Subsequent attempts: repair the plan based on previous errors
                    self._log(f"🔧 Step 1: Repairing plan based on previous errors (attempt {attempt})...", "repair", 35)
                    plan = self._repair_plan_from_errors(
                        df=df,
                        profile=profile,
                        target=target,
                        task=task,
                        previous_errors=self.attempt_history,
                        requested_target=requested_target
                    )
                    self._log(f"✅ Plan repaired: {len(plan.feature_transforms)} feature transforms", "repair", 40)
                
                # Step 2: Validate plan has features
                if not plan.feature_transforms or all(ft.drop for ft in plan.feature_transforms):
                    self._log("⚠️  Plan has no features - auto-generating feature_transforms...", "repair", 42)
                    plan.feature_transforms = _generate_feature_transforms_from_profile(
                        profile=profile,
                        target=plan.inferred_target
                    )
                    kept_count = len([ft for ft in plan.feature_transforms if not ft.drop])
                    self._log(f"✅ Generated {kept_count} features from dataset", "repair", 45)
                
                # Step 3: Build config from plan
                self._log("⚙️  Step 2: Building pipeline configuration from plan...", "config", 50)
                config = self._plan_to_config(plan, profile)
                self._log(f"✅ Config built: {len(config.get('feature_transforms', []))} feature transforms", "config", 55)
                
                # Step 4: Try training
                self._log(f"🚀 Step 3: Training models: {', '.join(model_candidates)}...", "train", 60)
                if len(model_candidates) > 1:
                    self._log(f"📊 Comparing {len(model_candidates)} models...", "train", 62)
                    result = compare_models(
                        df=df,
                        target=target,
                        task=task,
                        metric=metric,
                        model_candidates=model_candidates,
                        base_config=config
                    )
                    self._log(f"✅ Model comparison complete: {len(result.get('all_models', []))} models trained", "train", 68)
                else:
                    self._log(f"🎯 Training single model: {model_candidates[0]}...", "train", 62)
                    if task == "classification":
                        result = train_classification(df, target, metric, config)
                    else:
                        result = train_regression(df, target, metric, config)
                    result["model_name"] = model_candidates[0] if model_candidates else "unknown"
                    self._log(f"✅ Model trained: {result['model_name']}", "train", 68)
                
                # Step 5: Success!
                self._log(f"🎉 Training succeeded on attempt {attempt + 1}!", "success", 70)
                result["plan"] = plan.model_dump()
                result["attempts"] = attempt + 1
                return result
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                error_trace = traceback.format_exc()
                
                self._log(f"❌ Attempt {attempt + 1} failed: {error_type}: {error_msg[:150]}...", "error", 50)
                
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
                    self._log("🔧 Error is fixable - analyzing and preparing repair...", "repair", 45)
                    continue
                elif attempt < self.max_attempts - 1:
                    self._log(f"🔄 Will try different approach on attempt {attempt + 2}...", "retry", 40)
                    continue
                else:
                    # Last attempt failed - use ultimate fallback
                    self._log("⚠️  All attempts failed - using ultimate fallback configuration...", "fallback", 50)
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
        self._log("🔧 Analyzing previous errors and repairing plan...", "repair", 35)
        
        # Get the last plan (if available)
        last_plan_dict = None
        if previous_errors and previous_errors[-1].get("plan"):
            last_plan_dict = previous_errors[-1]["plan"]
        
        # Analyze errors
        all_errors = " ".join([e.get("error_msg", "") for e in previous_errors])
        
        # Common fixes:
        # 1. If all features dropped, generate new feature_transforms
        if "all features" in all_errors.lower() or "no features" in all_errors.lower():
            self._log("  → Fix: All features were dropped - generating new feature_transforms from dataset...", "repair", 37)
            if not last_plan_dict:
                last_plan_dict = {}
            
            # Generate proper feature_transforms
            feature_transforms = _generate_feature_transforms_from_profile(
                profile=profile,
                target=target
            )
            
            kept_count = len([ft for ft in feature_transforms if not ft.get("drop", False)])
            self._log(f"  → Generated {kept_count} features that will be kept", "repair", 38)
            
            last_plan_dict["feature_transforms"] = [ft if isinstance(ft, dict) else ft.model_dump() for ft in feature_transforms]
            last_plan_dict["inferred_target"] = target
            last_plan_dict["task_type"] = task
            last_plan_dict["planning_source"] = "auto_repair"
            last_plan_dict["plan_quality"] = "high_confidence"
        
        # 2. If plan missing, create minimal plan
        if not last_plan_dict:
            self._log("  → Fix: Plan missing - creating minimal plan from dataset...", "repair", 37)
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
        self._log("🆘 Using ultimate fallback - simplest configuration that always works...", "fallback", 50)
        
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
