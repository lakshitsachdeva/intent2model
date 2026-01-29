"""
Autonomous Executor - Self-Healing, Epistemically Honest ML Engineer.

Tries → Fails → Diagnoses → Fixes → Retries → Succeeds OR Refuses

This is NOT AutoML. This is an ML ENGINEER THAT CAN DEBUG ITSELF.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from agents.automl_agent import plan_automl
from agents.plan_normalizer import normalize_plan_dict, _generate_feature_transforms_from_profile
from agents.diagnosis_agent import DiagnosisAgent
from agents.error_gating import (
    compute_target_stats,
    compute_normalized_metrics,
    check_regression_error_gates,
    detect_target_transformation_need,
    analyze_residuals,
    detect_variance_fit_illusion
)
from schemas.pipeline_schema import AutoMLPlan
from schemas.failure_schema import FailureReport
from ml.profiler import profile_dataset
from ml.trainer import compare_models, train_classification, train_regression
import traceback
import numpy as np


class AutonomousExecutor:
    """
    Autonomous agent that executes ML training with automatic error recovery.
    Tries → Fails → Learns → Fixes → Retries → Succeeds
    """
    
    def __init__(self, run_id: Optional[str] = None, log_callback=None, llm_provider: str = "gemini"):
        self.max_attempts = 5
        self.attempt_history = []
        self.run_id = run_id
        self.log_callback = log_callback  # Function to call for logging
        self.llm_provider = llm_provider
        self.diagnosis_agent = DiagnosisAgent(llm_provider=llm_provider)
        self.failure_history = []  # Track all failures for diagnosis
    
    def _log(
        self,
        message: str,
        stage: str = "executor",
        progress: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        attempt_count: Optional[int] = None,
    ):
        """Log a message; optional payload/attempt_count for agent event timeline (GET /runs/{run_id})."""
        if self.log_callback:
            try:
                self.log_callback(
                    self.run_id, message, stage, progress,
                    payload=payload, attempt_count=attempt_count,
                )
            except TypeError:
                self.log_callback(self.run_id, message, stage, progress)
        print(f"[AutonomousExecutor] {message}")
    
    def execute_with_auto_fix(
        self,
        df: pd.DataFrame,
        target: str,
        task: str,
        metric: str,
        model_candidates: List[str],
        requested_target: Optional[str] = None,
        llm_provider: str = "gemini",
    ) -> Dict[str, Any]:
        """
        Execute training with automatic error detection and fixing.
        Never gives up. Always finds a solution.
        """
        profile = profile_dataset(df)
        
        for attempt in range(self.max_attempts):
            try:
                self._log(
                    f"🔄 Attempt {attempt + 1}/{self.max_attempts}: Starting planning and training...",
                    "executor", 30 + attempt * 10, attempt_count=attempt + 1,
                )
                
                # Step 1: Get plan (or repair previous plan)
                if attempt == 0:
                    # First attempt: get plan from LLM
                    self._log("📋 Step 1: Getting plan from LLM...", "plan", 35)
                    plan = plan_automl(df, requested_target=requested_target, llm_provider=llm_provider)
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
                        requested_target=requested_target,
                        llm_provider=llm_provider,
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
                # CRITICAL: execution-side task is authoritative; don't let plan.task_type drift override it.
                # plan.task_type is from LLM and may be wrong; config.task controls model family selection.
                config["task"] = task
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
                
                # Step 5: Evaluate (check error gates for regression)
                if task == "regression":
                    self._log("🔍 Evaluating model quality (error gates)...", "evaluate", 70)
                    evaluation_passed, failure_info = self._evaluate_regression_model(
                        df=df,
                        target=target,
                        result=result,
                        plan=plan
                    )
                    
                    if not evaluation_passed:
                        # Evaluation failed - create failure report and diagnose
                        self._log(
                            "❌ Model failed error gates - creating failure report...",
                            "evaluate", 72,
                            payload={"failed_gates": failure_info.get("failed_gates", []), "stage": "metric_gate_failed"},
                        )
                        failure_report = self._create_failure_report(
                            failure_stage="evaluation",
                            failed_gates=failure_info["failed_gates"],
                            metrics=result.get("metrics", {}),
                            target_stats=failure_info.get("target_stats", {}),
                            feature_summary=self._get_feature_summary(df, plan, config),
                            model_used=result.get("model_name", "unknown"),
                            model_hyperparameters={},
                            previous_plan=plan.model_dump(),
                            error_message=failure_info.get("error_message", "Error gates failed"),
                            attempt_number=attempt + 1
                        )
                        
                        # Diagnose with LLM
                        self._log("🧠 Diagnosing failure with LLM...", "diagnose", 75)
                        diagnosis = self.diagnosis_agent.diagnose_failure(failure_report)
                        self._log(
                            f"✅ Diagnosis received (confidence: {diagnosis.recovery_confidence:.2f})",
                            "diagnose", 78,
                            payload={"recovery_confidence": diagnosis.recovery_confidence, "plan_changes": getattr(diagnosis, "plan_changes", [])},
                        )
                        
                        # Check if we should stop
                        if diagnosis.suggested_stop or diagnosis.recovery_confidence < 0.3:
                            self._log("🛑 LLM suggests stopping - task may not be learnable", "refuse", 80)
                            return self._create_refusal_result(
                                plan=plan,
                                failure_report=failure_report,
                                diagnosis=diagnosis,
                                attempts=attempt + 1
                            )
                        
                        # Apply plan changes for next attempt
                        self._log("🔧 Applying LLM-suggested plan changes...", "repair", 80)
                        plan = self._apply_diagnosis_changes(plan, diagnosis, df, target, profile)
                        self.failure_history.append({
                            "attempt": attempt + 1,
                            "failure_report": failure_report.model_dump(),
                            "diagnosis": diagnosis.model_dump()
                        })
                        continue
                    else:
                        self._log("✅ Model passed error gates!", "evaluate", 75)
                
                # Step 6: Success!
                self._log(f"🎉 Training succeeded on attempt {attempt + 1}!", "success", 80)
                result["plan"] = plan.model_dump()
                result["attempts"] = attempt + 1
                result["failure_history"] = self.failure_history
                return result
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                error_trace = traceback.format_exc()
                
                # Determine failure stage
                if "COMPILER ERROR" in error_msg or "compiler" in error_msg.lower():
                    failure_stage = "compiler"
                elif "TRAINING REFUSED" in error_msg or "error gates" in error_msg.lower():
                    failure_stage = "evaluation"
                else:
                    failure_stage = "training"

                # Heuristic auto-fix: if task is regression but errors indicate classification mismatch,
                # flip task to classification for next attempt.
                if task == "regression":
                    mismatch_signals = [
                        "unknown regression model: logistic_regression",
                        "unknown regression model: naive_bayes",
                        "could not convert string to float",
                    ]
                    if any(sig in error_msg.lower() for sig in mismatch_signals):
                        self._log("🔧 Detected regression/classification mismatch — switching task to classification", "repair", 45)
                        task = "classification"
                        metric = "accuracy"
                        # Keep only classification-safe models
                        model_candidates = [m for m in model_candidates if m in ["logistic_regression", "random_forest", "svm", "naive_bayes", "gradient_boosting"]]
                        if not model_candidates:
                            model_candidates = ["logistic_regression", "random_forest"]
                        self._log(f"✅ Updated task={task}, metric={metric}, models={model_candidates}", "repair", 48)
                
                self._log(
                    f"❌ Attempt {attempt + 1} failed ({failure_stage}): {error_type}: {error_msg[:150]}...",
                    "error", 50,
                    payload={"error_type": error_type, "failure_stage": failure_stage, "message_preview": error_msg[:300]},
                    attempt_count=attempt + 1,
                )
                
                # Store error for learning
                error_record = {
                    "attempt": attempt + 1,
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "error_trace": error_trace[:500],
                    "plan": plan.model_dump() if 'plan' in locals() else None,
                    "failure_stage": failure_stage
                }
                self.attempt_history.append(error_record)
                
                # For evaluation failures, create failure report and diagnose
                if failure_stage == "evaluation" or "TRAINING REFUSED" in error_msg:
                    try:
                        # Extract metrics from error message or result if available
                        metrics = {}
                        target_stats = {}
                        if 'result' in locals():
                            metrics = result.get("metrics", {})
                            if "_target_stats" in metrics:
                                target_stats = metrics.pop("_target_stats")
                        
                        failure_report = self._create_failure_report(
                            failure_stage=failure_stage,
                            failed_gates=metrics.get("_failed_gates", [error_msg]),
                            metrics=metrics,
                            target_stats=target_stats,
                            feature_summary=self._get_feature_summary(df, plan, config) if 'plan' in locals() and 'config' in locals() else {},
                            model_used=result.get("model_name", "unknown") if 'result' in locals() else "unknown",
                            model_hyperparameters={},
                            previous_plan=plan.model_dump() if 'plan' in locals() else None,
                            error_message=error_msg,
                            error_traceback=error_trace[:1000],
                            attempt_number=attempt + 1
                        )
                        
                        # Diagnose with LLM
                        self._log("🧠 Diagnosing failure with LLM...", "diagnose", 52)
                        diagnosis = self.diagnosis_agent.diagnose_failure(failure_report)
                        self._log(f"✅ Diagnosis received (confidence: {diagnosis.recovery_confidence:.2f})", "diagnose", 55)
                        
                        # Check if we should stop
                        if diagnosis.suggested_stop or diagnosis.recovery_confidence < 0.3:
                            self._log("🛑 LLM suggests stopping - task may not be learnable", "refuse", 60)
                            return self._create_refusal_result(
                                plan=plan if 'plan' in locals() else None,
                                failure_report=failure_report,
                                diagnosis=diagnosis,
                                attempts=attempt + 1
                            )
                        
                        # Apply plan changes for next attempt
                        if 'plan' in locals():
                            self._log("🔧 Applying LLM-suggested plan changes...", "repair", 55)
                            plan = self._apply_diagnosis_changes(plan, diagnosis, df, target, profile)
                            self.failure_history.append({
                                "attempt": attempt + 1,
                                "failure_report": failure_report.model_dump(),
                                "diagnosis": diagnosis.model_dump()
                            })
                            continue
                    except Exception as diag_error:
                        self._log(f"⚠️  Diagnosis failed: {str(diag_error)[:100]}...", "diagnose", 52)
                
                # Check if it's a fixable error
                if self._is_fixable_error(error_msg):
                    self._log("🔧 Error is fixable - analyzing and preparing repair...", "repair", 45)
                    continue
                elif attempt < self.max_attempts - 1:
                    self._log(
                        f"🔄 Will try different approach on attempt {attempt + 2}...",
                        "retry", 40, attempt_count=attempt + 2,
                    )
                    continue
                else:
                    # Last attempt failed - use ultimate fallback
                    self._log("⚠️  All attempts failed - using ultimate fallback configuration...", "fallback", 50)
                    return self._ultimate_fallback(df, target, task, metric, model_candidates, profile)
        
        # Should never reach here, but just in case
        return self._ultimate_fallback(df, target, task, metric, model_candidates, profile)
    
    def _evaluate_regression_model(
        self,
        df: pd.DataFrame,
        target: str,
        result: Dict[str, Any],
        plan: AutoMLPlan
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate regression model against error gates.
        
        Returns:
            (passed: bool, failure_info: Dict)
        """
        pipeline = result.get("best_model")
        if pipeline is None:
            return True, {}  # No model to evaluate
        
        X = df.drop(columns=[target])
        y = df[target]
        
        # Get predictions
        y_pred = pipeline.predict(X)
        
        # Compute target stats and normalized metrics
        target_stats = compute_target_stats(y)
        metrics = compute_normalized_metrics(y, y_pred, target_stats)
        
        # Check error gates
        gates_passed, failed_gates = check_regression_error_gates(metrics, target_stats)
        
        # Check for variance-fit illusion
        is_illusion, illusion_desc = detect_variance_fit_illusion(metrics, target_stats)
        
        # Analyze residuals
        is_heteroscedastic, residual_desc = analyze_residuals(y, y_pred)
        
        failure_info = {
            "failed_gates": failed_gates,
            "target_stats": target_stats,
            "metrics": metrics,
            "is_variance_fit_illusion": is_illusion,
            "is_heteroscedastic": is_heteroscedastic,
        }
        
        if not gates_passed:
            error_msg = "TRAINING REFUSED: Absolute error unacceptable.\n"
            error_msg += "Failed gates:\n"
            for gate in failed_gates:
                error_msg += f"  - {gate}\n"
            if is_illusion:
                error_msg += f"\n⚠️ {illusion_desc}\n"
            if is_heteroscedastic:
                error_msg += f"\n⚠️ {residual_desc}\n"
            failure_info["error_message"] = error_msg
        
        # Update result metrics
        result["metrics"].update(metrics)
        result["metrics"]["_target_stats"] = target_stats
        result["metrics"]["_failed_gates"] = failed_gates
        
        return gates_passed, failure_info
    
    def _create_failure_report(
        self,
        failure_stage: str,
        failed_gates: List[str],
        metrics: Dict[str, float],
        target_stats: Dict[str, float],
        feature_summary: Dict[str, Any],
        model_used: str,
        model_hyperparameters: Dict[str, Any],
        previous_plan: Optional[Dict[str, Any]],
        error_message: str,
        error_traceback: Optional[str] = None,
        attempt_number: int = 1
    ) -> FailureReport:
        """Create structured failure report."""
        return FailureReport(
            failure_stage=failure_stage,
            failed_gates=failed_gates,
            metrics=metrics,
            target_stats=target_stats,
            feature_summary=feature_summary,
            model_used=model_used,
            model_hyperparameters=model_hyperparameters,
            previous_plan=previous_plan,
            error_message=error_message,
            error_traceback=error_traceback,
            attempt_number=attempt_number
        )
    
    def _get_feature_summary(
        self,
        df: pd.DataFrame,
        plan: AutoMLPlan,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get feature preprocessing summary."""
        feature_transforms = plan.feature_transforms
        kept_features = [ft.name for ft in feature_transforms if not ft.drop]
        dropped_features = [ft.name for ft in feature_transforms if ft.drop]
        
        return {
            "feature_count_before": len(df.columns) - 1,  # Exclude target
            "feature_count_after": len(kept_features),
            "dropped_features": dropped_features,
            "kept_features": kept_features,
            "encoded_features": [ft.name for ft in feature_transforms if ft.encode != "none"],
            "scaled_features": [ft.name for ft in feature_transforms if ft.scale != "none"],
        }
    
    def _apply_diagnosis_changes(
        self,
        plan: AutoMLPlan,
        diagnosis,
        df: pd.DataFrame,
        target: str,
        profile: Dict[str, Any]
    ) -> AutoMLPlan:
        """
        Apply LLM-suggested changes to the plan.
        """
        plan_changes = diagnosis.plan_changes
        
        # Apply target transformation
        if "target_transformation" in plan_changes:
            transform = plan_changes["target_transformation"]
            if transform:
                self._log(f"  → Applying target transformation: {transform}", "repair", 82)
                # Note: Target transformation is applied during training, not in plan
                # We'll store it in plan metadata for notebook generation
                if not hasattr(plan, "target_transformation"):
                    plan_dict = plan.model_dump()
                    plan_dict["target_transformation"] = transform
                    plan = AutoMLPlan(**plan_dict)
        
        # Apply feature transform changes
        if "feature_transforms" in plan_changes:
            changes = plan_changes["feature_transforms"]
            for change in changes:
                action = change.get("action")
                feature = change.get("feature")
                
                if action == "drop":
                    # Mark feature for dropping
                    for ft in plan.feature_transforms:
                        if ft.name == feature:
                            ft.drop = True
                            self._log(f"  → Dropping feature: {feature}", "repair", 83)
                
                elif action == "add_encoding":
                    encoding = change.get("encoding")
                    # Find or create feature transform
                    found = False
                    for ft in plan.feature_transforms:
                        if ft.name == feature:
                            ft.encode = encoding
                            found = True
                            self._log(f"  → Adding {encoding} encoding to: {feature}", "repair", 83)
                            break
                    
                    if not found:
                        # Create new feature transform
                        from schemas.pipeline_schema import FeatureTransform
                        new_ft = FeatureTransform(
                            name=feature,
                            inferred_dtype="object",
                            kind="nominal",
                            encode=encoding
                        )
                        plan.feature_transforms.append(new_ft)
                        self._log(f"  → Created new feature transform for: {feature}", "repair", 83)
        
        # Apply model selection change
        if "model_selection" in plan_changes:
            new_model = plan_changes["model_selection"]
            self._log(f"  → Changing model to: {new_model}", "repair", 84)
            if plan.model_candidates:
                plan.model_candidates[0].model_name = new_model
                plan.model_candidates[0].reason_md = f"Changed to {new_model} based on LLM diagnosis"
        
        # Apply evaluation metrics change
        if "evaluation_metrics" in plan_changes:
            new_metrics = plan_changes["evaluation_metrics"]
            if new_metrics:
                self._log(f"  → Changing primary metric to: {new_metrics[0]}", "repair", 85)
                plan_dict = plan.model_dump()
                plan_dict["primary_metric"] = new_metrics[0]
                plan_dict["additional_metrics"] = new_metrics[1:] if len(new_metrics) > 1 else []
                plan = AutoMLPlan(**plan_dict)
        
        # Mark plan as repaired
        plan_dict = plan.model_dump()
        plan_dict["planning_source"] = "auto_repair"
        plan_dict["planning_error"] = f"Repaired based on LLM diagnosis (confidence: {diagnosis.recovery_confidence:.2f})"
        plan = AutoMLPlan(**plan_dict)
        
        return plan
    
    def _create_refusal_result(
        self,
        plan: Optional[AutoMLPlan],
        failure_report: FailureReport,
        diagnosis,
        attempts: int
    ) -> Dict[str, Any]:
        """
        Create a result that represents a REFUSAL (epistemically honest failure).
        """
        if plan is None:
            # Create minimal plan for refusal
            plan_dict = {
                "plan_schema_version": "v1",
                "inferred_target": "unknown",
                "target_confidence": 0.0,
                "alternative_targets": [],
                "task_type": "regression",
                "task_confidence": 0.0,
                "task_inference_md": "Task inference failed",
                "dataset_intelligence_md": "Dataset analysis incomplete",
                "transformation_strategy_md": "Transformation strategy failed",
                "model_selection_md": "Model selection failed",
                "training_validation_md": "Training validation failed",
                "error_behavior_analysis_md": diagnosis.diagnosis_md,
                "explainability_md": "Explainability analysis unavailable",
                "primary_metric": "rmse",
                "additional_metrics": [],
                "feature_transforms": [],
                "model_candidates": [],
                "planning_source": "refusal",
                "planning_error": "Task refused - not reliably solvable",
                "plan_quality": "fallback_low_confidence",
            }
            plan = AutoMLPlan(**plan_dict)
        
        return {
            "refused": True,
            "refusal_reason": diagnosis.diagnosis_md,
            "failed_gates": failure_report.failed_gates,
            "metrics": failure_report.metrics,
            "target_stats": failure_report.target_stats,
            "plan": plan.model_dump(),
            "attempts": attempts,
            "failure_history": self.failure_history,
            "diagnosis": diagnosis.model_dump(),
        }
    
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
        requested_target: Optional[str] = None,
        llm_provider: str = "gemini",
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
            
            # IMPORTANT: notebook/compiler expects a schema-valid AutoMLPlan dict.
            # Ultimate fallback must still match schema exactly (marked as fallback).
            from agents.plan_normalizer import normalize_plan_dict
            fallback_plan = {
                "plan_schema_version": "v1",
                "inferred_target": target,
                "target_confidence": 1.0,
                "alternative_targets": [],
                "task_type": "regression" if task == "regression" else "binary_classification",
                "task_confidence": 1.0,
                "task_inference_md": f"Ultimate fallback used. Task='{task}' selected from request/execution context.",
                "dataset_intelligence_md": "Ultimate fallback (no LLM): dataset profiled execution-side.",
                "transformation_strategy_md": "Ultimate fallback (no LLM): conservative preprocessing based on profile.",
                "model_selection_md": f"Ultimate fallback: selected baseline model '{config.get('model')}'.",
                "training_validation_md": "Ultimate fallback: minimal robust training flow.",
                "error_behavior_analysis_md": "Ultimate fallback: basic error analysis.",
                "explainability_md": "Ultimate fallback: basic explainability if supported.",
                "primary_metric": metric or ("accuracy" if task == "classification" else "r2"),
                "additional_metrics": ["precision", "recall", "f1"] if task == "classification" else ["mae", "rmse"],
                "feature_transforms": config.get("feature_transforms") or [],
                "model_candidates": [{"model_name": config.get("model", "random_forest"), "reason_md": "Ultimate fallback baseline.", "params": {}}],
                "planning_source": "fallback",
                "planning_error": "ultimate_fallback",
                "plan_warnings": ["Ultimate fallback was used because planning/execution failed multiple times."],
                "plan_quality": "fallback_low_confidence",
            }
            fallback_plan = normalize_plan_dict(fallback_plan, profile=profile, requested_target=target)
            result["plan"] = fallback_plan
            result["attempts"] = self.max_attempts
            return result
        except Exception as e:
            # Even ultimate fallback failed - this is very bad
            raise RuntimeError(
                f"CRITICAL: Even ultimate fallback failed. "
                f"This suggests a fundamental issue with the dataset. "
                f"Error: {str(e)}"
            )
