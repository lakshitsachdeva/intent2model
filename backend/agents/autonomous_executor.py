"""
Autonomous Executor - Self-Healing, Epistemically Honest ML Engineer.

Tries → Fails → Diagnoses → Fixes → Retries → Succeeds OR Refuses

This is NOT AutoML. This is an ML ENGINEER THAT CAN DEBUG ITSELF.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
from agents.automl_agent import plan_automl
from agents.plan_normalizer import normalize_plan_dict, _generate_feature_transforms_from_profile
from agents.diagnosis_agent import DiagnosisAgent
from agents.error_gating import (
    compute_target_stats,
    compute_normalized_metrics,
    check_regression_error_gates,
    check_model_quality_minimum,
    detect_target_transformation_need,
    analyze_residuals,
    detect_variance_fit_illusion
)
from schemas.pipeline_schema import AutoMLPlan
from schemas.failure_schema import FailureReport
from ml.profiler import profile_dataset
from ml.trainer import compare_models, train_classification, train_regression
from ml.evaluator import prune_features_aggressive, infer_classification_primary_metric
from agents.execution_planner import (
    execution_plan_to_config,
    apply_repair_plan,
    plan_changes_to_repair_plan,
    merged_plan_dict,
)
from agents.plan_compiler import RefuseCodeGeneration, IncompleteExecutionPlan
from schemas.pipeline_schema import StructuralPlan, ExecutionPlan, RepairPlan
import traceback
import numpy as np

# Agentic loop: max retries (failure-driven re-plan)
AGENTIC_MAX_ATTEMPTS = 3


class AutonomousExecutor:
    """
    Autonomous agent that executes ML training with automatic error recovery.
    Tries → Fails → Learns → Fixes → Retries → Succeeds
    """
    
    def __init__(self, run_id: Optional[str] = None, log_callback=None, llm_provider: str = "gemini"):
        self.max_attempts = 10
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
        structural_plan: Optional[StructuralPlan] = None,
        first_execution_plan: Optional[ExecutionPlan] = None,
        user_constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute training with automatic error detection and fixing.
        When structural_plan and first_execution_plan are provided: TRUE AGENTIC LOOP
        (StructuralPlan once → ExecutionPlan per attempt → Compile → Train → Evaluate → Repair → Re-execute, max 2–3).
        Otherwise: legacy flow (plan_automl per attempt).
        """
        # Aggressive feature pruning (mandatory)
        df_pruned, dropped_cols = prune_features_aggressive(df, target, task)
        if dropped_cols:
            self._log(f"📉 Feature pruning dropped {len(dropped_cols)} features: {dropped_cols[:10]}{'...' if len(dropped_cols) > 10 else ''}", "config", 22)
            df = df_pruned
        profile = profile_dataset(df)

        # Classification: infer and LOCK primary metric
        locked_metric = metric
        if task == "classification":
            inferred = infer_classification_primary_metric(df, target)
            locked_metric = inferred
            if metric != inferred:
                self._log(f"🔒 Classification primary metric locked: {inferred} (was {metric})", "config", 24)
            metric = locked_metric

        use_agentic = structural_plan is not None and first_execution_plan is not None
        max_attempts = AGENTIC_MAX_ATTEMPTS if use_agentic else self.max_attempts
        current_execution_plan: Optional[ExecutionPlan] = first_execution_plan if use_agentic else None
        execution_plans_used: List[ExecutionPlan] = []  # agentic: all ExecutionPlans used (for notebook)
        # Chat/session can pass extended model_candidates (e.g. "try something stronger" → xgboost); use them when aggressive
        param_model_candidates: List[str] = list(model_candidates) if model_candidates else []

        for attempt in range(max_attempts):
            try:
                self._log(
                    f"🔄 Attempt {attempt + 1}/{max_attempts}: {'Agentic' if use_agentic else 'Planning'}...",
                    "executor", 30 + attempt * 10, attempt_count=attempt + 1,
                )
                
                if use_agentic:
                    # Agentic: use ExecutionPlan for this attempt (no LLM this round)
                    execution_plan = current_execution_plan
                    if execution_plan is None:
                        break
                    # Restrict feature_transforms to columns present in (pruned) df
                    feature_cols_set = set(c for c in df.columns if c != target)
                    execution_plan = ExecutionPlan(
                        **{
                            **execution_plan.model_dump(),
                            "feature_transforms": [
                                ft for ft in execution_plan.feature_transforms
                                if getattr(ft, "name", None) in feature_cols_set
                            ],
                        }
                    )
                    # User constraints override LLM preferences (chat-first)
                    execution_plan = self._apply_user_constraints(execution_plan, user_constraints)
                    execution_plans_used.append(execution_plan)
                    # Structured event: attempt_start — visible in chat
                    self._log(
                        f"Attempt {attempt + 1}: ExecutionPlan v{attempt + 1} — target_transform={execution_plan.target_transformation}, models={[mc.model_name for mc in execution_plan.model_candidates][:5]}",
                        "attempt_start",
                        32 + attempt * 10,
                        payload={
                            "attempt": attempt + 1,
                            "execution_plan_summary": {
                                "target_transformation": execution_plan.target_transformation,
                                "model_candidates": [mc.model_name for mc in execution_plan.model_candidates],
                                "plan_quality": execution_plan.plan_quality,
                                "primary_metric": execution_plan.primary_metric,
                            },
                        },
                        attempt_count=attempt + 1,
                    )
                    config = execution_plan_to_config(execution_plan, task)
                    config["task"] = task
                    if task == "classification":
                        config["primary_metric_locked"] = locked_metric
                    model_candidates = [mc.model_name for mc in execution_plan.model_candidates]
                    if not model_candidates:
                        model_candidates = ["ridge", "random_forest"] if task == "regression" else ["logistic_regression", "random_forest"]
                    # When user said "try something stronger" / "go for it", main.py passes extended list (xgboost, etc.); use it
                    if param_model_candidates and (
                        user_constraints and user_constraints.get("performance_mode") == "aggressive"
                        or len(param_model_candidates) > len(model_candidates)
                    ):
                        model_candidates = param_model_candidates
                        self._log(f"📊 Using extended model list (aggressive/stronger): {', '.join(model_candidates[:6])}{'...' if len(model_candidates) > 6 else ''}", "train", 58)
                    plan = None  # Not used for compile in agentic; config has feature_transforms
                else:
                    # Legacy: get plan from LLM or repair
                    if attempt == 0:
                        self._log("📋 Step 1: Getting plan from LLM...", "plan", 35)
                        plan = plan_automl(df, requested_target=requested_target, llm_provider=llm_provider)
                        self._log(f"✅ Plan received: {len(plan.feature_transforms)} feature transforms, {len(plan.model_candidates)} models", "plan", 40)
                    else:
                        self._log(f"🔧 Step 1: Repairing plan based on previous errors (attempt {attempt})...", "repair", 35)
                        plan = self._repair_plan_from_errors(
                            df=df, profile=profile, target=target, task=task,
                            previous_errors=self.attempt_history, requested_target=requested_target, llm_provider=llm_provider,
                        )
                        self._log(f"✅ Plan repaired: {len(plan.feature_transforms)} feature transforms", "repair", 40)
                    feature_cols_set = set(c for c in df.columns if c != target)
                    plan.feature_transforms = [ft for ft in plan.feature_transforms if (getattr(ft, "name", None) or (ft.get("name") if isinstance(ft, dict) else None)) in feature_cols_set]
                    if not plan.feature_transforms or all(getattr(ft, "drop", False) for ft in plan.feature_transforms):
                        self._log("⚠️  Plan has no features - auto-generating feature_transforms...", "repair", 42)
                        plan.feature_transforms = _generate_feature_transforms_from_profile(profile=profile, target=plan.inferred_target)
                    self._log("⚙️  Step 2: Building pipeline configuration from plan...", "config", 50)
                    config = self._plan_to_config(plan, profile)
                    config["task"] = task
                    if task == "classification":
                        config["primary_metric_locked"] = locked_metric
                    if len(model_candidates) < 4:
                        MORE_REGRESSION = ["ridge", "lasso", "linear_regression", "gradient_boosting", "random_forest", "svm", "xgboost"]
                        MORE_CLASSIFICATION = ["logistic_regression", "naive_bayes", "gradient_boosting", "random_forest", "svm", "xgboost"]
                        extra = MORE_REGRESSION if task == "regression" else MORE_CLASSIFICATION
                        model_candidates = list(dict.fromkeys(model_candidates + [m for m in extra if m not in model_candidates]))
                
                self._log(f"🚀 Step 3: Training models: {', '.join(model_candidates)}...", "train", 60)
                if len(model_candidates) > 1:
                    self._log(f"📊 Comparing {len(model_candidates)} models...", "train", 62)
                    def _progress_cb(msg: str, cur: int, tot: int):
                        pct = 60 + int(8 * cur / max(tot, 1))
                        self._log(msg, "train", min(pct, 68))
                    result = compare_models(
                        df=df,
                        target=target,
                        task=task,
                        metric=metric,
                        model_candidates=model_candidates,
                        base_config=config,
                        progress_callback=_progress_cb,
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
                plan_for_eval = plan if not use_agentic else current_execution_plan  # agentic: use execution_plan for reporting only
                if task == "regression":
                    self._log("🔍 Evaluating model quality (error gates)...", "evaluate", 70)
                    evaluation_passed, failure_info = self._evaluate_regression_model(
                        df=df,
                        target=target,
                        result=result,
                        plan=plan_for_eval
                    )
                    
                    if not evaluation_passed:
                        # Agentic: RepairPlan path. Legacy: heteroscedastic auto-fix + _apply_diagnosis_changes.
                        if use_agentic:
                            self._log(
                                "Attempt failed: error gates triggered — creating failure report",
                                "attempt_failure", 72,
                                payload={
                                    "attempt": attempt + 1,
                                    "failed_gates": failure_info.get("failed_gates", []),
                                    "stage": "metric_gate_failed",
                                    "metrics": result.get("metrics", {}),
                                },
                            )
                            failure_report = self._create_failure_report(
                                failure_stage="evaluation",
                                failed_gates=failure_info["failed_gates"],
                                metrics=result.get("metrics", {}),
                                target_stats=failure_info.get("target_stats", {}),
                                feature_summary=self._get_feature_summary(df, execution_plan, config),
                                model_used=result.get("model_name", "unknown"),
                                model_hyperparameters={},
                                previous_plan=execution_plan.model_dump(),
                                error_message=failure_info.get("error_message", "Error gates failed"),
                                attempt_number=attempt + 1,
                                last_execution_plan=execution_plan.model_dump(),
                                residual_diagnostics=failure_info.get("residual_diagnostics"),
                            )
                            self._log("🧠 Diagnosing failure with LLM...", "diagnose", 75)
                            diagnosis = self.diagnosis_agent.diagnose_failure(failure_report)
                            self._log(
                                f"✅ Diagnosis received (confidence: {diagnosis.recovery_confidence:.2f})",
                                "diagnose", 78,
                                payload={"recovery_confidence": diagnosis.recovery_confidence, "plan_changes": getattr(diagnosis, "plan_changes", [])},
                            )
                            if diagnosis.suggested_stop or diagnosis.recovery_confidence < 0.3:
                                self._log("🛑 LLM suggests stopping - task may not be learnable", "refuse", 80)
                                return self._create_refusal_result(
                                    plan=None,
                                    failure_report=failure_report,
                                    diagnosis=diagnosis,
                                    attempts=attempt + 1,
                                    structural_plan=structural_plan,
                                    execution_plan=execution_plan,
                                )
                            repair_changes = diagnosis.plan_changes or (getattr(diagnosis, "repair_plan") or {})
                            if failure_info.get("is_heteroscedastic") and not repair_changes.get("target_transformation"):
                                try:
                                    y_min = float(df[target].min())
                                    if y_min > 0:
                                        repair_changes = {**repair_changes, "target_transformation": "log1p"}
                                        self._log("🔧 Heteroscedastic residuals → adding log1p to repair", "repair", 73)
                                except Exception:
                                    pass
                            repair = plan_changes_to_repair_plan(repair_changes)
                            next_execution_plan = apply_repair_plan(execution_plan, repair)
                            if not self._execution_plan_meaningfully_different(execution_plan, next_execution_plan):
                                self._log(
                                    "🛑 Repair produced no meaningful change — refusing identical retry",
                                    "refuse", 80,
                                    payload={"reason": "no_meaningful_diff"},
                                )
                                return self._create_refusal_result(
                                    plan=None,
                                    failure_report=failure_report,
                                    diagnosis=diagnosis,
                                    attempts=attempt + 1,
                                    structural_plan=structural_plan,
                                    execution_plan=execution_plan,
                                )
                            current_execution_plan = next_execution_plan
                            repair_diff = {
                                "change_target_transform": repair.change_target_transform,
                                "drop_features": repair.drop_features,
                                "add_features": repair.add_features,
                                "replace_model": repair.replace_model,
                                "reorder_models": repair.reorder_models,
                                "change_encoding": repair.change_encoding,
                            }
                            repair_diff = {k: v for k, v in repair_diff.items() if v is not None}
                            self._log(
                                "Repair applied → next attempt: " + ", ".join(f"{k}={v}" for k, v in list(repair_diff.items())[:5]),
                                "repair_proposal", 80,
                                payload={
                                    "attempt": attempt + 1,
                                    "repair_diff": repair_diff,
                                    "diagnosis_summary": (diagnosis.diagnosis_md or "")[:300],
                                },
                            )
                            self.failure_history.append({
                                "attempt": attempt + 1,
                                "failure_report": failure_report.model_dump(),
                                "diagnosis": diagnosis.model_dump(),
                            })
                            continue
                        # Legacy path
                        if failure_info.get("is_heteroscedastic"):
                            suggested = detect_target_transformation_need(failure_info.get("target_stats", {}))
                            if suggested and suggested == "log":
                                try:
                                    y_min = float(df[target].min())
                                    if y_min > 0:
                                        plan_dict = plan.model_dump()
                                        plan_dict["target_transformation"] = "log1p"
                                        plan = AutoMLPlan(**plan_dict)
                                        self._log("🔧 Heteroscedastic residuals → applying log1p target transformation on next attempt", "repair", 73)
                                except Exception:
                                    pass
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
                            attempt_number=attempt + 1,
                            residual_diagnostics=failure_info.get("residual_diagnostics"),
                        )
                        self._log("🧠 Diagnosing failure with LLM...", "diagnose", 75)
                        diagnosis = self.diagnosis_agent.diagnose_failure(failure_report)
                        self._log(
                            f"✅ Diagnosis received (confidence: {diagnosis.recovery_confidence:.2f})",
                            "diagnose", 78,
                            payload={"recovery_confidence": diagnosis.recovery_confidence, "plan_changes": getattr(diagnosis, "plan_changes", [])},
                        )
                        if diagnosis.suggested_stop or diagnosis.recovery_confidence < 0.3:
                            self._log("🛑 LLM suggests stopping - task may not be learnable", "refuse", 80)
                            return self._create_refusal_result(
                                plan=plan,
                                failure_report=failure_report,
                                diagnosis=diagnosis,
                                attempts=attempt + 1
                            )
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
                else:
                    # Classification: no regression error gates; we run reinforcing quality gate below
                    pass
                
                # Reinforcing quality gate: reject "useless" models even if they passed error gates
                metrics_for_gate = result.get("metrics", {})
                cv_mean_val = result.get("cv_mean")
                quality_passed, quality_failed = check_model_quality_minimum(
                    metrics_for_gate, task, cv_mean=cv_mean_val
                )
                if not quality_passed:
                    plan_for_gate = plan if not use_agentic else execution_plan
                    self._log(
                        "❌ Model failed reinforcing quality gate (model too weak to be useful)",
                        "evaluate", 72,
                        payload={"quality_failed": quality_failed, "stage": "reinforcing_gate_failed"},
                    )
                    failure_report = self._create_failure_report(
                        failure_stage="evaluation",
                        failed_gates=quality_failed,
                        metrics=metrics_for_gate,
                        target_stats=result.get("metrics", {}).get("_target_stats", {}),
                        feature_summary=self._get_feature_summary(df, plan_for_gate, config),
                        model_used=result.get("model_name", "unknown"),
                        model_hyperparameters={},
                        previous_plan=plan_for_gate.model_dump() if plan_for_gate else None,
                        error_message="Reinforcing gate: " + "; ".join(quality_failed),
                        attempt_number=attempt + 1,
                        last_execution_plan=execution_plan.model_dump() if use_agentic else None,
                        residual_diagnostics=None,
                    )
                    self._log("🧠 Diagnosing (reinforcing gate)...", "diagnose", 75)
                    diagnosis = self.diagnosis_agent.diagnose_failure(failure_report)
                    if diagnosis.suggested_stop or diagnosis.recovery_confidence < 0.3:
                        self._log("🛑 Model quality too low — refusing to deliver", "refuse", 80)
                        return self._create_refusal_result(
                            plan=plan if not use_agentic else None,
                            failure_report=failure_report,
                            diagnosis=diagnosis,
                            attempts=attempt + 1,
                            structural_plan=structural_plan if use_agentic else None,
                            execution_plan=execution_plan if use_agentic else None,
                        )
                    if use_agentic:
                        repair_changes = diagnosis.plan_changes or (getattr(diagnosis, "repair_plan") or {})
                        repair = plan_changes_to_repair_plan(repair_changes)
                        next_execution_plan = apply_repair_plan(execution_plan, repair)
                        if not self._execution_plan_meaningfully_different(execution_plan, next_execution_plan):
                            self._log(
                                "🛑 Repair produced no meaningful change — refusing identical retry",
                                "refuse", 80,
                                payload={"reason": "no_meaningful_diff"},
                            )
                            return self._create_refusal_result(
                                plan=None,
                                failure_report=failure_report,
                                diagnosis=diagnosis,
                                attempts=attempt + 1,
                                structural_plan=structural_plan,
                                execution_plan=execution_plan,
                            )
                        current_execution_plan = next_execution_plan
                        repair_diff = {k: v for k, v in {
                            "change_target_transform": repair.change_target_transform,
                            "drop_features": repair.drop_features,
                            "replace_model": repair.replace_model,
                        }.items() if v is not None}
                        self._log(
                            "Repair (reinforcing gate) applied → next attempt: " + ", ".join(f"{k}={v}" for k, v in repair_diff.items()),
                            "repair_proposal", 80,
                            payload={"attempt": attempt + 1, "repair_diff": repair_diff},
                        )
                    else:
                        plan = self._apply_diagnosis_changes(plan, diagnosis, df, target, profile)
                    self.failure_history.append({
                        "attempt": attempt + 1,
                        "failure_report": failure_report.model_dump(),
                        "diagnosis": diagnosis.model_dump()
                    })
                    continue
                
                # Step 6: Success!
                self._log(f"🎉 Training succeeded on attempt {attempt + 1}!", "success", 80)
                if use_agentic and structural_plan is not None:
                    result["plan"] = merged_plan_dict(structural_plan, execution_plan)
                    result["structural_plan"] = structural_plan.model_dump()
                    result["execution_plans"] = [ep.model_dump() for ep in execution_plans_used]
                else:
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
                        metric = infer_classification_primary_metric(df, target)  # Lock metric (no raw accuracy default)
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
                
                # Store error for learning (agentic: execution_plan is the one we used this attempt)
                _plan_snap = (execution_plan.model_dump() if use_agentic and 'execution_plan' in locals() and execution_plan is not None
                             else (plan.model_dump() if 'plan' in locals() and plan is not None else None))
                error_record = {
                    "attempt": attempt + 1,
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "error_trace": error_trace[:500],
                    "plan": _plan_snap,
                    "failure_stage": failure_stage
                }
                self.attempt_history.append(error_record)
                
                # For evaluation failures, create failure report and diagnose
                if failure_stage == "evaluation" or "TRAINING REFUSED" in error_msg:
                    try:
                        metrics = {}
                        target_stats = {}
                        if 'result' in locals():
                            metrics = result.get("metrics", {})
                            if "_target_stats" in metrics:
                                target_stats = metrics.pop("_target_stats")
                        _plan_or_ep = execution_plan if use_agentic and 'execution_plan' in locals() else (plan if 'plan' in locals() else None)
                        _feature_summary = self._get_feature_summary(df, _plan_or_ep, config) if 'config' in locals() else {}
                        _last_ep = execution_plan.model_dump() if use_agentic and 'execution_plan' in locals() and execution_plan else None
                        failure_report = self._create_failure_report(
                            failure_stage=failure_stage,
                            failed_gates=metrics.get("_failed_gates", [error_msg]),
                            metrics=metrics,
                            target_stats=target_stats,
                            feature_summary=_feature_summary,
                            model_used=result.get("model_name", "unknown") if 'result' in locals() else "unknown",
                            model_hyperparameters={},
                            previous_plan=_last_ep if use_agentic else (plan.model_dump() if 'plan' in locals() and plan else None),
                            error_message=error_msg,
                            error_traceback=error_trace[:1000],
                            attempt_number=attempt + 1,
                            last_execution_plan=_last_ep,
                            residual_diagnostics=None,
                        )
                        self._log("🧠 Diagnosing failure with LLM...", "diagnose", 52)
                        diagnosis = self.diagnosis_agent.diagnose_failure(failure_report)
                        self._log(f"✅ Diagnosis received (confidence: {diagnosis.recovery_confidence:.2f})", "diagnose", 55)
                        if diagnosis.suggested_stop or diagnosis.recovery_confidence < 0.3:
                            self._log("🛑 LLM suggests stopping - task may not be learnable", "refuse", 60)
                            return self._create_refusal_result(
                                plan=plan if not use_agentic and 'plan' in locals() else None,
                                failure_report=failure_report,
                                diagnosis=diagnosis,
                                attempts=attempt + 1,
                                structural_plan=structural_plan if use_agentic else None,
                                execution_plan=execution_plan if use_agentic and 'execution_plan' in locals() else None,
                            )
                        if use_agentic and 'execution_plan' in locals() and execution_plan is not None:
                            self._log("🔧 Applying RepairPlan (from exception path)...", "repair", 55)
                            repair_changes = diagnosis.plan_changes or (getattr(diagnosis, "repair_plan") or {})
                            repair = plan_changes_to_repair_plan(repair_changes)
                            next_execution_plan = apply_repair_plan(execution_plan, repair)
                            if not self._execution_plan_meaningfully_different(execution_plan, next_execution_plan):
                                self._log(
                                    "🛑 Repair produced no meaningful change — refusing identical retry",
                                    "refuse", 55,
                                    payload={"reason": "no_meaningful_diff"},
                                )
                                return self._create_refusal_result(
                                    plan=None,
                                    failure_report=failure_report,
                                    diagnosis=diagnosis,
                                    attempts=attempt + 1,
                                    structural_plan=structural_plan,
                                    execution_plan=execution_plan,
                                )
                            current_execution_plan = next_execution_plan
                            self.failure_history.append({
                                "attempt": attempt + 1,
                                "failure_report": failure_report.model_dump(),
                                "diagnosis": diagnosis.model_dump()
                            })
                            continue
                        if 'plan' in locals() and plan is not None:
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
        plan: Optional[Union[AutoMLPlan, ExecutionPlan]] = None,
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
        
        residual_diagnostics = {
            "is_heteroscedastic": is_heteroscedastic,
            "description": residual_desc,
            "is_variance_fit_illusion": is_illusion,
            "illusion_description": illusion_desc if is_illusion else None,
        }
        
        failure_info = {
            "failed_gates": failed_gates,
            "target_stats": target_stats,
            "metrics": metrics,
            "is_variance_fit_illusion": is_illusion,
            "is_heteroscedastic": is_heteroscedastic,
            "residual_diagnostics": residual_diagnostics,
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
        attempt_number: int = 1,
        last_execution_plan: Optional[Dict[str, Any]] = None,
        residual_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> FailureReport:
        """Create structured failure report (agentic: last_execution_plan, residual_diagnostics)."""
        return FailureReport(
            failure_stage=failure_stage,
            failed_gates=failed_gates,
            metrics=metrics,
            target_stats=target_stats,
            residual_diagnostics=residual_diagnostics,
            feature_summary=feature_summary,
            model_used=model_used,
            model_hyperparameters=model_hyperparameters,
            previous_plan=previous_plan,
            last_execution_plan=last_execution_plan,
            error_message=error_message,
            error_traceback=error_traceback,
            attempt_number=attempt_number
        )
    
    def _apply_user_constraints(
        self, execution_plan: ExecutionPlan, user_constraints: Optional[Dict[str, Any]]
    ) -> ExecutionPlan:
        """Apply user constraints (chat messages) — override LLM preferences."""
        if not user_constraints:
            return execution_plan
        ep_dict = execution_plan.model_dump()
        # drop_columns: set drop=True for those features
        if user_constraints.get("drop_columns"):
            drop_set = set(c.strip() for c in user_constraints["drop_columns"] if c)
            new_fts = []
            for ft in ep_dict["feature_transforms"]:
                ft_dict = ft if isinstance(ft, dict) else (ft.model_dump() if hasattr(ft, "model_dump") else ft)
                ft_dict = dict(ft_dict)
                if ft_dict.get("name") in drop_set:
                    ft_dict["drop"] = True
                new_fts.append(ft_dict)
            ep_dict["feature_transforms"] = new_fts
        if user_constraints.get("exclude_models"):
            exclude = set(m.strip().lower() for m in user_constraints["exclude_models"])
            ep_dict["model_candidates"] = [
                mc for mc in ep_dict["model_candidates"]
                if (mc.get("model_name") or "").lower() not in exclude
            ]
            if not ep_dict["model_candidates"]:
                ep_dict["model_candidates"] = execution_plan.model_dump()["model_candidates"]
        if user_constraints.get("keep_features"):
            keep = set(user_constraints["keep_features"])
            new_fts = []
            for ft in ep_dict["feature_transforms"]:
                ft_dict = ft if isinstance(ft, dict) else (ft.model_dump() if hasattr(ft, "model_dump") else ft)
                if ft_dict.get("name") in keep:
                    ft_dict = dict(ft_dict)
                    ft_dict["drop"] = False
                new_fts.append(ft_dict)
            ep_dict["feature_transforms"] = new_fts
        if user_constraints.get("primary_metric"):
            ep_dict["primary_metric"] = user_constraints["primary_metric"]
        return ExecutionPlan(**ep_dict)

    def _execution_plan_meaningfully_different(
        self, prev: ExecutionPlan, next_ep: ExecutionPlan
    ) -> bool:
        """
        True if next_ep differs from prev in a way that could change training outcome.
        If repair produces no meaningful change, we must REFUSE (no silent identical retry).
        """
        if prev.target_transformation != next_ep.target_transformation:
            return True
        prev_names = {ft.name for ft in prev.feature_transforms}
        next_names = {ft.name for ft in next_ep.feature_transforms}
        if prev_names != next_names:
            return True
        prev_encodes = {ft.name: getattr(ft, "encode", "none") for ft in prev.feature_transforms}
        next_encodes = {ft.name: getattr(ft, "encode", "none") for ft in next_ep.feature_transforms}
        if prev_encodes != next_encodes:
            return True
        prev_models = [mc.model_name for mc in prev.model_candidates]
        next_models = [mc.model_name for mc in next_ep.model_candidates]
        if prev_models != next_models:
            return True
        return False

    def _get_feature_summary(
        self,
        df: pd.DataFrame,
        plan_or_execution: Optional[Union[AutoMLPlan, ExecutionPlan]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get feature preprocessing summary (accepts AutoMLPlan or ExecutionPlan)."""
        if plan_or_execution is None:
            return {
                "feature_count_before": len(df.columns) - 1,
                "feature_count_after": 0,
                "dropped_features": [],
                "kept_features": [],
                "encoded_features": [],
                "scaled_features": [],
            }
        feature_transforms = plan_or_execution.feature_transforms
        kept_features = [ft.name for ft in feature_transforms if not getattr(ft, "drop", False)]
        dropped_features = [ft.name for ft in feature_transforms if getattr(ft, "drop", False)]
        return {
            "feature_count_before": len(df.columns) - 1,
            "feature_count_after": len(kept_features),
            "dropped_features": dropped_features,
            "kept_features": kept_features,
            "encoded_features": [ft.name for ft in feature_transforms if getattr(ft, "encode", "none") != "none"],
            "scaled_features": [ft.name for ft in feature_transforms if getattr(ft, "scale", "none") != "none"],
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
        attempts: int,
        structural_plan: Optional[StructuralPlan] = None,
        execution_plan: Optional[ExecutionPlan] = None,
    ) -> Dict[str, Any]:
        """
        Create a result that represents a REFUSAL (epistemically honest failure).
        Agentic: pass structural_plan + execution_plan to build plan dict; otherwise use plan or minimal.
        """
        if structural_plan is not None and execution_plan is not None:
            plan_dict = merged_plan_dict(structural_plan, execution_plan)
            plan_dict["planning_source"] = "refusal"
            plan_dict["planning_error"] = "Task refused - not reliably solvable"
            plan_dict["plan_quality"] = "fallback_low_confidence"
        elif plan is not None:
            plan_dict = plan.model_dump()
        else:
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
        
        return {
            "refused": True,
            "refusal_reason": diagnosis.diagnosis_md,
            "failed_gates": failure_report.failed_gates,
            "metrics": failure_report.metrics,
            "target_stats": failure_report.target_stats,
            "plan": plan_dict,
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
        cfg = {
            "task": plan.task_type,
            "model": plan.model_candidates[0].model_name if plan.model_candidates else "random_forest",
            "feature_transforms": [ft.model_dump() if hasattr(ft, 'model_dump') else ft for ft in plan.feature_transforms],
            "automl_plan": plan.model_dump()
        }
        if getattr(plan, "target_transformation", None):
            cfg["target_transformation"] = plan.target_transformation
        return cfg
    
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
                def _fallback_progress_cb(msg: str, cur: int, tot: int):
                    self._log(msg, "train", 50 + int(10 * cur / max(tot, 1)))
                result = compare_models(
                    df, target, task, metric, model_candidates, config,
                    progress_callback=_fallback_progress_cb,
                )
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
