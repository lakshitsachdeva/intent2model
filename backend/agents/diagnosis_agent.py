"""
Diagnosis Agent - LLM-powered failure diagnosis and recovery planning.

When training fails, this agent analyzes the failure report and suggests
principled fixes using LLM reasoning.
"""

from typing import Dict, Any, Optional
from schemas.failure_schema import FailureReport, DiagnosisResponse
from agents.llm_interface import LLMInterface
import json
import re


class DiagnosisAgent:
    """
    LLM-powered agent that diagnoses training failures and suggests recovery plans.
    """
    
    def __init__(self, llm_provider: str = "gemini"):
        self.llm = LLMInterface(provider=llm_provider)
        self.llm_provider = llm_provider
    
    def diagnose_failure(self, failure_report: FailureReport) -> DiagnosisResponse:
        """
        Diagnose a training failure and suggest recovery plan.
        
        Args:
            failure_report: Structured failure report
            
        Returns:
            DiagnosisResponse with diagnosis, plan changes, and confidence
        """
        prompt = self._build_diagnosis_prompt(failure_report)
        system_prompt = self._build_system_prompt()
        
        try:
            response = self.llm.generate(prompt, system_prompt)
            diagnosis_dict = self._extract_diagnosis_json(response)
            
            # Validate and return
            return DiagnosisResponse(**diagnosis_dict)
        except Exception as e:
            # Fallback diagnosis if LLM fails
            return self._fallback_diagnosis(failure_report, str(e))
    
    def _build_system_prompt(self) -> str:
        return """You are an ML engineer advisor diagnosing a failed training run.
You DO NOT have access to the codebase, filesystem, or tools. Do NOT claim to inspect or modify files or code.

Your job is to:
1. Explain WHY the model failed (from the failure report only)
2. Identify which assumptions were wrong
3. Propose ONE or TWO structured changes as a diff only (e.g. change_target_transformation, drop_features, replace_model)
4. Say whether this task is realistically learnable with this data

You MUST return:
- diagnosis_md: markdown explanation (reasoning only; do not claim you applied or inspected code)
- repair_plan / plan_changes: STRUCTURED DIFF ONLY. No full plans. No free text for execution.
- recovery_confidence: 0-1
- suggested_stop: true if task is not learnable

You are NOT allowed to:
- Claim you inspected, modified, or applied code or files
- Write any code (no sklearn, pandas, pipeline code)
- Disable or relax error gates or metrics
- Force or pretend success
- Output anything other than structured diffs and the required JSON fields

Be epistemically honest. Refusal is a valid outcome. Your response MUST be valid JSON."""
    
    def _format_residual_diagnostics(self, rd: Optional[Dict[str, Any]]) -> str:
        if not rd:
            return "  (none)"
        lines = []
        for k, v in rd.items():
            if isinstance(v, bool):
                lines.append(f"  - {k}: {v}")
            elif isinstance(v, str):
                lines.append(f"  - {k}: {v}")
            elif v is not None:
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines) if lines else "  (none)"

    def _build_diagnosis_prompt(self, failure_report: FailureReport) -> str:
        report_dict = failure_report.model_dump()
        
        prompt = f"""Analyze this training failure and provide a diagnosis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILURE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Failure Stage: {report_dict['failure_stage']}
Attempt Number: {report_dict['attempt_number']}
Error Message: {report_dict['error_message']}

Failed Gates:
{chr(10).join(f"  - {gate}" for gate in report_dict['failed_gates'])}

Metrics:
{chr(10).join(f"  - {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  - {k}: {v}" for k, v in report_dict['metrics'].items())}

Target Statistics:
{chr(10).join(f"  - {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  - {k}: {v}" for k, v in report_dict['target_stats'].items())}

Residual Diagnostics (if present):
{self._format_residual_diagnostics(report_dict.get('residual_diagnostics'))}

Feature Summary:
{chr(10).join(f"  - {k}: {v}" for k, v in report_dict['feature_summary'].items())}

Model Used: {report_dict['model_used']}
Hyperparameters: {json.dumps(report_dict['model_hyperparameters'], indent=2)}

Last ExecutionPlan (what was used this attempt):
{json.dumps(report_dict.get('last_execution_plan') or {}, indent=2)[:2000]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide a JSON response with:
1. diagnosis_md: Markdown explanation of why it failed and what assumptions were wrong
2. plan_changes (or repair_plan): STRUCTURED DIFF ONLY. Allowed keys:
   - target_transformation: "log" | "log1p" | "quantile" | "robust" | null
   - drop_features: list of feature names to drop
   - add_features: list of feature names to add back (if previously dropped)
   - replace_model: single model name to use instead
   - add_models: list of model names to add to candidates
   - remove_models: list of model names to remove
   - reorder_models: list of model names in desired order
   - change_encoding: {{ "feature_name": "one_hot"|"ordinal"|"frequency" }}
3. recovery_confidence: 0-1
4. is_task_learnable: bool
5. suggested_stop: bool (true if task is not learnable)

You must NOT: write code, disable gates, relax metrics, or force success. Return ONLY valid JSON, no markdown code blocks."""
        
        return prompt
    
    def _extract_diagnosis_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        
        # If no JSON found, try to parse the whole response
        try:
            return json.loads(response)
        except:
            # Fallback: return minimal diagnosis
            return {
                "diagnosis_md": f"LLM response parsing failed. Raw response: {response[:500]}",
                "plan_changes": {},
                "recovery_confidence": 0.3,
                "is_task_learnable": True,
                "suggested_stop": False,
            }
    
    def _fallback_diagnosis(self, failure_report: FailureReport, error: str) -> DiagnosisResponse:
        """Fallback diagnosis when LLM fails."""
        failed_gates = failure_report.failed_gates
        metrics = failure_report.metrics
        
        diagnosis_md = f"""Automatic diagnosis failed (LLM error: {error}).

Observed failures:
{chr(10).join(f"- {gate}" for gate in failed_gates)}

Metrics:
- RMSE: {metrics.get('RMSE', 'N/A')}
- normalized_RMSE: {metrics.get('normalized_RMSE', 'N/A')}
- R²: {metrics.get('R²', 'N/A')}

Suggested fixes:
1. Try target transformation (log/quantile) if skew is high
2. Try simpler baseline models (linear_regression)
3. Check for data leakage or insufficient features
"""
        
        plan_changes = {}
        
        # Auto-suggest target transformation if normalized errors are high
        if metrics.get("normalized_RMSE", 0) > 0.5:
            target_stats = failure_report.target_stats
            if target_stats.get("skew", 0) > 2.0:
                plan_changes["target_transformation"] = "log"
            elif abs(target_stats.get("skew", 0)) > 3.0:
                plan_changes["target_transformation"] = "quantile"
        
        return DiagnosisResponse(
            diagnosis_md=diagnosis_md,
            plan_changes=plan_changes,
            recovery_confidence=0.4,
            is_task_learnable=True,
            suggested_stop=False,
        )
