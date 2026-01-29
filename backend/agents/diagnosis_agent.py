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
        return """You are a senior ML engineer diagnosing a failed model training.

Your job is to:
1. Understand WHY the model failed
2. Identify which assumptions were wrong
3. Suggest STRUCTURED changes to the AutoMLPlan
4. Assess whether the task is learnable with the available data

Be epistemically honest. If the task is not learnable, say so.
If the data is insufficient, say so.
If the model architecture is wrong, suggest alternatives.

Your response MUST be valid JSON matching the DiagnosisResponse schema."""
    
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

Feature Summary:
{chr(10).join(f"  - {k}: {v}" for k, v in report_dict['feature_summary'].items())}

Model Used: {report_dict['model_used']}
Hyperparameters: {json.dumps(report_dict['model_hyperparameters'], indent=2)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide a JSON response with:
1. diagnosis_md: Markdown explanation of why it failed and what assumptions were wrong
2. plan_changes: Structured changes to apply:
   - target_transformation: Optional transformation ("log", "quantile", "robust", null)
   - feature_transforms: List of changes (add/drop/modify features)
   - model_selection: Alternative model to try
   - evaluation_metrics: Different metrics to use
3. recovery_confidence: 0-1 confidence that changes will fix it
4. is_task_learnable: Whether this task is learnable with this data
5. suggested_stop: Whether to stop trying (if task is not learnable)

Example plan_changes:
{{
  "target_transformation": "log",
  "feature_transforms": [
    {{"action": "drop", "feature": "leakage_candidate_col"}},
    {{"action": "add_encoding", "feature": "high_cardinality_col", "encoding": "frequency"}}
  ],
  "model_selection": "gradient_boosting",
  "evaluation_metrics": ["rmse", "mae"]
}}

Return ONLY valid JSON, no markdown code blocks."""
        
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
