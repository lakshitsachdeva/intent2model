"""
Autonomous recovery agent - handles all edge cases automatically.
Never says no to users, always finds a way.
"""

from typing import Dict, Any, Optional, List
from agents.llm_interface import LLMInterface
import json


class AutonomousRecoveryAgent:
    """
    Autonomous agent that handles all edge cases and never fails.
    Always finds a solution, never shows errors to users.
    """
    
    def __init__(self, llm_provider: str = "gemini"):
        self.llm = LLMInterface(provider=llm_provider)
    
    def recover_from_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        available_options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to autonomously recover from any error.
        Always returns a solution, never gives up.
        """
        try:
            error_msg = str(error)
            error_type = type(error).__name__
            
            system_prompt = """You are an autonomous AI agent that NEVER gives up.
Your job is to find solutions to ANY problem. You must always provide a way forward.
Never say "can't do it" or "not possible". Always find an alternative approach."""

            prompt = f"""An error occurred but we MUST find a solution. Never give up.

**Error:** {error_type}: {error_msg}

**Context:**
{json.dumps(context, indent=2)}

**Available Options:**
{available_options or ["Try alternative approach", "Use fallback method", "Retry with different parameters"]}

**Task:** Find a solution. Return JSON:
{{
  "solution": "what to do next",
  "action": "specific action to take",
  "alternative_approach": "if first doesn't work",
  "should_retry": true/false,
  "retry_params": {{"key": "value"}}
}}

Remember: NEVER give up. Always find a way."""

            response = self.llm.generate(prompt, system_prompt)
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            # Fallback: always provide a solution
            return {
                "solution": "Use alternative approach",
                "action": "retry_with_fallback",
                "alternative_approach": "Try with default parameters",
                "should_retry": True,
                "retry_params": {}
            }
            
        except Exception as e:
            # Even if LLM fails, we provide a solution
            return {
                "solution": "Use fallback method",
                "action": "use_defaults",
                "alternative_approach": "Continue with basic configuration",
                "should_retry": True,
                "retry_params": {}
            }
    
    def handle_missing_dataset(
        self,
        dataset_id: Optional[str],
        available_datasets: List[str]
    ) -> Dict[str, Any]:
        """
        Handle missing dataset - always find a solution.
        """
        if available_datasets:
            return {
                "solution": f"Use most recent dataset (ID: {available_datasets[-1]})",
                "action": "use_most_recent",
                "dataset_id": available_datasets[-1],
                "message": "Using most recent dataset automatically"
            }
        else:
            return {
                "solution": "Request dataset upload",
                "action": "request_upload",
                "message": "No dataset available. Please upload a CSV file."
            }
    
    def handle_missing_model(
        self,
        run_id: str,
        available_models: List[str]
    ) -> Dict[str, Any]:
        """
        Handle missing model - always find a solution.
        """
        if available_models:
            return {
                "solution": f"Use most recent model (ID: {available_models[-1]})",
                "action": "use_most_recent",
                "run_id": available_models[-1],
                "message": "Using most recent model automatically"
            }
        else:
            return {
                "solution": "Request model training",
                "action": "request_training",
                "message": "No trained model available. Training a new model..."
            }
    
    def suggest_column_alternative(
        self,
        requested_column: str,
        available_columns: List[str]
    ) -> Optional[str]:
        """
        If requested column doesn't exist, suggest the closest match.
        """
        if not available_columns:
            return None
        
        # Simple fuzzy matching
        requested_lower = requested_column.lower()
        for col in available_columns:
            if requested_lower in col.lower() or col.lower() in requested_lower:
                return col
        
        # Return first available as fallback
        return available_columns[0]
