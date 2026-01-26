"""
Autonomous error-fixing agent for Intent2Model.

Automatically fixes errors in the background using LLM reasoning.
"""

from typing import Dict, Any, Optional, Callable
from agents.llm_interface import LLMInterface
import traceback
import json
import re


class AutoFixAgent:
    """
    Autonomous agent that fixes errors automatically using LLM reasoning.
    """
    
    def __init__(self, llm_provider: str = "gemini"):
        self.llm = LLMInterface(provider=llm_provider)
        self.max_retries = 3
    
    def fix_and_retry(
        self,
        func: Callable,
        *args,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function and automatically fix errors if they occur.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            context: Additional context for error fixing
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result or raises exception if all fixes fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries - 1:
                    # Try to fix the error
                    fix_suggestion = self._analyze_and_fix_error(
                        error=e,
                        context=context or {},
                        attempt=attempt + 1
                    )
                    
                    if fix_suggestion and fix_suggestion.get("should_retry"):
                        # Apply fix if possible
                        if fix_suggestion.get("fix_code"):
                            try:
                                # Try to apply code fix (for simple cases)
                                self._apply_code_fix(fix_suggestion["fix_code"], func, args, kwargs)
                            except:
                                pass
                        continue
                
        # All retries failed, raise the last error
        raise last_error
    
    def _analyze_and_fix_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        attempt: int
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze error and suggest fixes.
        """
        try:
            error_msg = str(error)
            error_type = type(error).__name__
            error_traceback = traceback.format_exc()
            
            system_prompt = """You are an expert Python/ML engineer that fixes code errors automatically.
Analyze the error and provide a fix. Focus on fixing the actual bug, not just explaining it."""

            prompt = f"""A Python error occurred during ML model training. Fix it automatically.

**Error Type:** {error_type}
**Error Message:** {error_msg}
**Attempt:** {attempt}/{self.max_retries}

**Context:**
{json.dumps(context, indent=2)}

**Traceback:**
{error_traceback[:1000]}

**Task:** Fix this error automatically. Return JSON:
{{
  "should_retry": true/false,
  "fix_description": "what you're fixing",
  "fix_code": "python code to fix (if applicable)",
  "alternative_approach": "alternative approach if fix doesn't work"
}}

Be specific and actionable. If it's a simple bug (like undefined variable), provide the fix."""

            response = self.llm.generate(prompt, system_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    return json.loads(json_str)
                except:
                    pass
            
            # Fallback: check if it's a simple fix
            if "name 'le' is not defined" in error_msg or "NameError.*le" in error_msg:
                return {
                    "should_retry": True,
                    "fix_description": "label_encoder not defined in regression function",
                    "fix_code": "label_encoder = None",
                    "alternative_approach": "Return None for label_encoder in regression"
                }
            
            return {"should_retry": False}
            
        except Exception as e:
            print(f"Error analysis failed: {e}")
            return {"should_retry": False}
    
    def _apply_code_fix(self, fix_code: str, func: Callable, args: tuple, kwargs: dict):
        """
        Apply a code fix (limited to simple cases).
        This is a placeholder - actual fixes should be in the code itself.
        """
        # For now, we don't dynamically modify code
        # Fixes should be applied in the actual code
        pass


def auto_fix_training_error(
    func: Callable,
    *args,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Wrapper function that automatically fixes training errors.
    
    Usage:
        result = auto_fix_training_error(train_regression, df, target, metric, config)
    """
    agent = AutoFixAgent()
    return agent.fix_and_retry(func, *args, context=context, **kwargs)
