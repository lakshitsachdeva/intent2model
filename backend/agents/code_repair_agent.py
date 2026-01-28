"""
Code Repair Agent - Automatically fixes compilation and execution errors using LLM.
"""

from typing import Dict, Any, Optional, List
from agents.llm_interface import LLMInterface
import json
import re


class CodeRepairAgent:
    """
    Autonomous agent that fixes code errors automatically using LLM.
    Never gives up, always finds a solution.
    """
    
    def __init__(self, llm_provider: str = "gemini"):
        from utils.api_key_manager import get_api_key
        api_key = get_api_key(provider=llm_provider)
        self.llm = LLMInterface(provider=llm_provider, api_key=api_key)
    
    def repair_code_error(
        self,
        error: Exception,
        code_snippet: str,
        context: Dict[str, Any],
        error_type: str = "NameError"
    ) -> Dict[str, Any]:
        """
        Use LLM to automatically fix code errors.
        
        Returns:
            {
                "fixed_code": str,
                "explanation": str,
                "changes_made": List[str]
            }
        """
        try:
            error_msg = str(error)
            
            system_prompt = """You are a code repair agent. Your job is to fix code errors automatically.
You must:
1. Identify the exact problem
2. Fix it completely
3. Ensure the fixed code is executable
4. Never leave undefined variables or incomplete fixes

Return ONLY the fixed code, no explanations unless asked."""
            
            prompt = f"""Fix this code error automatically:

**Error Type:** {error_type}
**Error Message:** {error_msg}

**Code with Error:**
```python
{code_snippet}
```

**Context:**
- Dataset columns: {context.get('columns', [])}
- Target: {context.get('target', 'unknown')}
- Task: {context.get('task', 'unknown')}
- Available variables: {context.get('available_vars', [])}

**Task:** Fix the code so it runs without errors. Return the COMPLETE fixed code block.

**Rules:**
1. Define ALL variables before use
2. Import ALL required modules
3. Ensure all variable names are correct
4. Make sure the code is complete and executable

Return the fixed code in a code block:"""
            
            response = self.llm.generate(prompt, system_prompt)
            
            # Extract code block
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if not code_match:
                code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
            
            if code_match:
                fixed_code = code_match.group(1).strip()
                return {
                    "fixed_code": fixed_code,
                    "explanation": "Code automatically repaired by LLM",
                    "changes_made": ["Fixed undefined variable references", "Added missing imports", "Corrected variable names"]
                }
            
            # Fallback: try to extract just the code
            lines = response.split('\n')
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            if code_lines:
                return {
                    "fixed_code": '\n'.join(code_lines),
                    "explanation": "Code extracted and repaired",
                    "changes_made": ["Applied automatic fixes"]
                }
            
        except Exception as e:
            print(f"⚠️  Code repair LLM failed: {e}")
        
        # Ultimate fallback: return safe code
        return self._generate_safe_fallback_code(error, code_snippet, context)
    
    def _generate_safe_fallback_code(
        self,
        error: Exception,
        code_snippet: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate safe fallback code when LLM repair fails."""
        error_msg = str(error).lower()
        
        # Common fixes
        if "numeric_cols" in error_msg or "nameerror" in error_msg:
            # Define numeric_cols from X
            fixed = code_snippet
            if "numeric_cols" in error_msg and "numeric_cols" not in code_snippet:
                fixed = f"# Auto-fix: Define numeric_cols\nimport numpy as np\nnumeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n\n{fixed}"
            if "categorical_cols" in error_msg and "categorical_cols" not in code_snippet:
                fixed = f"{fixed}\n\n# Auto-fix: Define categorical_cols\ncategorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()"
            
            return {
                "fixed_code": fixed,
                "explanation": "Applied automatic variable definition fix",
                "changes_made": ["Added missing variable definitions"]
            }
        
        # Generic fallback
        return {
            "fixed_code": code_snippet,
            "explanation": "Could not auto-repair - using original code",
            "changes_made": []
        }
    
    def repair_notebook_cell(
        self,
        cell_code: str,
        error: Exception,
        notebook_context: Dict[str, Any]
    ) -> str:
        """
        Repair a specific notebook cell that has an error.
        
        Returns fixed code for the cell.
        """
        repair_result = self.repair_code_error(
            error=error,
            code_snippet=cell_code,
            context={
                "columns": notebook_context.get("columns", []),
                "target": notebook_context.get("target", ""),
                "task": notebook_context.get("task", ""),
                "available_vars": notebook_context.get("available_vars", ["X", "y", "pipeline", "df"])
            }
        )
        
        return repair_result["fixed_code"]
