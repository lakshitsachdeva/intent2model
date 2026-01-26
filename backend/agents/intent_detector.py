"""
Intent detection agent - uses LLM to understand user intent.
"""

from typing import Dict, Any, Optional, Literal
from agents.llm_interface import LLMInterface
import json
import re


class IntentDetectionAgent:
    """
    Uses LLM to detect user intent from natural language.
    """
    
    def __init__(self, llm_provider: str = "gemini"):
        self.llm = LLMInterface(provider=llm_provider)
    
    def detect_intent(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect user intent from natural language input.
        
        Returns:
            {
                "intent": "train" | "predict" | "report" | "query" | "unknown",
                "target_column": str or None,
                "confidence": float,
                "reasoning": str
            }
        """
        try:
            system_prompt = """You are an intent detection agent. Analyze user input and determine their intent.
Intents:
- "train": User wants to train a model (mentions column name, "train", "build model", etc.)
- "predict": User wants to make predictions ("yes", "predict", "make prediction", "want to predict", etc.)
- "report": User wants to see results ("report", "show results", "summary", etc.)
- "query": User is asking a question or wants information
- "unknown": Can't determine intent

Be smart about context. If user just says "yes" after being asked about predictions, they mean "predict".
If user says "want to make prediction", they mean "predict", not "train"."""

            prompt = f"""Detect the user's intent from this input:

**User Input:** "{user_input}"

**Context:**
- Has trained model: {context.get('has_trained_model', False)}
- Available columns: {context.get('available_columns', [])}
- Previous message: {context.get('previous_message', 'none')}

**Task:** Determine the user's intent. Return JSON:
{{
  "intent": "train" | "predict" | "report" | "query" | "unknown",
  "target_column": "column name if intent is train, null otherwise",
  "confidence": 0.0-1.0,
  "reasoning": "why you chose this intent"
}}

Examples:
- "yes" after asking about predictions → intent: "predict"
- "want to make prediction" → intent: "predict"
- "variety" → intent: "train", target_column: "variety"
- "sepal.length" → intent: "train", target_column: "sepal.length"
- "report" → intent: "report"
- "show me results" → intent: "report"
"""

            response = self.llm.generate(prompt, system_prompt)
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return {
                        "intent": result.get("intent", "unknown"),
                        "target_column": result.get("target_column"),
                        "confidence": result.get("confidence", 0.5),
                        "reasoning": result.get("reasoning", "")
                    }
                except:
                    pass
            
            # Fallback: simple rule-based detection
            return self._fallback_intent_detection(user_input, context)
            
        except Exception as e:
            print(f"Intent detection failed: {e}")
            return self._fallback_intent_detection(user_input, context)
    
    def _fallback_intent_detection(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback rule-based intent detection."""
        lower_input = user_input.lower().strip()
        
        # Prediction keywords (high priority)
        predict_keywords = [
            "predict", "prediction", "yes", "sure", "ok", "okay", 
            "want to make", "make prediction", "let's predict", "lets predict"
        ]
        if any(kw in lower_input for kw in predict_keywords):
            return {
                "intent": "predict",
                "target_column": None,
                "confidence": 0.8,
                "reasoning": "Contains prediction keywords"
            }
        
        # Report keywords
        report_keywords = ["report", "summary", "results", "show", "view"]
        if any(kw in lower_input for kw in report_keywords):
            return {
                "intent": "report",
                "target_column": None,
                "confidence": 0.8,
                "reasoning": "Contains report keywords"
            }
        
        # Check if it's a column name
        available_columns = context.get("available_columns", [])
        for col in available_columns:
            if col.lower() == lower_input:
                return {
                    "intent": "train",
                    "target_column": col,
                    "confidence": 0.9,
                    "reasoning": f"Matches column name: {col}"
                }
        
        # Default: treat as train (legacy behavior)
        return {
            "intent": "train",
            "target_column": user_input.strip(),
            "confidence": 0.5,
            "reasoning": "Default to train intent"
        }
