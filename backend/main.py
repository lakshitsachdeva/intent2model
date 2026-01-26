@app.post("/detect-intent")
async def detect_intent(request: Dict[str, Any]):
    """
    Detect user intent from natural language using LLM.
    AUTONOMOUS: LLM makes all decisions.
    """
    try:
        agent = IntentDetectionAgent()
        
        intent_result = agent.detect_intent(
            user_input=request.get("user_input", ""),
            context=request.get("context", {})
        )
        
        return intent_result
    except Exception as e:
        # Fallback to simple detection
        return {
            "intent": "unknown",
            "target_column": None,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }
