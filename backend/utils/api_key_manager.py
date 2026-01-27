"""
API Key Manager for LLM providers.
Allows storing and retrieving custom API keys.
"""

from typing import Optional
import os

# In-memory storage for custom API keys
_custom_api_keys = {}

def set_custom_api_key(key: str, provider: str = "gemini"):
    """Set a custom API key for a provider."""
    _custom_api_keys[provider] = key

def get_api_key(provider: str = "gemini") -> Optional[str]:
    """Get API key - custom first, then environment variable."""
    # Check custom keys first
    if provider in _custom_api_keys:
        return _custom_api_keys[provider]
    
    # Fall back to environment variables
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "")
    elif provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    elif provider == "groq":
        return os.getenv("GROQ_API_KEY", "")
    
    return None
