"""
API Key Manager for LLM providers.
Automatically uses .env file keys by default.
Custom keys (from UI) override env keys only if explicitly set.
"""

from typing import Optional
import os
from dotenv import load_dotenv

# Load .env file once on module import
load_dotenv()

# In-memory storage for custom API keys (from UI)
_custom_api_keys = {}

def set_custom_api_key(key: str, provider: str = "gemini"):
    """Set a custom API key for a provider (overrides .env key)."""
    if key and key.strip():
        _custom_api_keys[provider] = key.strip()
    else:
        # Clear custom key to use .env default
        _custom_api_keys.pop(provider, None)

def get_api_key(provider: str = "gemini") -> Optional[str]:
    """
    Get API key - .env file is DEFAULT, custom keys override only if set.
    
    Priority:
    1. Custom key (if explicitly set via UI)
    2. .env file key (default, automatic)
    3. Environment variable (fallback)
    """
    # First check if custom key is explicitly set (from UI)
    if provider in _custom_api_keys and _custom_api_keys[provider]:
        return _custom_api_keys[provider]
    
    # Default: Use .env file key (automatically loaded)
    # This ensures .env is always used unless user explicitly sets a custom key
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if key and key.strip():
            return key.strip()
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if key and key.strip():
            return key.strip()
    elif provider == "groq":
        key = os.getenv("GROQ_API_KEY", "")
        if key and key.strip():
            return key.strip()
    
    return None
