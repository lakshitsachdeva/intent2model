"""
LLM interface for Intent2Model.

Abstract interface for LLM providers (Gemini, OpenAI, Groq).
"""

from typing import Optional
import os

# Global tracking for current model usage
_current_model_name = None
_current_model_reason = None

def get_current_model_info():
    """Get info about which model is currently being used."""
    return {
        "model": _current_model_name,
        "reason": _current_model_reason
    }

class LLMInterface:
    """
    Abstract interface for LLM providers.
    Supports Gemini, OpenAI, and Groq.
    """
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "gemini":
            return os.getenv("GEMINI_API_KEY", "")
        elif self.provider == "groq":
            return os.getenv("GROQ_API_KEY", "")
        return ""
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            return self._openai_generate(prompt, system_prompt)
        elif self.provider == "gemini":
            return self._gemini_generate(prompt, system_prompt)
        elif self.provider == "groq":
            return self._groq_generate(prompt, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _openai_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate using OpenAI API."""
        try:
            import openai
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _gemini_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate using Gemini API with automatic model fallback on rate limits."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Model priority list - ordered by preference, with alternatives for rate limits
            # Based on rate limits: gemini-2.5-flash (21/20 RPD limit), try alternatives
            model_priority = [
                {'name': 'gemini-2.5-flash', 'reason': 'Latest fast model'},
                {'name': 'gemini-2.5-flash-lite', 'reason': 'Lite version (higher limits)'},
                {'name': 'gemini-3-flash', 'reason': 'Newer flash model'},
                {'name': 'gemini-flash-latest', 'reason': 'Latest flash fallback'},
                {'name': 'gemini-2.5-pro', 'reason': 'Pro model (different quota)'},
                {'name': 'gemini-1.5-flash', 'reason': 'Older stable model'},
                {'name': 'gemini-1.5-pro', 'reason': 'Older pro model'},
            ]
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            last_error = None
            rate_limit_models = []  # Track which models hit rate limits
            
            for model_info in model_priority:
                model_name = model_info['name']
                reason = model_info['reason']
                
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(full_prompt)
                    
                    # Success! Log which model worked and why
                    global _current_model_name, _current_model_reason
                    _current_model_name = model_name
                    
                    if model_name != model_priority[0]['name']:
                        fallback_reason = f"Fallback: {reason} (primary model hit rate limit)"
                        _current_model_reason = fallback_reason
                        print(f"✅ Using {model_name} ({fallback_reason})")
                    else:
                        _current_model_reason = reason
                        print(f"✅ Using {model_name} ({reason})")
                    
                    return response.text
                    
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check if it's a rate limit error
                    is_rate_limit = (
                        '429' in str(e) or 
                        'quota' in error_str or 
                        'rate limit' in error_str or
                        'resourceexhausted' in error_str
                    )
                    
                    if is_rate_limit:
                        rate_limit_models.append(model_name)
                        print(f"⚠️  {model_name} hit rate limit, trying next model...")
                        last_error = f"Rate limit on {model_name}"
                        continue
                    else:
                        # Other error (model not found, etc.) - try next
                        last_error = f"{model_name}: {str(e)[:100]}"
                        continue
            
            # All models failed
            if rate_limit_models:
                raise Exception(
                    f"All Gemini models hit rate limits. Tried: {', '.join(rate_limit_models)}. "
                    f"Please wait or provide a different API key. Last error: {last_error}"
                )
            else:
                raise Exception(f"Could not use any Gemini model. Last error: {last_error}")
                
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _groq_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate using Groq API."""
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("Groq package not installed. Install with: pip install groq")
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
