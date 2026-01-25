"""
LLM interface for Intent2Model.

Abstract interface for LLM providers (Gemini, OpenAI, Groq).
"""

from typing import Optional
import os


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
        """Generate using Gemini API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = model.generate_content(full_prompt)
            return response.text
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
