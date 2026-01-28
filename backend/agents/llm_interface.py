"""
LLM interface for Intent2Model.

Abstract interface for LLM providers (Gemini, OpenAI, Groq).
"""

from typing import Optional
import os
import subprocess
import shlex
import shutil

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
        # Use provided key, or get from api_key_manager (which uses .env by default)
        if api_key:
            self.api_key = api_key
        else:
            from utils.api_key_manager import get_api_key
            self.api_key = get_api_key(provider=provider) or ""
    
    def _get_api_key(self) -> str:
        """Get API key from api_key_manager (uses .env by default)."""
        from utils.api_key_manager import get_api_key
        return get_api_key(provider=self.provider) or ""
    
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
        elif self.provider == "gemini_cli":
            return self._gemini_cli_generate(prompt, system_prompt)
        elif self.provider == "groq":
            return self._groq_generate(prompt, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _gemini_cli_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate using a local Gemini CLI (no API key usage from our backend).
        
        This is intentionally generic because Gemini CLI tools vary by installation.
        Configure with env:
          - GEMINI_CLI_CMD (default: "gemini")
          - GEMINI_CLI_ARGS (default: "")
        
        We pass a single combined prompt via STDIN and read STDOUT.
        """
        # Prefer explicit GEMINI_CLI_CMD, else try Homebrew path on macOS, else "gemini"
        default_cmd = "/opt/homebrew/bin/gemini" if os.path.exists("/opt/homebrew/bin/gemini") else "gemini"
        cmd = os.getenv("GEMINI_CLI_CMD", default_cmd).strip() or default_cmd
        # Default args: non-interactive + read-only (no tool approvals)
        args_str = os.getenv("GEMINI_CLI_ARGS", "--approval-mode plan").strip()
        args = shlex.split(args_str) if args_str else []

        exe = shutil.which(cmd)
        if not exe:
            raise Exception(
                f"Gemini CLI not found: '{cmd}'. Install it or set GEMINI_CLI_CMD to the correct binary."
            )

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Track model/provider for UI transparency
        global _current_model_name, _current_model_reason
        _current_model_name = f"{cmd} (cli)"
        _current_model_reason = "Using local Gemini CLI (no API key consumed by this backend)"

        try:
            # Gemini CLI can auth via:
            # - OAuth (your Google account login in the CLI) OR
            # - API key (GOOGLE_API_KEY)
            #
            # Default: DO NOT inject API keys so OAuth-based CLI sessions work.
            # If you want to force key auth, set GEMINI_CLI_AUTH_MODE=api_key
            auth_mode = (os.getenv("GEMINI_CLI_AUTH_MODE", "oauth") or "oauth").strip().lower()
            child_env = os.environ.copy()
            if auth_mode == "api_key":
                from utils.api_key_manager import get_api_key
                gemini_key = get_api_key(provider="gemini") or ""
                if gemini_key and not child_env.get("GOOGLE_API_KEY"):
                    child_env["GOOGLE_API_KEY"] = gemini_key
                if gemini_key and not child_env.get("GEMINI_API_KEY"):
                    child_env["GEMINI_API_KEY"] = gemini_key

            # Use -p to ensure one-shot prompt execution (avoid interactive UI)
            proc = subprocess.run(
                [exe, *args, "-p", full_prompt],
                input="",
                text=True,
                capture_output=True,
                env=child_env,
                timeout=int(os.getenv("GEMINI_CLI_TIMEOUT_SEC", "180")),
            )
        except subprocess.TimeoutExpired:
            raise Exception("Gemini CLI timed out")
        except Exception as e:
            raise Exception(f"Gemini CLI error: {str(e)}")

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            raise Exception(f"Gemini CLI failed (exit={proc.returncode}): {err[:400] or 'unknown error'}")
        if not out:
            raise Exception(f"Gemini CLI returned empty output. stderr={err[:400]}")
        return out
    
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
                    error_full = str(e)
                    
                    # Check if it's a rate limit error
                    is_rate_limit = (
                        '429' in error_full or 
                        'quota' in error_str or 
                        'rate limit' in error_str or
                        'resourceexhausted' in error_str or
                        'exceeded' in error_str
                    )
                    
                    # Check if model doesn't exist
                    is_model_not_found = (
                        'not found' in error_str or
                        '404' in error_full or
                        'not supported' in error_str
                    )
                    
                    if is_rate_limit:
                        rate_limit_models.append(model_name)
                        print(f"⚠️  {model_name} hit rate limit, trying next model...")
                        last_error = f"Rate limit on {model_name}"
                        continue
                    elif is_model_not_found:
                        # Model doesn't exist, skip it
                        print(f"⚠️  {model_name} not available, trying next model...")
                        last_error = f"Model {model_name} not found"
                        continue
                    else:
                        # Other error - log but try next
                        print(f"⚠️  {model_name} error: {str(e)[:100]}, trying next model...")
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
