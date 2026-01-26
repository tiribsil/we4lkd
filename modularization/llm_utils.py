import os
import time
import re
from pathlib import Path
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from google import genai
from groq import Groq
from dotenv import load_dotenv
from typing import Optional, Protocol, Any
from utils import get_logger

class LLMInterface(Protocol):
    def create_completion(self, prompt: str, **kwargs) -> Any:
        ...

class GroqWrapper:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.logger = get_logger(self.__class__.__name__)

    def create_completion(self, prompt: str, **kwargs) -> dict:
        """Mimics llama-cpp-python output for compatibility."""
        max_retries = 5
        base_delay = 2
        
        for i in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", 10),
                    temperature=kwargs.get("temperature", 0.0),
                )
                return {
                    'choices': [{
                        'text': response.choices[0].message.content
                    }]
                }
            except Exception as e:
                error_str = str(e).upper()
                if "429" in error_str or "RATE_LIMIT" in error_str:
                    # Groq usually indicates retry time or we use exponential backoff
                    delay = base_delay * (2 ** i)
                    self.logger.warning(f"Groq Rate limit hit. Waiting {delay}s before retry {i+1}/{max_retries}...")
                    time.sleep(delay)
                else:
                    self.logger.warning(f"Groq API error attempt {i+1}: {e}")
                    if i < max_retries - 1:
                        time.sleep(1)
                    else:
                        raise e
        raise RuntimeError("Groq failed after multiple retries.")

class GeminiWrapper:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.logger = get_logger(self.__class__.__name__)

    def create_completion(self, prompt: str, **kwargs) -> dict:
        """Mimics llama-cpp-python output for compatibility or provides a simple wrapper."""
        max_retries = 5
        base_delay = 10  # Start with 10s delay for 429
        
        for i in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "max_output_tokens": kwargs.get("max_tokens", 10),
                        "temperature": kwargs.get("temperature", 0.0),
                    }
                )
                return {
                    'choices': [{
                        'text': response.text
                    }]
                }
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Parse retry delay if available in the error message
                    # Example: "Please retry in 12.19956213s."
                    retry_match = re.search(r'retry in ([\d\.]+)s', error_str)
                    delay = float(retry_match.group(1)) + 1.0 if retry_match else base_delay * (2 ** i)
                    
                    self.logger.warning(f"Rate limit hit. Waiting {delay:.2f}s before retry {i+1}/{max_retries}...")
                    time.sleep(delay)
                else:
                    self.logger.warning(f"Gemini API attempt {i+1} failed: {e}")
                    if i < max_retries - 1:
                        time.sleep(2 ** i)
                    else:
                        raise e
        
        raise RuntimeError(f"Failed to get response from Gemini after {max_retries} attempts due to rate limiting.")

class LLMManager:
    """Manages local GGUF models and LLM initialization."""
    
    DEFAULT_MODEL_DIR = Path("models")
    
    def __init__(self, model_repo: str, model_file: str, n_ctx: int = 2048, logger = None):
        self.logger = logger or get_logger(self.__class__.__name__)
        self.model_path = self.DEFAULT_MODEL_DIR / model_file
        self.model_repo = model_repo
        self.model_file = model_file
        self.n_ctx = n_ctx
        self._llm: Optional[Llama] = None

    def _ensure_model(self):
        if not self.model_path.exists():
            self.logger.info(f"Model not found at {self.model_path}. Downloading from {self.model_repo}...")
            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=self.model_repo,
                    filename=self.model_file,
                    local_dir=str(self.model_path.parent),
                    local_dir_use_symlinks=False
                )
                self.logger.info(f"Model downloaded successfully to {self.model_path}")
            except Exception as e:
                self.logger.error(f"Failed to download model: {e}")
                raise RuntimeError(f"Failed to download model: {e}")

    def get_llm(self, **kwargs) -> Llama:
        if self._llm is None:
            self._ensure_model()
            
            # Default parameters
            params = {
                "model_path": str(self.model_path),
                "n_ctx": self.n_ctx,
                "n_gpu_layers": 0, # CPU only by default
                "n_threads": os.cpu_count(),
                "verbose": False,
            }
            params.update(kwargs)
            
            self._llm = Llama(**params)
        return self._llm

def get_report_year_llm(engine: str = "groq") -> Any:
    """Helper to get the LLM for year extraction. engine can be 'groq', 'gemini', or 'local'."""
    if engine == "groq":
        return GroqWrapper()
    elif engine == "gemini":
        return GeminiWrapper()
    
    # Fallback to local 14B if explicitly requested
    manager = LLMManager(
        model_repo="bartowski/Qwen2.5-14B-Instruct-GGUF",
        model_file="Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        n_ctx=16384
    )
    return manager.get_llm()

def get_biomedical_llm(n_ctx: int = 32768) -> Llama:
    """Helper to get the BioMistral LLM for contextualization."""
    manager = LLMManager(
        model_repo="QuantFactory/BioMistral-7B-GGUF",
        model_file="BioMistral-7B.Q4_K_M.gguf",
        n_ctx=n_ctx
    )
    return manager.get_llm()
