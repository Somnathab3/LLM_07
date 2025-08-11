"""
Configuration example for Ollama LLM integration
"""

from src.cdr.ai.llm_client import LLMConfig, LLMProvider

# Default Ollama configuration
OLLAMA_CONFIG = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model_name="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
    max_tokens=1000,
    timeout=30.0
)

# Alternative configurations for different use cases

# Fast responses (smaller model)
FAST_CONFIG = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model_name="mistral:7b",
    base_url="http://localhost:11434",
    temperature=0.1,
    max_tokens=500,
    timeout=15.0
)

# High accuracy (larger context)
ACCURATE_CONFIG = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model_name="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.05,  # Lower temperature for consistency
    max_tokens=1500,   # More tokens for detailed responses
    timeout=45.0       # Longer timeout for complex reasoning
)

# Remote Ollama server
REMOTE_CONFIG = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model_name="llama3.1:8b",
    base_url="http://your-ollama-server:11434",
    temperature=0.1,
    max_tokens=1000,
    timeout=60.0  # Longer timeout for remote connections
)

# Configuration for different models
MODEL_CONFIGS = {
    "llama3.1:8b": LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        temperature=0.1,
        max_tokens=1000
    ),
    "mistral:7b": LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="mistral:7b",
        temperature=0.1,
        max_tokens=800
    ),
    "codellama:7b": LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="codellama:7b",
        temperature=0.05,
        max_tokens=1200
    )
}

def get_config(model_name: str = "llama3.1:8b") -> LLMConfig:
    """Get configuration for specific model"""
    return MODEL_CONFIGS.get(model_name, OLLAMA_CONFIG)

def get_production_config() -> LLMConfig:
    """Get production-ready configuration"""
    return LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=1000,
        timeout=30.0
    )
