from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from app.config import settings


def get_llm():
    """
    Factory function — returns the correct LLM based on LLM_PROVIDER env var.

    In local dev:   LLM_PROVIDER=ollama  → ChatOllama (free, runs on your machine)
    In production:  LLM_PROVIDER=anthropic → ChatAnthropic (paid API)

    The caller never needs to know which one it got. This is the
    Open/Closed Principle: adding a new provider means adding a branch
    here, not touching any other file.
    """
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return ChatOllama(
            model=settings.ollama_model,
            temperature=0.2,
        )

    if provider == "anthropic":
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0.2,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
        f"Expected 'ollama' or 'anthropic'."
    )
