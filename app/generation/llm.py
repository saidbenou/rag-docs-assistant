"""LLM factory: local Ollama or any OpenAI-compatible API."""
from langchain_core.language_models import BaseChatModel

from app.config import get_settings


def get_llm() -> BaseChatModel:
    s = get_settings()
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=s.ollama_model, base_url=s.ollama_base_url, temperature=s.temperature)
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.openai_model,
            api_key=s.openai_api_key,
            base_url=s.openai_base_url or None,
            temperature=s.temperature,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")
