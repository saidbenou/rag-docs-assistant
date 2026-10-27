"""Centralised configuration, read from environment variables / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Data
    data_dir: str = "data/pdfs"
    chroma_persist_dir: str = "data/chroma"
    collection_name: str = "documents"
    chroma_host: str | None = None
    chroma_port: int = 8000

    # Chunking & retrieval
    embedding_model: str = "BAAI/bge-m3"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5

    # LLM
    llm_provider: str = "ollama"  # "ollama" | "openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Langfuse monitoring (disabled while the keys are empty)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Ragas evaluation: "judge" LLM (defaults to the assistant's own LLM)
    eval_llm_provider: str | None = None  # "ollama" | "openai"
    eval_model: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
