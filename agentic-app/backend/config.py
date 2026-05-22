"""
config.py — Application settings using Pydantic BaseSettings.
Reads from environment variables or .env file automatically.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API Keys
    gemini_api_key: str = ""  # Set in .env — validated at request time, not startup

    # Model Config
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/text-embedding-004"
    whisper_model: str = "base"

    # App Config
    log_level: str = "INFO"
    session_ttl_seconds: int = 3600
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 50

    # RAG Config
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_top_k: int = 5

    # LLM Retry Config
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — reads .env once."""
    return Settings()
