"""
llm_factory.py — Centralised LangChain LLM builder.

All agents and tools should call get_llm() instead of instantiating
ChatGoogleGenerativeAI directly. This ensures:
  - Model name is always sourced from config (one place to change)
  - API key is always sourced from config
  - Consistent temperature / retry settings across all tools
"""
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import get_settings


def get_llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """
    Return a configured Gemini LLM instance.
    Not cached — creates a new instance each call (stateless by design).
    Model name and API key always come from settings/env.
    """
    settings = get_settings()
    model = settings.gemini_model

    # Strip the 'models/' prefix if present — langchain adds it internally
    if model.startswith("models/"):
        model = model[len("models/"):]

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        max_retries=3,
    )
