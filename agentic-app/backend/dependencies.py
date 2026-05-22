"""
dependencies.py — FastAPI dependency injection providers.
Provides singleton instances of Gemini model, Whisper model, and session store.
"""
from functools import lru_cache
from typing import Dict, Any
import google.generativeai as genai

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# In-memory session store: {session_id: AgentState dict}
_session_store: Dict[str, Any] = {}

# Whisper model singleton (lazy-loaded)
_whisper_model = None


def get_session_store() -> Dict[str, Any]:
    """Return the global in-memory session store."""
    return _session_store


def get_gemini_model():
    """
    Dependency: returns configured Gemini GenerativeModel.
    Configures the SDK once using GEMINI_API_KEY from settings.
    """
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(model_name=settings.gemini_model)
    return model


def get_whisper_model():
    """
    Dependency: lazy-load Whisper model (only on first audio request).
    Reuses the same model instance for the lifetime of the process.
    """
    global _whisper_model
    if _whisper_model is None:
        import whisper
        settings = get_settings()
        logger.info("whisper_model_loading", model=settings.whisper_model)
        _whisper_model = whisper.load_model(settings.whisper_model)
        logger.info("whisper_model_ready", model=settings.whisper_model)
    return _whisper_model
