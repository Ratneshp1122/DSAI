"""
embedder.py — Embedding model wrapper for RAG pipeline.
Uses sentence-transformers/all-MiniLM-L6-v2 locally (free, fast, no API cost).
Falls back to Google text-embedding-004 if configured.
"""
from functools import lru_cache
from typing import List
from backend.utils.logger import get_logger

logger = get_logger(__name__)

LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_local_model():
    """Load sentence-transformers model once (singleton)."""
    from sentence_transformers import SentenceTransformer
    logger.info("embedding_model_loading", model=LOCAL_MODEL_NAME)
    model = SentenceTransformer(LOCAL_MODEL_NAME)
    logger.info("embedding_model_ready", model=LOCAL_MODEL_NAME)
    return model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text chunks.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []

    model = _get_local_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    logger.info("texts_embedded", count=len(texts))
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Args:
        query: The user's search query.

    Returns:
        Embedding vector.
    """
    model = _get_local_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding[0].tolist()
