"""
vector_store.py — FAISS index management for RAG pipeline.
Stores per-session FAISS indices in memory.
"""
import numpy as np
from typing import List, Tuple, Dict
import faiss

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# In-memory store: {session_id: (faiss_index, chunks_list)}
_indexes: Dict[str, Tuple[faiss.Index, List[str]]] = {}


def build_index(session_id: str, chunks: List[str], embeddings: List[List[float]]) -> None:
    """
    Build a FAISS flat L2 index for a session's document chunks.

    Args:
        session_id: Unique session identifier.
        chunks: Original text chunks (stored for retrieval).
        embeddings: Embedding vectors for each chunk.
    """
    if not embeddings:
        logger.warning("no_embeddings_to_index", session_id=session_id)
        return

    vectors = np.array(embeddings, dtype=np.float32)
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    _indexes[session_id] = (index, chunks)
    logger.info("faiss_index_built", session_id=session_id, chunks=len(chunks), dim=dimension)


def search_index(
    session_id: str,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Search the FAISS index for the top-k most similar chunks.

    Args:
        session_id: The session to search in.
        query_embedding: Embedding of the user's query.
        top_k: Number of chunks to retrieve.

    Returns:
        List of (chunk_text, distance_score) tuples, closest first.
    """
    if session_id not in _indexes:
        logger.warning("faiss_index_not_found", session_id=session_id)
        return []

    index, chunks = _indexes[session_id]
    query_vec = np.array([query_embedding], dtype=np.float32)

    distances, indices = index.search(query_vec, min(top_k, len(chunks)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0 and idx < len(chunks):
            results.append((chunks[idx], float(dist)))

    logger.info("faiss_search_complete", session_id=session_id, results=len(results))
    return results


def has_index(session_id: str) -> bool:
    """Check if a FAISS index exists for this session."""
    return session_id in _indexes


def delete_index(session_id: str) -> None:
    """Remove the FAISS index for a session (cleanup)."""
    if session_id in _indexes:
        del _indexes[session_id]
        logger.info("faiss_index_deleted", session_id=session_id)
