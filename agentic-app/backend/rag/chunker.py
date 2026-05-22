"""
chunker.py — Text splitter for RAG pipeline.
Uses LangChain's RecursiveCharacterTextSplitter with 512-token chunks and 50-token overlap.
"""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Per the plan: chunk_size=512 tokens (~2000 chars), chunk_overlap=50 tokens (~200 chars)
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
)


def chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping chunks for vector indexing.

    Args:
        text: Full document text.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    chunks = _splitter.split_text(text)
    logger.info("text_chunked", total_chunks=len(chunks), text_length=len(text))
    return chunks
