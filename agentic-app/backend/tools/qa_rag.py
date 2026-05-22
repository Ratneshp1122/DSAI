"""
qa_rag.py — RAG-powered Question Answering executor tool.

Pipeline:
  1. Chunk extracted text
  2. Embed chunks + build FAISS index
  3. Embed user query
  4. Retrieve top-5 chunks
  5. Generate grounded answer with Gemini using retrieved context
"""
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.utils.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from backend.schemas.state import AgentState
from backend.rag.chunker import chunk_text
from backend.rag.embedder import embed_texts, embed_query
from backend.rag.vector_store import build_index, search_index, has_index
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

QA_SYSTEM = """You are a precise question-answering assistant.
Answer the user's question using ONLY the provided context chunks.
If the answer is not in the context, say "I couldn't find that information in the provided document."

Format your response:
ANSWER: <direct answer to the question>

EVIDENCE:
• <relevant quote from context chunk 1>
• <relevant quote from context chunk 2>

CONFIDENCE: <High | Medium | Low>
CONFIDENCE REASON: <one sentence explaining your confidence>"""


def run_qa_rag(state: AgentState) -> str:
    """
    Executor tool: RAG-powered QA over uploaded document.
    Returns the formatted answer string.
    """
    session_id = state.get("session_id", "default")
    query = state.get("raw_input", "").strip()
    text = state.get("extracted_text", "").strip()

    if not query:
        return "Please provide a question to search for in the document."
    if not text:
        return "No document content available. Please upload a file first."

    t0 = time.time()

    # Build FAISS index if not cached for this session
    if not has_index(session_id):
        logger.info("rag_building_index", session_id=session_id)
        chunks = chunk_text(text)
        if not chunks:
            return "The document could not be processed into searchable chunks."
        embeddings = embed_texts(chunks)
        build_index(session_id, chunks, embeddings)
    else:
        logger.info("rag_using_cached_index", session_id=session_id)

    # Embed query and search
    query_vec = embed_query(query)
    results = search_index(session_id, query_vec, top_k=5)

    if not results:
        return "No relevant chunks found in the document for your query."

    # Build context from retrieved chunks
    context_parts = []
    for i, (chunk, score) in enumerate(results, 1):
        context_parts.append(f"[Chunk {i}] (relevance score: {score:.3f})\n{chunk}")
    context = "\n\n".join(context_parts)

    # Generate grounded answer
    llm = get_llm(temperature=0.2)

    response = llm.invoke([
        SystemMessage(content=QA_SYSTEM),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])

    elapsed = round(time.time() - t0, 2)
    logger.info("rag_qa_complete", session_id=session_id, chunks_retrieved=len(results), elapsed=elapsed)
    return response.content.strip()
