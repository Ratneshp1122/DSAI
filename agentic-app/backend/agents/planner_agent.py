"""
planner_agent.py — Generates an ordered execution plan and cost estimate.

Given a classified intent, produces:
  - Step-by-step plan (human-readable tool names)
  - Cost estimate for the upcoming LLM calls
"""
import time
from backend.schemas.state import AgentState
from backend.utils.cost_estimator import estimate_cost
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Static execution plans per intent
INTENT_PLANS = {
    "summarize": [
        "Step 1: [text_prep] Prepare extracted text",
        "Step 2: [summarizer] Generate 1-liner + 3 key points + 5-sentence summary",
        "Step 3: [formatter] Format structured output",
    ],
    "sentiment": [
        "Step 1: [text_prep] Prepare extracted text",
        "Step 2: [sentiment_analyzer] Classify sentiment label + confidence + justification",
        "Step 3: [formatter] Format structured output",
    ],
    "code_explain": [
        "Step 1: [text_prep] Extract code from content",
        "Step 2: [code_explainer] Detect language + explain + find bugs + calculate complexity",
        "Step 3: [formatter] Format structured output",
    ],
    "qa_rag": [
        "Step 1: [rag_indexer] Chunk text and build FAISS vector index",
        "Step 2: [query_embedder] Embed user query",
        "Step 3: [faiss_retriever] Retrieve top-5 most relevant chunks",
        "Step 4: [qa_rag] Generate answer grounded in retrieved context",
        "Step 5: [formatter] Format answer with source references",
    ],
    "transcribe": [
        "Step 1: [audio_transcriber] Transcribe audio via faster-whisper",
        "Step 2: [summarizer] Generate summary of transcript",
        "Step 3: [formatter] Format with metadata (language, duration)",
    ],
    "youtube_transcript": [
        "Step 1: [youtube_fetcher] Detect URL and fetch transcript",
        "Step 2: [summarizer] Summarize transcript",
        "Step 3: [formatter] Format output",
    ],
    "converse": [
        "Step 1: [context_builder] Inject conversation history",
        "Step 2: [conversational] Generate contextual response",
        "Step 3: [formatter] Return response",
    ],
    "extract_only": [
        "Step 1: [text_prep] Return extracted text as-is",
        "Step 2: [formatter] Format with metadata (confidence, word count)",
    ],
}


def plan_execution(state: AgentState) -> AgentState:
    """
    LangGraph node: generates execution plan and cost estimate.
    Populates: execution_plan, estimated_cost, logs.
    """
    t0 = time.time()
    logs = list(state.get("logs", []))
    intent = state.get("intent", "converse")

    plan = INTENT_PLANS.get(intent, INTENT_PLANS["converse"])

    # Cost estimation
    text = state.get("extracted_text", "") + " " + state.get("raw_input", "")
    cost_info = estimate_cost(text=text, intent=intent)
    estimated_cost = cost_info["estimated_cost_usd"]

    elapsed = round(time.time() - t0, 2)
    logs.append(
        f"[planner_node] intent={intent} steps={len(plan)} "
        f"~{cost_info['estimated_input_tokens']} tokens "
        f"≈${estimated_cost:.6f} ({elapsed}s)"
    )

    logger.info(
        "execution_plan_generated",
        intent=intent,
        steps=len(plan),
        estimated_cost=estimated_cost,
    )

    return {
        **state,
        "execution_plan": plan,
        "estimated_cost": estimated_cost,
        "logs": logs,
    }
