"""
state.py — LangGraph shared AgentState TypedDict.
This is the single object passed between every node in the graph.
"""
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # ── Session ────────────────────────────────────────────────────────────────
    session_id: str

    # ── Input ──────────────────────────────────────────────────────────────────
    raw_input: str                  # User's typed message
    extracted_text: str             # Text from OCR / ASR / PDF / YouTube
    extraction_confidence: float    # 0.0–1.0
    file_type: str                  # "text"|"image"|"pdf"|"audio"|"youtube"|"none"
    file_metadata: Dict[str, Any]   # page_count, duration, etc.

    # ── Intent ─────────────────────────────────────────────────────────────────
    intent: str                     # Classified intent label
    intent_confidence: float        # Model's confidence in the intent
    is_intent_clear: bool
    ambiguous_intents: List[str]    # Alternatives when unclear
    follow_up_question: Optional[str]  # Generated question when UNCLEAR

    # ── Execution ──────────────────────────────────────────────────────────────
    execution_plan: List[str]       # Ordered step descriptions
    tool_results: Dict[str, Any]    # Results keyed by tool name
    final_output: str               # Text-only response to user

    # ── Observability ──────────────────────────────────────────────────────────
    logs: List[str]                 # Execution trace for explainability
    estimated_cost: float           # Bonus: token cost estimate

    # ── Conversation history ───────────────────────────────────────────────────
    messages: List[BaseMessage]     # Full conversation history
