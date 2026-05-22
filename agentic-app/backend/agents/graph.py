"""
graph.py — LangGraph StateGraph definition for the agentic pipeline.

Flow:
  START
    ↓
  [intent_node] — classify intent
    ↓
  route_after_intent()
    ├─ is_clear=True  → [planner_node] → [executor_node] → [formatter_node] → END
    └─ is_clear=False → [followup_node] → END (returns question to user)
"""
import time
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.schemas.state import AgentState
from backend.agents.intent_agent import classify_intent, route_after_intent
from backend.agents.followup_agent import generate_followup
from backend.agents.planner_agent import plan_execution
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ── Executor node (dispatches to the right tool based on intent) ──────────────

def execute_task(state: AgentState) -> AgentState:
    """
    Dispatcher node: calls the correct executor tool based on classified intent.
    Lazily imports tools to avoid circular imports.
    """
    intent = state.get("intent", "converse")
    logs = list(state.get("logs", []))
    logs.append(f"[executor_node] dispatching to intent={intent}")

    t0 = time.time()

    try:
        if intent == "summarize":
            from backend.tools.summarizer import run_summarizer
            result = run_summarizer(state)

        elif intent == "sentiment":
            from backend.tools.sentiment_analyzer import run_sentiment
            result = run_sentiment(state)

        elif intent == "code_explain":
            from backend.tools.code_explainer import run_code_explainer
            result = run_code_explainer(state)

        elif intent == "qa_rag":
            from backend.tools.qa_rag import run_qa_rag
            result = run_qa_rag(state)

        elif intent in ("transcribe", "youtube_transcript"):
            from backend.tools.summarizer import run_summarizer
            # Text already extracted; summarize the transcript
            result = run_summarizer(state)

        elif intent == "extract_only":
            text = state.get("extracted_text", "")
            wc = len(text.split())
            result = (
                f"EXTRACTED TEXT:\n{text}\n\n"
                f"Word count: {wc} | "
                f"Confidence: {state.get('extraction_confidence', 1.0):.0%}"
            )

        else:  # converse / fallback
            from backend.tools.conversational import run_conversational
            result = run_conversational(state)

    except Exception as e:
        logger.error("executor_failed", intent=intent, error=str(e))
        result = f"I encountered an error while processing your request: {e}"
        logs.append(f"[executor_node] ERROR: {e}")

    elapsed = round(time.time() - t0, 2)
    logs.append(f"[executor_node] completed in {elapsed}s")

    return {**state, "final_output": result, "logs": logs}


# ── Formatter node ─────────────────────────────────────────────────────────────

def format_response(state: AgentState) -> AgentState:
    """
    Appends metadata footer to final output (OCR confidence, token cost, etc.)
    """
    logs = list(state.get("logs", []))
    output = state.get("final_output", "")

    meta_parts = []
    conf = state.get("extraction_confidence")
    if conf is not None and conf < 1.0:
        meta_parts.append(f"OCR confidence: {conf:.0%}")

    cost = state.get("estimated_cost")
    if cost:
        meta_parts.append(f"Est. cost: ${cost:.6f}")

    if meta_parts:
        output = output + "\n\n---\n_" + " | ".join(meta_parts) + "_"

    logs.append("[formatter_node] response formatted")

    return {**state, "final_output": output, "logs": logs}


# ── Build and compile the graph ────────────────────────────────────────────────

def build_graph():
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("intent", classify_intent)
    builder.add_node("followup", generate_followup)
    builder.add_node("planner", plan_execution)
    builder.add_node("executor", execute_task)
    builder.add_node("formatter", format_response)

    # Entry point
    builder.add_edge(START, "intent")

    # Conditional routing after intent classification
    builder.add_conditional_edges(
        "intent",
        route_after_intent,
        {"planner": "planner", "followup": "followup"},
    )

    # Followup → END (return question to user, no task execution)
    builder.add_edge("followup", END)

    # Happy path: planner → executor → formatter → END
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "formatter")
    builder.add_edge("formatter", END)

    # Compile with in-memory checkpointer for multi-turn sessions
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("langgraph_compiled")
    return graph


# Singleton graph instance
_graph = None


def get_graph():
    """Return the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
