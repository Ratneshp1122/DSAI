"""
chat.py — POST /chat endpoint — runs the LangGraph agent pipeline.

Also exposes:
  GET /chat/session/{session_id} — retrieve session info
  DELETE /chat/session/{session_id} — clear session
"""
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from langchain_core.messages import HumanMessage, AIMessage

from backend.schemas.request import ChatRequest
from backend.schemas.response import AgentResponse
from backend.schemas.state import AgentState
from backend.agents.graph import get_graph
from backend.dependencies import get_session_store
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _build_initial_state(request: ChatRequest, session_data: dict) -> AgentState:
    """Construct AgentState from the request and stored session data."""
    return AgentState(
        session_id=request.session_id,
        raw_input=request.message,
        extracted_text=session_data.get("extracted_text", ""),
        extraction_confidence=session_data.get("extraction_confidence", 1.0),
        file_type=session_data.get("file_type", "none"),
        file_metadata=session_data.get("file_metadata", {}),
        intent="",
        intent_confidence=0.0,
        is_intent_clear=False,
        ambiguous_intents=[],
        follow_up_question=None,
        execution_plan=[],
        tool_results={},
        final_output="",
        logs=[],
        estimated_cost=0.0,
        messages=session_data.get("messages", []),
    )


@router.post("/chat", response_model=AgentResponse, tags=["chat"])
async def chat(
    request: ChatRequest,
    session_store: dict = Depends(get_session_store),
) -> AgentResponse:
    """
    Run the LangGraph agent pipeline on a user message.
    session_id must be from a prior /upload response, or a previous /chat response.
    """
    t_start = time.time()
    settings = get_settings()

    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured. Add it to your .env file."
        )

    # Retrieve or create session
    session_data = session_store.get(request.session_id)

    if not session_data:
        # Fresh text-only session (no file)
        session_data = {
            "file_id": None,
            "file_type": "none",
            "extracted_text": "",
            "extraction_confidence": 1.0,
            "file_metadata": {},
            "messages": [],
        }
        session_store[request.session_id] = session_data

    logger.info(
        "chat_request",
        session_id=request.session_id,
        message_preview=request.message[:80],
        file_type=session_data.get("file_type", "none"),
    )

    # Build initial state
    state = _build_initial_state(request, session_data)

    # Run the LangGraph
    graph = get_graph()
    config = {"configurable": {"thread_id": request.session_id}}

    try:
        final_state = graph.invoke(state, config=config)
    except Exception as e:
        err_str = str(e)
        logger.error("graph_invoke_failed", error=err_str, session_id=request.session_id)
        if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Your Gemini API key is invalid. "
                    "Get a free key at https://aistudio.google.com/app/apikey "
                    "and update GEMINI_API_KEY in your .env file."
                )
            )
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exhausted. This means your API key has no remaining quota. "
                    "Fix: 1) Get a FREE key from https://aistudio.google.com/app/apikey (not GCP Console). "
                    "2) Or wait and retry in ~1 minute. "
                    "3) Or set GEMINI_MODEL=gemini-1.5-flash-8b in your .env file."
                )
            )
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {err_str}")

    # Update conversation history in session
    messages = session_data.get("messages", [])
    messages.append(HumanMessage(content=request.message))

    if final_state.get("final_output"):
        messages.append(AIMessage(content=final_state["final_output"]))

    session_store[request.session_id]["messages"] = messages

    # Determine response status
    if not final_state.get("is_intent_clear") and final_state.get("follow_up_question"):
        status = "need_clarification"
    elif final_state.get("final_output"):
        status = "complete"
    else:
        status = "error"

    elapsed = round(time.time() - t_start, 2)
    logger.info(
        "chat_complete",
        session_id=request.session_id,
        status=status,
        intent=final_state.get("intent"),
        elapsed=elapsed,
    )

    return AgentResponse(
        session_id=request.session_id,
        status=status,
        extracted_text=final_state.get("extracted_text") or None,
        extraction_confidence=final_state.get("extraction_confidence"),
        file_type=final_state.get("file_type"),
        intent=final_state.get("intent") or None,
        intent_confidence=final_state.get("intent_confidence"),
        follow_up_question=final_state.get("follow_up_question"),
        execution_plan=final_state.get("execution_plan") or None,
        result=final_state.get("final_output") or None,
        logs=final_state.get("logs", []),
        estimated_cost=final_state.get("estimated_cost"),
    )


@router.get("/chat/session/{session_id}", tags=["chat"])
async def get_session(
    session_id: str,
    session_store: dict = Depends(get_session_store),
):
    """Retrieve session metadata (file type, word count, message count)."""
    data = session_store.get(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "file_type": data.get("file_type"),
        "filename": data.get("filename"),
        "word_count": data.get("file_metadata", {}).get("word_count", 0),
        "message_count": len(data.get("messages", [])),
    }


@router.delete("/chat/session/{session_id}", tags=["chat"])
async def clear_session(
    session_id: str,
    session_store: dict = Depends(get_session_store),
):
    """Clear a session and its stored file index."""
    if session_id in session_store:
        # Clean up RAG index if exists
        try:
            from backend.rag.vector_store import delete_index
            delete_index(session_id)
        except Exception:
            pass
        del session_store[session_id]
    return {"message": "Session cleared", "session_id": session_id}


@router.post("/chat/text", response_model=AgentResponse, tags=["chat"])
async def chat_text_only(
    request: ChatRequest,
    session_store: dict = Depends(get_session_store),
) -> AgentResponse:
    """
    Text-only chat — no file required.
    Auto-creates a session if session_id is not in the store.
    """
    return await chat(request, session_store)
