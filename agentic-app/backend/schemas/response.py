"""
response.py — AgentResponse schema returned by the /chat endpoint.
"""
from typing import Optional, List, Literal
from pydantic import BaseModel


class AgentResponse(BaseModel):
    session_id: str
    status: Literal["complete", "need_clarification", "error"]

    # Extraction info
    extracted_text: Optional[str] = None
    extraction_confidence: Optional[float] = None
    file_type: Optional[str] = None

    # Agent reasoning
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    follow_up_question: Optional[str] = None   # Present when status == "need_clarification"

    # Execution
    execution_plan: Optional[List[str]] = None
    result: Optional[str] = None               # Final text-only output

    # Observability
    logs: List[str] = []
    estimated_cost: Optional[float] = None
    duration_seconds: Optional[float] = None   # For audio files

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "session_id": "abc-123",
                "status": "complete",
                "intent": "summarize",
                "result": "ONE-LINE SUMMARY: ...",
                "logs": ["[intent_node] classified as summarize (0.95)"]
            },
            {
                "session_id": "abc-123",
                "status": "need_clarification",
                "follow_up_question": "Should I summarize or analyze sentiment?"
            }
        ]
    }}
