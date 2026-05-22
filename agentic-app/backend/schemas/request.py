"""
request.py — FastAPI request/response schemas for upload and chat endpoints.
"""
from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier from /upload or prior /chat")
    message: str = Field(..., description="User's message or question")
    file_id: Optional[str] = Field(None, description="File ID if referring to an uploaded file")

    model_config = {"json_schema_extra": {
        "example": {
            "session_id": "abc-123",
            "message": "What are the action items?",
            "file_id": "def-456"
        }
    }}


class UploadResponse(BaseModel):
    file_id: str
    session_id: str
    file_type: str
    extracted_text: Optional[str]
    confidence: Optional[float]
    page_count: Optional[int]
    duration_seconds: Optional[float]
    word_count: Optional[int]
    method: Optional[str]
    warning: Optional[str]
    message: str = "File processed successfully"
