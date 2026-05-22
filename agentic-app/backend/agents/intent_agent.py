"""
intent_agent.py — Classifies user intent using Gemini with structured JSON output.

Intent categories:
  summarize        → Summarize the content
  sentiment        → Sentiment analysis
  code_explain     → Explain / analyze code
  qa_rag           → Question answering over uploaded document
  transcribe       → Audio/video transcription
  youtube_transcript → Fetch + summarize YouTube video
  converse         → General conversational Q&A
  extract_only     → Just show the extracted text
  UNCLEAR          → Multiple plausible intents — needs clarification
"""
import json
import re
import time
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from backend.schemas.state import AgentState
from backend.config import get_settings
from backend.utils.llm_factory import get_llm
from backend.utils.logger import get_logger

logger = get_logger(__name__)

VALID_INTENTS = {
    "summarize", "sentiment", "code_explain", "qa_rag",
    "transcribe", "youtube_transcript", "converse", "extract_only", "UNCLEAR"
}

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a multi-modal AI agent.

Given extracted text from a document/audio/image and a user query, classify the user's intent.

Respond ONLY with a valid JSON object (no markdown, no extra text):
{
  "intent": "<one of: summarize | sentiment | code_explain | qa_rag | transcribe | youtube_transcript | converse | extract_only | UNCLEAR>",
  "confidence": <float 0.0-1.0>,
  "reason": "<one-line explanation>",
  "ambiguous_intents": ["<list if multiple plausible intents>"]
}

Rules:
- If only a file was uploaded and NO query given → "extract_only" (unless content clearly suggests another task)
- If confidence < 0.6 for a single intent → "UNCLEAR"
- Code detected in text without explicit instruction → "UNCLEAR" (clarify: explain or rewrite?)
- YouTube URL in query → "youtube_transcript"
- Audio file + no query → "transcribe"
- Specific question about document → "qa_rag"
- Generic "summarize", "tldr", "brief" → "summarize"
- "how does user feel", "positive/negative", "tone" → "sentiment"
- "explain code", "what does this do", "bugs" → "code_explain"
"""




def classify_intent(state: AgentState) -> AgentState:
    """
    LangGraph node: classifies user intent.
    Populates: intent, intent_confidence, is_intent_clear,
               ambiguous_intents, follow_up_question (if UNCLEAR), logs.
    """
    t0 = time.time()
    logs = list(state.get("logs", []))

    user_query = state.get("raw_input", "").strip()
    extracted_text = state.get("extracted_text", "").strip()
    file_type = state.get("file_type", "none")

    # Build context snippet (truncate to 2000 chars to avoid huge prompts)
    context_snippet = extracted_text[:2000] if extracted_text else "(no extracted text)"

    user_content = f"""File type: {file_type}
Extracted text (first 2000 chars):
{context_snippet}

User query: {user_query if user_query else "(none — only file uploaded)"}

Classify the intent."""

    logs.append(f"[intent_node] calling Gemini for intent classification")

    try:
        llm = get_llm(temperature=0.1)

        messages = [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        intent = parsed.get("intent", "UNCLEAR")
        confidence = float(parsed.get("confidence", 0.5))
        reason = parsed.get("reason", "")
        ambiguous = parsed.get("ambiguous_intents", [])

        # Validate intent label
        if intent not in VALID_INTENTS:
            intent = "UNCLEAR"
            confidence = 0.3
            ambiguous = ["converse", "extract_only"]

        is_clear = intent != "UNCLEAR" and confidence >= 0.6

        elapsed = round(time.time() - t0, 2)
        logs.append(f"[intent_node] intent={intent} confidence={confidence:.2f} ({elapsed}s) reason={reason}")

        return {
            **state,
            "intent": intent,
            "intent_confidence": confidence,
            "is_intent_clear": is_clear,
            "ambiguous_intents": ambiguous,
            "logs": logs,
        }

    except Exception as e:
        logs.append(f"[intent_node] ERROR: {e}")
        logger.error("intent_classification_failed", error=str(e))
        # Default to conversational on error
        return {
            **state,
            "intent": "converse",
            "intent_confidence": 0.5,
            "is_intent_clear": True,
            "ambiguous_intents": [],
            "logs": logs,
        }


def route_after_intent(state: AgentState) -> str:
    """Conditional edge: determines next node based on intent clarity."""
    if state.get("is_intent_clear", False):
        return "planner"
    return "followup"
