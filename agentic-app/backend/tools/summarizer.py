"""
summarizer.py — Generates structured 3-tier summaries using Gemini.

Output format (strictly enforced by prompt):
  ONE-LINE SUMMARY: [...]

  KEY POINTS:
  • [Point 1]
  • [Point 2]
  • [Point 3]

  DETAILED SUMMARY:
  [5 sentences]
"""
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.utils.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from backend.schemas.state import AgentState
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

SUMMARIZE_SYSTEM = """You are a professional summarization assistant.
Summarize the given text in EXACTLY this format — no deviations:

ONE-LINE SUMMARY: [single sentence capturing the main idea]

KEY POINTS:
• [most important point]
• [second most important point]
• [third most important point]

DETAILED SUMMARY:
[Exactly 5 sentences providing a comprehensive overview. Be specific, not vague.]

Rules:
- Use the EXACT headers shown above
- Bullet points must use • (bullet character)
- DETAILED SUMMARY must be exactly 5 sentences
- Do not add any other sections"""


def run_summarizer(state: AgentState) -> str:
    """
    Executor tool: generates a structured 3-tier summary.
    Returns the formatted summary string.
    """
    text = state.get("extracted_text", "") or state.get("raw_input", "")
    if not text.strip():
        return "No text content available to summarize."

    # Truncate very long texts to ~8000 chars (~2000 tokens)
    if len(text) > 8000:
        text = text[:8000] + "\n\n[... content truncated for processing ...]"

    llm = get_llm(temperature=0.3)

    t0 = time.time()
    response = llm.invoke([
        SystemMessage(content=SUMMARIZE_SYSTEM),
        HumanMessage(content=f"Summarize the following text:\n\n{text}"),
    ])
    elapsed = round(time.time() - t0, 2)
    logger.info("summarizer_complete", chars=len(text), elapsed=elapsed)
    return response.content.strip()
