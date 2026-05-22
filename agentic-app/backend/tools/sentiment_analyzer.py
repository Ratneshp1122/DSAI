"""
sentiment_analyzer.py — Sentiment analysis with label, confidence, and justification.

Output format:
  SENTIMENT: Positive | Negative | Neutral | Mixed
  CONFIDENCE: X%
  EMOTIONS DETECTED: [comma-separated]
  JUSTIFICATION: [one paragraph explaining the sentiment]
  KEY PHRASES: [most sentiment-bearing phrases]
"""
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.utils.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from backend.schemas.state import AgentState
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

SENTIMENT_SYSTEM = """You are an expert sentiment analysis assistant.
Analyze the sentiment of the given text and respond in EXACTLY this format:

SENTIMENT: <Positive | Negative | Neutral | Mixed>
CONFIDENCE: <percentage, e.g. 87%>
EMOTIONS DETECTED: <comma-separated list of emotions, e.g. joy, frustration, hope>
JUSTIFICATION: <One paragraph (3-4 sentences) explaining WHY you classified it this way. Quote specific phrases.>
KEY PHRASES: <3-5 phrases from the text that most strongly indicate the sentiment>

Rules:
- Use EXACT headers
- CONFIDENCE must be a percentage (0-100%)
- EMOTIONS DETECTED must list at least 2 emotions
- JUSTIFICATION must reference actual text evidence"""


def run_sentiment(state: AgentState) -> str:
    """
    Executor tool: performs sentiment analysis.
    Returns the formatted sentiment report string.
    """
    text = state.get("extracted_text", "") or state.get("raw_input", "")
    if not text.strip():
        return "No text content available for sentiment analysis."

    if len(text) > 6000:
        text = text[:6000] + "\n[... truncated ...]"

    llm = get_llm(temperature=0.2)

    t0 = time.time()
    response = llm.invoke([
        SystemMessage(content=SENTIMENT_SYSTEM),
        HumanMessage(content=f"Analyze the sentiment of:\n\n{text}"),
    ])
    elapsed = round(time.time() - t0, 2)
    logger.info("sentiment_complete", chars=len(text), elapsed=elapsed)
    return response.content.strip()
