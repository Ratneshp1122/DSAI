"""
cost_estimator.py — Token cost calculator for Gemini API calls.
Estimates cost BEFORE execution so the UI can display it.
"""
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Gemini 1.5 Flash pricing (per 1K tokens, as of 2024)
PRICING = {
    "gemini-1.5-flash": {
        "input_per_1k": 0.000075,
        "output_per_1k": 0.0003,
    },
    "gemini-1.5-pro": {
        "input_per_1k": 0.00125,
        "output_per_1k": 0.005,
    },
}

# Overhead prompt tokens per intent type
PROMPT_OVERHEAD = {
    "summarize": 200,
    "sentiment": 150,
    "code_explain": 250,
    "qa_rag": 800,  # includes retrieved RAG chunks
    "transcribe": 100,
    "youtube_transcript": 200,
    "converse": 100,
    "extract_only": 50,
    "UNCLEAR": 100,
}


def estimate_cost(
    text: str,
    intent: str,
    model: str = "gemini-1.5-flash",
    estimated_output_tokens: int = 500,
) -> dict:
    """
    Estimates Gemini API cost before execution.

    Args:
        text: The extracted/input text to process.
        intent: The classified intent (determines prompt overhead).
        model: Gemini model name to use for pricing.
        estimated_output_tokens: Expected output tokens (default 500).

    Returns:
        Dict with token counts and estimated USD cost.
    """
    # Rough word→token conversion: ~1.3 tokens per word
    word_count = len(text.split())
    input_tokens_from_text = int(word_count * 1.3)

    overhead = PROMPT_OVERHEAD.get(intent, 200)
    total_input_tokens = input_tokens_from_text + overhead

    pricing = PRICING.get(model, PRICING["gemini-1.5-flash"])
    cost = (
        (total_input_tokens / 1000) * pricing["input_per_1k"]
        + (estimated_output_tokens / 1000) * pricing["output_per_1k"]
    )

    result = {
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_cost_usd": round(cost, 6),
        "model": model,
    }
    logger.debug("cost_estimated", **result)
    return result
