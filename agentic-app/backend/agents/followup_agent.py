"""
followup_agent.py — Generates a targeted clarification question when intent is UNCLEAR.

The generated question is:
- Single question only
- Friendly and specific to the ambiguous intents
- Never proceeds with any task
"""
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.utils.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from backend.schemas.state import AgentState
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

FOLLOWUP_SYSTEM_PROMPT = """You are a helpful AI assistant.
The user's intent is unclear. Generate EXACTLY ONE short, friendly clarification question.
Do NOT answer any question. Do NOT perform any task. Only ask for clarification.
Keep it under 30 words. Be specific about the options."""


def generate_followup(state: AgentState) -> AgentState:
    """
    LangGraph node: generates a clarification question.
    Populates: follow_up_question, logs.
    """
    t0 = time.time()
    logs = list(state.get("logs", []))

    ambiguous = state.get("ambiguous_intents", [])
    extracted_text = state.get("extracted_text", "")[:500]
    raw_input = state.get("raw_input", "")

    if ambiguous:
        options = " or ".join(ambiguous)
        user_content = (
            f"The user's intent could be: {options}.\n"
            f"Their message: '{raw_input}'\n"
            f"Content preview: '{extracted_text[:200]}'\n"
            f"Generate one clarification question."
        )
    else:
        user_content = (
            f"The user uploaded content and said: '{raw_input}'\n"
            f"Content preview: '{extracted_text[:200]}'\n"
            f"Ask what they'd like to do with it (summarize, find info, analyze, etc.)."
        )

    logs.append("[followup_node] generating clarification question")

    try:
        llm = get_llm(temperature=0.7)
        response = llm.invoke([
            SystemMessage(content=FOLLOWUP_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        question = response.content.strip()

    except Exception as e:
        logger.error("followup_generation_failed", error=str(e))
        # Friendly fallback question
        if ambiguous:
            options = " or ".join(ambiguous)
            question = f"I can see a few ways to help. Would you like me to {options}?"
        else:
            question = "I've processed your content! What would you like me to do — summarize it, answer a specific question, or something else?"

    elapsed = round(time.time() - t0, 2)
    logs.append(f"[followup_node] question generated ({elapsed}s): {question[:80]}...")

    return {
        **state,
        "follow_up_question": question,
        "is_intent_clear": False,
        "logs": logs,
    }
