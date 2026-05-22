"""
conversational.py — General-purpose conversational Q&A using conversation history.
"""
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.utils.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from backend.schemas.state import AgentState
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

CONVERSE_SYSTEM = """You are a helpful, knowledgeable AI assistant.
Answer the user's question clearly and concisely.
If context from an uploaded document is provided, prioritize that information.
Be direct and specific. Format your response with clear paragraphs."""


def run_conversational(state: AgentState) -> str:
    """
    Executor tool: general conversational response using history + context.
    """
    user_query = state.get("raw_input", "")
    extracted_text = state.get("extracted_text", "")
    messages_history = state.get("messages", [])

    if not user_query.strip():
        return "Please ask me a question and I'll be happy to help!"

    llm = get_llm(temperature=0.7)

    # Build message list
    system_content = CONVERSE_SYSTEM
    if extracted_text:
        system_content += (
            f"\n\nDocument context available:\n{extracted_text[:3000]}"
            + (" [truncated]" if len(extracted_text) > 3000 else "")
        )

    messages = [SystemMessage(content=system_content)]

    # Add recent conversation history (last 6 exchanges)
    for msg in messages_history[-12:]:
        if hasattr(msg, "type"):
            if msg.type == "human":
                messages.append(HumanMessage(content=msg.content))
            elif msg.type == "ai":
                messages.append(AIMessage(content=msg.content))

    messages.append(HumanMessage(content=user_query))

    t0 = time.time()
    response = llm.invoke(messages)
    elapsed = round(time.time() - t0, 2)
    logger.info("conversational_complete", elapsed=elapsed)
    return response.content.strip()
