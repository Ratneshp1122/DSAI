"""
code_explainer.py — Analyzes code: language detection, explanation, bug detection, complexity.

Output format:
  LANGUAGE: [detected language]
  EXPLANATION: [what the code does, line by line if needed]
  POTENTIAL BUGS: [numbered list of issues found, or "None detected"]
  TIME COMPLEXITY: O(...)
  SPACE COMPLEXITY: O(...)
  SUGGESTIONS: [improvement ideas]
"""
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.utils.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from backend.schemas.state import AgentState
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

CODE_EXPLAIN_SYSTEM = """You are an expert software engineer and code reviewer.
Analyze the given code and respond in EXACTLY this format:

LANGUAGE: <detected programming language>

EXPLANATION:
<Clear explanation of what this code does. Walk through the key logic, data structures, and algorithms used. Be thorough but concise.>

POTENTIAL BUGS:
<Numbered list of bugs, edge cases, or issues. If none found, write "None detected.">
1. [Bug description + line reference if possible]
2. [Bug description]

TIME COMPLEXITY: O(<notation>) — <brief explanation>
SPACE COMPLEXITY: O(<notation>) — <brief explanation>

SUGGESTIONS:
<2-3 specific improvements for readability, performance, or safety>

Rules:
- Use EXACT section headers
- If no code is present, say "No code detected in the provided text."
- Never add markdown code fences in your response"""


def run_code_explainer(state: AgentState) -> str:
    """
    Executor tool: explains code, finds bugs, calculates complexity.
    Returns the formatted code analysis string.
    """
    text = state.get("extracted_text", "") or state.get("raw_input", "")
    if not text.strip():
        return "No code content available to analyze."

    if len(text) > 8000:
        text = text[:8000] + "\n[... truncated ...]"

    llm = get_llm(temperature=0.1)

    t0 = time.time()
    response = llm.invoke([
        SystemMessage(content=CODE_EXPLAIN_SYSTEM),
        HumanMessage(content=f"Analyze this code:\n\n{text}"),
    ])
    elapsed = round(time.time() - t0, 2)
    logger.info("code_explainer_complete", chars=len(text), elapsed=elapsed)
    return response.content.strip()
