# 🤖 Agentic Multi-Modal AI Application — Implementation Plan

> **Assignment Goal**: Build a production-grade agentic system that accepts Text, Images, PDFs, and Audio files, autonomously extracts content, understands user intent, and performs the correct task — all with text-only output.

---

## 📌 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Technology Stack & Theory Notes](#2-technology-stack--theory-notes)
3. [System Architecture](#3-system-architecture)
4. [Component Breakdown](#4-component-breakdown)
5. [Agent Orchestration Logic](#5-agent-orchestration-logic)
6. [RAG Pipeline Design](#6-rag-pipeline-design)
7. [Detailed Roadmap (Phase-by-Phase)](#7-detailed-roadmap-phase-by-phase)
8. [API Contract Design](#8-api-contract-design)
9. [UI/UX Design Specification](#9-uiux-design-specification)
10. [Error Handling & Robustness](#10-error-handling--robustness)
11. [Test Cases](#11-test-cases)
12. [Bonus: Multi-Agent & Cost Estimator](#12-bonus-multi-agent--cost-estimator)
13. [Deliverables Checklist](#13-deliverables-checklist)

---

## 1. Executive Summary

This system is a **multi-modal, intent-driven agentic application** built on FastAPI with a clean chat-like UI. It acts as an intelligent middleware:

```
[User Input: Text / Image / PDF / Audio]
         ↓
[Extraction Layer] → Raw Text
         ↓
[Intent Agent] → Understand Goal
         ↓ (if unclear)
[Follow-Up Question Mechanism]
         ↓ (once clear)
[Task Executor Agent] → Correct Task
         ↓
[Text-Only Output]
```

The system uses **Google Gemini** as the primary LLM backbone, **Tesseract OCR** for image/PDF text extraction, **Whisper** for speech-to-text, and **RAG** for document-heavy queries. Everything is served via **FastAPI** with a **Jinja2 + Vanilla JS** front-end.

---

## 2. Technology Stack & Theory Notes

### 2.1 FastAPI — *The Backend Backbone*

**What it is**: FastAPI is a modern Python web framework for building APIs, based on Pydantic and Starlette. It auto-generates OpenAPI docs and supports async natively.

**Why here**: The application is I/O-bound — it reads files, calls external APIs (Gemini, OCR), streams responses. FastAPI's `async/await` support allows it to serve hundreds of concurrent requests without blocking, unlike Flask's synchronous model. It also handles multipart file uploads out of the box.

**Key features used**:
- `APIRouter` for modular route separation
- `UploadFile` for streaming file uploads
- `BackgroundTasks` for non-blocking processing
- `WebSocket` for streaming chat responses
- `Depends()` for dependency injection (session manager, config)

---

### 2.2 Google Gemini API — *The LLM Brain*

**What it is**: Gemini is Google DeepMind's multi-modal large language model. The `gemini-1.5-flash` and `gemini-1.5-pro` variants support text, image, audio, and PDF inputs in a single call via the `google-generativeai` Python SDK.

**Why here**: Gemini's native multi-modal support means the system can optionally pass images directly to the model (not just OCR'd text). Its large context window (1M tokens in Pro) handles long PDFs. Its instruction-following capability drives the agent's planning and intent-detection layers.

**Key usage**:
- `model.generate_content([prompt, image_part])` — multi-modal input
- `GenerationConfig(response_mime_type="application/json")` — structured JSON outputs for intent classification
- Function calling / tool use for routing to the correct executor
- Streaming responses via `model.generate_content(..., stream=True)`

---

### 2.3 LangChain / LangGraph — *Agent Orchestration Framework*

**What it is**: LangChain is a framework for composing LLM-powered pipelines. LangGraph extends it by enabling stateful, graph-based multi-agent workflows where nodes represent agents and edges represent conditional routing logic.

**Why here**: The system requires **conditional branching**:
- If intent is clear → execute task
- If intent is unclear → ask follow-up
- If task is summarization → run summarizer
- If task is code explanation → run code explainer

LangGraph models this as a directed graph with **state machines**, making the orchestration logic explicit, testable, and maintainable — far superior to nested if/else chains.

**Key concepts**:
- `StateGraph` — defines the flow between agents
- `MessageState` — shared state object passed between nodes
- `conditional_edges` — route based on intent classification result
- `ToolNode` — wraps individual task-executor tools
- `Checkpointer` — persists conversation state for multi-turn follow-ups

---

### 2.4 Tesseract OCR (via `pytesseract`) — *Image Text Extraction*

**What it is**: Tesseract is an open-source OCR engine maintained by Google. `pytesseract` is the Python wrapper. It converts images to text using trained ML models.

**Why here**: When users upload JPG/PNG images containing text (screenshots, scanned docs), the system must extract readable text before the LLM can process it. Tesseract provides confidence scores per word, enabling the agent to flag low-confidence extractions.

**Key usage**:
- `pytesseract.image_to_string(img)` — basic extraction
- `pytesseract.image_to_data(img, output_type=Output.DICT)` — includes per-word confidence
- `PIL.Image` preprocessing (grayscale, threshold) before OCR for better accuracy

---

### 2.5 PyMuPDF (`fitz`) — *PDF Parsing*

**What it is**: PyMuPDF is a Python binding for MuPDF, a high-performance PDF rendering engine. It extracts text, images, and metadata from PDFs with layout preservation.

**Why here**: PDFs come in two flavors — **text-based** (searchable, directly parseable) and **scanned** (image-only, require OCR). PyMuPDF handles the text-based case natively. For scanned PDFs, it renders pages as images which are then fed to Tesseract. This dual-path approach gives maximum coverage.

**Key usage**:
- `fitz.open(filepath)` → `page.get_text("text")` — direct text extraction
- `page.get_pixmap()` → image rendering for OCR fallback
- `doc.metadata` — title, author, page count extraction

---

### 2.6 OpenAI Whisper — *Audio Speech-to-Text*

**What it is**: Whisper is an open-source automatic speech recognition (ASR) model by OpenAI, trained on 680,000 hours of multilingual audio. It runs locally via the `openai-whisper` package.

**Why here**: Processing audio locally (not via API) ensures privacy, avoids per-second API costs, and works offline. Whisper's `large-v3` model achieves state-of-the-art accuracy. For the assignment's use cases (5-min lectures), the `base` or `small` model gives fast, accurate results.

**Key usage**:
- `whisper.load_model("base")` — load model once at startup
- `model.transcribe(audio_path)` — returns `{"text": ..., "segments": [...], "language": ...}`
- Segments contain timestamps, enabling duration calculation

---

### 2.7 FAISS + LangChain — *RAG (Retrieval-Augmented Generation)*

**What it is**: RAG is a technique where relevant document chunks are retrieved from a vector store and injected into the LLM prompt to ground its answers in actual document content. FAISS (Facebook AI Similarity Search) is a high-performance vector similarity search library.

**Why here**: For large PDFs (meeting notes, research papers), passing the entire document to the LLM context is expensive and unreliable. RAG retrieves only the top-K most relevant chunks, improving accuracy, reducing token cost, and enabling answering specific questions like "What are the action items?"

**Pipeline**:
```
PDF Text → Chunker (512 tokens, 50 overlap)
         → Embedder (text-embedding-004 / sentence-transformers)
         → FAISS Index
         ↓
User Query → Embed Query → Similarity Search (top-5 chunks)
                         → Inject into LLM Prompt → Answer
```

---

### 2.8 YouTube Transcript API — *Transcript Fetching*

**What it is**: `youtube-transcript-api` is a Python library that fetches auto-generated or manual captions from YouTube videos without the YouTube Data API.

**Why here**: When a YouTube URL is detected in user input, the system fetches the transcript instead of downloading and transcribing the audio — this is faster (sub-second vs minutes) and free.

**Key usage**:
- URL detection via regex: `(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)`
- `YouTubeTranscriptApi.get_transcript(video_id)` → list of `{text, start, duration}`
- Fallback: if transcript unavailable, inform user with explanation

---

### 2.9 Pydantic v2 — *Data Validation & Schemas*

**What it is**: Pydantic provides runtime type validation for Python dataclasses-style models. FastAPI uses it natively for request/response validation.

**Why here**: Every agent's input and output is modeled as a Pydantic schema, ensuring type safety, auto-documentation, and clear contracts between components.

---

### 2.10 Redis + Celery (Optional/Production) — *Async Task Queue*

**What it is**: For long-running tasks (5-min audio transcription), synchronous HTTP requests would time out. Celery is a distributed task queue; Redis is the message broker.

**Why here (Bonus)**: Audio files > 30s should be processed asynchronously — the API returns a `task_id` immediately, and the client polls `/status/{task_id}` for results. This prevents browser timeouts and improves UX.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Chat Window │ File Upload │ Extraction Preview Panel   │   │
│   └─────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP / WebSocket
┌───────────────────────────▼─────────────────────────────────────┐
│                      FASTAPI GATEWAY                            │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│   │/chat POST│  │/upload   │  │/status   │  │/health GET   │   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    EXTRACTION LAYER                             │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐             │
│  │ Image → OCR  │ │ PDF → PyMuPDF│ │ Audio→Whisper│            │
│  │ (Tesseract)  │ │ + OCR fallback│ │             │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬──────┘            │
│         └────────────────┼────────────────┘                    │
│                          ▼                                      │
│                   Extracted Raw Text                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   INTENT AGENT (LangGraph)                      │
│                                                                 │
│  Input Context → Gemini → Intent Classification                 │
│                                                                 │
│  Intents: [summarize | sentiment | code_explain |               │
│            qa_rag | transcribe | youtube | converse |           │
│            extract_only | UNCLEAR]                              │
│                                                                 │
│  if UNCLEAR → Follow-Up Question → Wait for User Response       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Classified Intent
┌───────────────────────────▼─────────────────────────────────────┐
│               PLANNER AGENT (LangGraph Node)                    │
│                                                                 │
│  Generates execution plan:                                      │
│  Step 1: [tool_name], Step 2: [tool_name], ...                  │
│  Estimated token cost displayed (Bonus)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│             EXECUTOR AGENTS (Task-Specific Tools)               │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Summarizer  │  │  Sentiment   │  │    Code Explainer      │ │
│  │ - 1 liner   │  │  Analyzer    │  │  - Language detection  │ │
│  │ - 3 bullets │  │  - Label     │  │  - Bug detection       │ │
│  │ - 5 sentences│  │  - Confidence│  │  - Time complexity     │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ QA via RAG  │  │  YouTube     │  │  Conversational        │ │
│  │ (FAISS)     │  │  Transcript  │  │  Answer                │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    RESPONSE FORMATTER                           │
│  - Adds execution log / plan trace                             │
│  - Formats text-only output                                    │
│  - Appends OCR confidence / duration metadata                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Breakdown

### 4.1 Project Directory Structure

```
agentic-app/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings (API keys, model names)
│   ├── dependencies.py            # DI: session store, model loader
│   │
│   ├── routers/
│   │   ├── chat.py                # POST /chat, WebSocket /ws/chat
│   │   ├── upload.py              # POST /upload (file handling)
│   │   └── health.py              # GET /health
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── image_ocr.py           # Tesseract OCR + confidence
│   │   ├── pdf_parser.py          # PyMuPDF + OCR fallback
│   │   ├── audio_transcriber.py   # Whisper STT
│   │   └── youtube_fetcher.py     # YouTube Transcript API
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── intent_agent.py        # Intent classification node
│   │   ├── planner_agent.py       # Execution plan generator
│   │   ├── followup_agent.py      # Follow-up question generator
│   │   └── graph.py               # LangGraph StateGraph definition
│   │
│   ├── tools/                     # Executor tools (LangGraph ToolNodes)
│   │   ├── __init__.py
│   │   ├── summarizer.py          # 1-liner + bullets + 5-sentence
│   │   ├── sentiment_analyzer.py  # Label + confidence + justification
│   │   ├── code_explainer.py      # Language + bugs + complexity
│   │   ├── qa_rag.py              # FAISS + RAG pipeline
│   │   └── conversational.py     # General Q&A
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunker.py             # Text splitter
│   │   ├── embedder.py            # Embedding model wrapper
│   │   └── vector_store.py        # FAISS index management
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py             # ChatRequest, UploadResponse
│   │   ├── response.py            # AgentResponse, ExtractionResult
│   │   └── state.py               # LangGraph AgentState
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py              # Structured logging
│   │   ├── cost_estimator.py      # Token cost calculator (Bonus)
│   │   └── file_handler.py        # Temp file management
│   │
│   └── tests/
│       ├── test_extraction.py
│       ├── test_agents.py
│       ├── test_tools.py
│       └── test_api.py
│
├── frontend/
│   ├── templates/
│   │   └── index.html             # Jinja2 template
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js             # Chat UI logic
│
├── docs/
│   └── architecture.png           # Architecture diagram
│
├── .env                           # GEMINI_API_KEY, etc.
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

### 4.2 Core Data Schemas

```python
# schemas/state.py — LangGraph shared state
class AgentState(TypedDict):
    session_id: str
    raw_input: str                  # User typed text
    extracted_text: str             # From OCR/ASR/PDF
    extraction_confidence: float    # OCR confidence score
    file_type: str                  # "text" | "image" | "pdf" | "audio" | "youtube"
    intent: str                     # Classified intent
    is_intent_clear: bool
    follow_up_question: str         # If intent unclear
    execution_plan: List[str]       # Step-by-step plan
    tool_results: Dict[str, Any]    # Results from each tool
    final_output: str               # Text-only final response
    logs: List[str]                 # Execution trace for explainability
    estimated_cost: float           # Token cost estimate (Bonus)
    messages: List[BaseMessage]     # Conversation history

# schemas/response.py
class AgentResponse(BaseModel):
    session_id: str
    status: Literal["complete", "need_clarification", "error"]
    extracted_text: Optional[str]
    extraction_confidence: Optional[float]
    intent: Optional[str]
    follow_up_question: Optional[str]   # Present if status == "need_clarification"
    execution_plan: Optional[List[str]]
    result: Optional[str]               # Final text output
    logs: List[str]
    estimated_cost: Optional[float]
    duration_seconds: Optional[float]   # For audio files
```

---

## 5. Agent Orchestration Logic

### 5.1 LangGraph State Machine

The orchestration is defined as a directed graph:

```
START
  │
  ▼
[extract_node]           ← Runs appropriate extractor based on file_type
  │
  ▼
[intent_node]            ← Classifies intent using Gemini + structured output
  │
  ├── is_clear=True ──→  [planner_node]  ──→  [executor_node]  ──→  [formatter_node]  ──→  END
  │
  └── is_clear=False ──→ [followup_node] ──→  END (returns question to user)
                                ↑
                                │ (user responds in next turn)
                                └── re-enters graph at [intent_node] with clarification
```

### 5.2 Intent Classification Prompt

```
You are an intent classifier for a multi-modal AI agent.

Given the following extracted text and user query, classify the user's intent.

Output a JSON with:
{
  "intent": one of [summarize, sentiment, code_explain, qa_rag, transcribe,
                    youtube_transcript, converse, extract_only, UNCLEAR],
  "confidence": float 0-1,
  "reason": "one-line explanation",
  "ambiguous_intents": ["list if multiple plausible intents"]
}

Rules:
- If no query is given and only a file is uploaded → intent = "extract_only" unless
  the content obviously suggests another task.
- If multiple intents are equally plausible (confidence < 0.6) → intent = "UNCLEAR"
- Code detected in text without explicit instruction → intent = "UNCLEAR"
  (ask: "Should I explain this code or rewrite it?")
```

### 5.3 Follow-Up Question Logic

The `followup_node` generates a **single, targeted question** based on `ambiguous_intents`:

```python
def generate_followup(state: AgentState) -> AgentState:
    ambiguous = state["ambiguous_intents"]  # e.g., ["summarize", "sentiment"]
    options = " or ".join(ambiguous)
    prompt = f"""
    The user's intent is unclear. The most likely options are: {options}.
    Generate a single, friendly, short clarification question.
    Do NOT proceed with any task. Only ask the question.
    """
    question = llm.invoke(prompt).content
    state["follow_up_question"] = question
    state["is_intent_clear"] = False
    return state
```

### 5.4 Planner Agent

```
Input: intent + extracted_text + user_query
Output: Ordered list of steps with tool names

Example Plan for "PDF meeting notes → action items":
  Step 1: [pdf_parser] Extract text from PDF
  Step 2: [rag_indexer] Index document into FAISS
  Step 3: [qa_rag] Query: "What are the action items?" → Top-5 chunks
  Step 4: [formatter] Format as bullet list

Estimated cost: ~2,400 tokens × $0.00015/1K = $0.00036
```

---

## 6. RAG Pipeline Design

### 6.1 Indexing Phase (on file upload)

```
PDF Text (full)
    │
    ▼
[RecursiveCharacterTextSplitter]
    chunk_size=512, chunk_overlap=50
    │
    ▼
[Embedding Model]
    Google: "models/text-embedding-004"
    OR local: "sentence-transformers/all-MiniLM-L6-v2"
    │
    ▼
[FAISS Index]
    Stored in memory (session-scoped) or on disk
    Key: session_id → Index mapping
```

### 6.2 Query Phase (on user question)

```
User Query
    │
    ▼
[Embed Query] → same embedding model
    │
    ▼
[FAISS.similarity_search(query_embedding, k=5)]
    │
    ▼
[Top-5 Chunks] → injected into prompt:
    "Use ONLY the following context to answer the question:
     [chunk 1] [chunk 2] ... [chunk 5]
     Question: {user_query}
     Answer concisely and only from the context."
    │
    ▼
[Gemini Response] → text-only answer
```

### 6.3 RAG Optimization Strategies

| Strategy | Implementation |
|---|---|
| Chunk overlap | 50-token overlap prevents context loss at boundaries |
| Hybrid search | BM25 lexical + FAISS semantic → re-rank by score |
| Self-querying | LLM generates better search query from natural language |
| Max marginal relevance | `search_type="mmr"` for diverse chunk selection |

---

## 7. Detailed Roadmap (Phase-by-Phase)

### Phase 1: Foundation Setup (Day 1)

**Tasks**:
- [ ] Initialize project with `uv` / `pip` virtual environment
- [ ] Create FastAPI skeleton (`main.py`, health endpoint)
- [ ] Configure `.env` and `config.py` (Pydantic Settings)
- [ ] Set up logging (`structlog` or `logging` with JSON format)
- [ ] Install all dependencies (`requirements.txt`)
- [ ] Docker setup (`Dockerfile`, `docker-compose.yml`)

**Dependencies to install**:
```
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
google-generativeai==0.8.0
langchain==0.3.0
langchain-google-genai==2.0.0
langgraph==0.2.0
pytesseract==0.3.13
Pillow==10.4.0
PyMuPDF==1.24.0
openai-whisper==20231117
youtube-transcript-api==0.6.2
faiss-cpu==1.8.0
langchain-community==0.3.0
sentence-transformers==3.1.0
pydantic[email]==2.9.0
python-dotenv==1.0.1
httpx==0.27.0
structlog==24.4.0
pytest==8.3.0
pytest-asyncio==0.23.0
httpx==0.27.0
```

---

### Phase 2: Extraction Layer (Day 2)

**Tasks**:
- [ ] Implement `image_ocr.py`:
  - Accept `UploadFile` → save to temp → PIL open → Tesseract
  - Return `{text, confidence, word_count}`
  - Add image preprocessing (grayscale, threshold)
- [ ] Implement `pdf_parser.py`:
  - Try PyMuPDF direct text extraction first
  - If `len(text.strip()) < 50` → render pages as images → OCR fallback
  - Return `{text, page_count, method: "direct" | "ocr_fallback", confidence}`
- [ ] Implement `audio_transcriber.py`:
  - Load Whisper model once at startup (singleton pattern)
  - Accept audio file → convert to WAV if needed (`pydub`) → transcribe
  - Return `{text, language, duration_seconds, segments}`
- [ ] Implement `youtube_fetcher.py`:
  - Regex URL detection
  - Fetch transcript → join segments → return `{text, video_id, available_languages}`
  - Fallback: `{"error": "Transcript not available for this video"}`
- [ ] Unit tests for each extractor

---

### Phase 3: LangGraph Agent Core (Day 3)

**Tasks**:
- [ ] Define `AgentState` TypedDict schema
- [ ] Implement `intent_agent.py`:
  - Gemini call with structured JSON output
  - Parse response → populate `intent`, `is_intent_clear`
- [ ] Implement `followup_agent.py`:
  - Generate targeted clarification question
  - Store in `follow_up_question`
- [ ] Implement `planner_agent.py`:
  - Generate ordered execution plan
  - Token cost estimation (Bonus)
- [ ] Wire `graph.py`:
  - Define all nodes
  - Add conditional edges from intent node
  - Compile graph: `graph.compile(checkpointer=MemorySaver())`
- [ ] Test graph execution with mock inputs

---

### Phase 4: Executor Tools (Day 4)

**Tasks**:
- [ ] `summarizer.py`:
  ```
  Output format (strict):
  ONE-LINE SUMMARY: [...]
  
  KEY POINTS:
  • [Point 1]
  • [Point 2]
  • [Point 3]
  
  DETAILED SUMMARY:
  [5 sentences...]
  ```
- [ ] `sentiment_analyzer.py`:
  ```
  SENTIMENT: Positive/Negative/Neutral/Mixed
  CONFIDENCE: X%
  JUSTIFICATION: [one line]
  ```
- [ ] `code_explainer.py`:
  - Language detection via Gemini
  - Explanation + bug warnings + Big-O complexity
- [ ] `qa_rag.py`:
  - Full RAG pipeline (index on upload, query on demand)
  - Cite source chunks in output
- [ ] `conversational.py`:
  - General chat using conversation history
- [ ] Register all tools in LangGraph `ToolNode`

---

### Phase 5: FastAPI Routes & Session Management (Day 5)

**Tasks**:
- [ ] `POST /upload`:
  - Accept multipart file
  - Detect MIME type → route to correct extractor
  - Store extracted text in session
  - Return `ExtractionResult`
- [ ] `POST /chat`:
  - Accept `{session_id, message, file_id?}`
  - Run LangGraph with session checkpointer
  - Return `AgentResponse`
- [ ] `GET /status/{task_id}` (for async audio processing)
- [ ] `GET /health` — liveness probe
- [ ] Session management: in-memory dict `{session_id: AgentState}` (MemorySaver)
- [ ] CORS configuration for frontend

---

### Phase 6: Frontend (Day 6)

**Tasks**:
- [ ] Chat UI layout (Jinja2 template):
  - Left: file upload + extraction preview panel
  - Right: chat message window (user/assistant bubbles)
  - Bottom: text input + send button + file attach icon
- [ ] `app.js`:
  - `POST /upload` on file select → show extraction preview
  - `POST /chat` on message send → append response
  - Handle `need_clarification` status → show follow-up question
  - Loading spinner during processing
  - Execution plan display (collapsible)
- [ ] CSS: Clean minimal dark/light mode

---

### Phase 7: Testing & Documentation (Day 7)

**Tasks**:
- [ ] `test_extraction.py` — unit tests with sample files
- [ ] `test_agents.py` — mock LLM, test intent classification
- [ ] `test_tools.py` — test each tool with known inputs
- [ ] `test_api.py` — `httpx.AsyncClient` end-to-end API tests
- [ ] Run all 3 sample test cases from requirements
- [ ] Generate architecture diagram
- [ ] Write `README.md`

---

## 8. API Contract Design

### POST /upload
```http
POST /upload
Content-Type: multipart/form-data

Body: file=<binary>

Response 200:
{
  "file_id": "uuid",
  "session_id": "uuid",
  "file_type": "pdf",
  "extracted_text": "Meeting notes from...",
  "confidence": 0.94,
  "page_count": 3,
  "duration_seconds": null,
  "word_count": 847
}
```

### POST /chat
```http
POST /chat
Content-Type: application/json

Body:
{
  "session_id": "uuid",
  "message": "What are the action items?",
  "file_id": "uuid"   // optional, if referring to uploaded file
}

Response 200:
{
  "session_id": "uuid",
  "status": "complete",
  "extracted_text": "Meeting notes: ...",
  "intent": "qa_rag",
  "follow_up_question": null,
  "execution_plan": [
    "Step 1: Retrieve PDF text from session",
    "Step 2: Query FAISS index for 'action items'",
    "Step 3: Generate answer from top-5 relevant chunks"
  ],
  "result": "Action items identified:\n• John to send report by Friday\n• ...",
  "logs": ["[intent_node] Classified as qa_rag (0.92)", "..."],
  "estimated_cost": 0.00042,
  "duration_seconds": null
}

Response 200 (clarification needed):
{
  "session_id": "uuid",
  "status": "need_clarification",
  "follow_up_question": "Could you clarify whether you want a summary or sentiment analysis of this text?",
  "result": null
}
```

---

## 9. UI/UX Design Specification

### Layout
```
┌─────────────────────────────────────────────────────┐
│  🤖 Agentic AI Assistant           [Dark/Light Toggle] │
├──────────────────────┬──────────────────────────────┤
│  📄 UPLOAD & PREVIEW │        CHAT WINDOW           │
│                      │                              │
│  [Drag & Drop Zone]  │  ┌──────────────────────┐   │
│  PNG / JPG / PDF /   │  │ 👤 User: [message]   │   │
│  MP3 / WAV / M4A     │  └──────────────────────┘   │
│                      │  ┌──────────────────────┐   │
│  ─── EXTRACTED ───   │  │ 🤖 Agent: [response] │   │
│  [Preview text box]  │  └──────────────────────┘   │
│  Confidence: 94%     │                              │
│  Pages: 3            │  ┌──────────────────────┐   │
│                      │  │ 🤖 ▶ Plan: [expand]  │   │
│  ─── PLAN ─────────  │  └──────────────────────┘   │
│  Step 1: extract     │                              │
│  Step 2: rag query   │  ┌────────────┬────────┐    │
│  Step 3: format      │  │  Type here │  Send  │    │
│                      │  └────────────┴────────┘    │
│                      │           [📎 Attach]       │
└──────────────────────┴──────────────────────────────┘
```

### Key UX Rules
1. **File upload → instant extraction preview** — user sees extracted text before asking
2. **Follow-up questions appear as agent chat bubbles** — seamless conversation
3. **Execution plan is collapsible** — shows agent's reasoning on demand
4. **Loading states**: spinner + "Extracting text..." / "Analyzing intent..." / "Generating response..."
5. **Error states**: friendly message + retry button

---

## 10. Error Handling & Robustness

| Scenario | Handling |
|---|---|
| Image with no readable text | Return `confidence: 0.1`, message: "No readable text detected. Try a higher-resolution image." |
| PDF is scanned but OCR fails | Retry with image dpi=300, then return partial results with warning |
| Audio file > 5 minutes | Process in chunks via Whisper segment API; aggregate results |
| YouTube transcript unavailable | Return: "Transcript not available. Auto-captions may be disabled." |
| LLM API timeout | Retry 3× with exponential backoff (1s, 2s, 4s) |
| Intent still unclear after follow-up | Ask one more time, then offer menu of options |
| Unsupported file format | Validate MIME type at upload; return clear error before processing |
| Empty uploaded file | Validate file size > 0 bytes before extraction |
| Session not found | Return 404 with `session_id` recreation suggestion |

---

## 11. Test Cases

### Test Case 1: Audio Lecture → Transcription + Summary
```
Input: lecture_5min.mp3
Expected:
  - extracted_text: full transcript (~750 words)
  - duration_seconds: 300
  - intent: transcribe (auto-detected)
  - result:
      ONE-LINE SUMMARY: The lecture covers machine learning fundamentals...
      KEY POINTS:
      • Supervised vs unsupervised learning defined
      • Neural networks explained with backpropagation
      • Real-world applications in healthcare and finance
      DETAILED SUMMARY: [5 sentences]
```

### Test Case 2: PDF Meeting Notes → Action Items (RAG)
```
Input: meeting_notes.pdf (3 pages)
Query: "What are the action items?"
Expected:
  - file_type: pdf, page_count: 3
  - intent: qa_rag
  - execution_plan includes "rag_indexer" and "qa_rag" steps
  - result: Bulleted list of action items extracted from document
  - logs show which chunks were retrieved
```

### Test Case 3: Screenshot with Code + "Explain"
```
Input: code_screenshot.png
Query: "Explain"
Expected:
  - extracted_text: OCR'd code (Python detected)
  - intent: code_explain
  - result:
      LANGUAGE: Python
      EXPLANATION: This function sorts a list using...
      POTENTIAL BUGS: Line 5 — index out of range if list is empty
      TIME COMPLEXITY: O(n log n)
```

### Test Case 4: Ambiguous Input (No Query)
```
Input: some_document.pdf (no query)
Expected:
  - status: need_clarification
  - follow_up_question: "I've extracted the text from your PDF. What would you like me to do with it — summarize, find specific information, or something else?"
```

### Test Case 5: YouTube URL
```
Input: "Summarize this: https://www.youtube.com/watch?v=abc123"
Expected:
  - intent: youtube_transcript (URL detected)
  - execution_plan: ["youtube_fetcher", "summarizer"]
  - result: 1-liner + bullets + 5-sentence summary of video
```

---

## 12. Bonus: Multi-Agent & Cost Estimator

### Multi-Agent Architecture (Separate Services)

```
Service 1: Planner (port 8001)
  - Receives: raw input + extracted text
  - Outputs: intent + execution plan + cost estimate
  - Exposes: POST /plan

Service 2: Executor (port 8002)
  - Receives: execution plan + tool inputs
  - Outputs: tool results + formatted response
  - Exposes: POST /execute

Gateway (port 8000):
  - Orchestrates Planner → Executor
  - Handles sessions, auth, rate limiting
```

**Communication**: REST HTTP calls between services (or gRPC for performance)

### Cost Estimator

```python
def estimate_cost(text: str, intent: str) -> dict:
    """Estimates Gemini API cost before execution."""
    input_tokens = len(text.split()) * 1.3  # rough estimate
    
    # Add prompt template tokens per task
    prompt_overhead = {
        "summarize": 200,
        "sentiment": 150,
        "code_explain": 250,
        "qa_rag": 800,  # includes retrieved chunks
    }
    
    total_input = input_tokens + prompt_overhead.get(intent, 200)
    output_tokens = 500  # estimated output
    
    # Gemini 1.5 Flash pricing (as of 2024)
    cost = (total_input * 0.000075 + output_tokens * 0.0003) / 1000
    
    return {
        "estimated_input_tokens": int(total_input),
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 6),
        "model": "gemini-1.5-flash"
    }
```

---

## 13. Deliverables Checklist

| Deliverable | Status |
|---|---|
| ✅ Clean codebase (modular, typed) | Build Phase 1-5 |
| ✅ Architecture diagram | Phase 7 |
| ✅ FastAPI backend | Phase 1, 5 |
| ✅ Simple chat UI | Phase 6 |
| ✅ Test cases (unit + integration) | Phase 7 |
| ✅ README.md | Phase 7 |
| ⭐ Multi-agent orchestration (bonus) | Phase 7 |
| ⭐ Cost estimator (bonus) | Phase 4 |

---

## Evaluation Alignment

| Rubric Item | Points | Implementation |
|---|---|---|
| Correctness | 30 | All 6 task types + 3 sample test cases |
| Autonomy & Planning | 20 | LangGraph planner node + conditional routing |
| Robustness | 15 | Error handling table + retries + fallbacks |
| Explainability | 10 | `logs` field in every response + plan display |
| Code Quality | 10 | Modular structure + Pydantic + pytest |
| UX & Demo | 10 | Chat UI + extraction preview + plan collapse |
| **Total** | **95** | **+ 5 bonus for multi-agent + cost estimator** |
