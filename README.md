# NeuraMind — Multi-Modal Agentic AI

<div align="center">

![NeuraMind Banner](https://img.shields.io/badge/NeuraMind-Multi--Modal%20AI-6366f1?style=for-the-badge&logo=openai&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-FF6B35?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini%202.5-Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade agentic system that processes Text, Images, PDFs and Audio — autonomously classifying intent and executing the right task.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Usage](#-usage) • [API Docs](#-api-reference) • [Tests](#-running-tests)

</div>

---

## ✨ Features

| Capability | Description |
|---|---|
| 📄 **PDF Processing** | Direct text extraction (PyMuPDF) + OCR fallback (Tesseract) |
| 🖼 **Image OCR** | Tesseract-powered text extraction with preprocessing |
| 🎧 **Audio Transcription** | OpenAI Whisper (runs locally, no API cost) |
| ▶️ **YouTube Summaries** | Transcript fetch + intelligent summarization |
| 🧠 **Intent Classification** | Gemini-powered JSON classifier with 9 intent labels |
| 💬 **RAG Q&A** | FAISS vector store + sentence-transformers embeddings |
| 📝 **Summarization** | 3-tier: one-liner → key points → detailed |
| 🎭 **Sentiment Analysis** | Label + confidence + emotion detection + evidence |
| 💻 **Code Explanation** | Language detection, bug spotting, Big-O complexity |
| 🔄 **Multi-turn Chat** | Session-persistent conversation history |
| ❓ **Clarification Flow** | Auto follow-up when intent is ambiguous |

---

## 🏗 Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────┐
│               FastAPI Backend                    │
│  POST /api/upload ──► Extraction Layer           │
│  POST /api/chat   ──► LangGraph Agent Pipeline   │
└─────────────────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   LangGraph StateGraph  │
            │                        │
            │  Intent Classifier      │
            │       │                │
            │  ┌────┴─────┐          │
            │  │          │          │
            │ Planner  Follow-up     │
            │  │          │          │
            │  └────┬─────┘          │
            │       │                │
            │   Executor Node        │
            │  ┌────┴──────────┐     │
            │  │ summarize     │     │
            │  │ sentiment     │     │
            │  │ code_explain  │     │
            │  │ qa_rag (FAISS)│     │
            │  │ converse      │     │
            │  └───────────────┘     │
            │       │                │
            │   Formatter            │
            └────────────────────────┘
                    │
                    ▼
            Structured Response
     (intent + plan + result + logs + cost)
```

---

## ⚡ Quick Start

### Prerequisites

| Requirement | Install |
|---|---|
| Python 3.10+ | [python.org](https://python.org) |
| Gemini API Key | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) *(free)* |
| Tesseract OCR | `winget install UB-Mannheim.TesseractOCR` *(for image/scanned PDF)* |

### 1. Clone the repo

```bash
git clone https://github.com/Ratneshp1122/DSAI.git
cd DSAI/agentic-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Copy the example and fill in your key
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=AIzaSy...your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 4. Start the server

```bash
py -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

### 5. Open the app

```
http://127.0.0.1:8001
```

---

## 🖥 Usage

### Web UI

The app ships with a dark glassmorphism chat interface:

1. **Upload a file** — drag and drop or click browse in the left panel  
   Supports: `PDF`, `JPG/PNG`, `MP3/WAV`, `TXT/MD`

2. **Ask anything** — the AI automatically classifies your intent:
   - *"Summarize this"* → structured summary
   - *"What is the revenue growth?"* → RAG Q&A with evidence
   - *"What's the sentiment?"* → sentiment analysis
   - *"Explain the code"* → code explanation with bug detection
   - Paste a YouTube URL → transcript + summary

3. **View execution plan** — click ⚙ Execution Plan to see step-by-step reasoning

### Example Queries

```
"Give me a one-page summary of this PDF"
"What does the author conclude about climate change?"
"Is this review positive or negative?"
"Explain what this Python function does and find any bugs"
"https://youtube.com/watch?v=xxx — summarize this video"
```

---

## 📡 API Reference

Base URL: `http://127.0.0.1:8001`  
Interactive docs: `http://127.0.0.1:8001/api/docs`

### `POST /api/upload`

Upload a file for processing.

```bash
curl -X POST http://127.0.0.1:8001/api/upload \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "session_id": "uuid",
  "file_id": "uuid",
  "file_type": "pdf",
  "extracted_text": "Preview of extracted text...",
  "confidence": 1.0,
  "word_count": 1204,
  "method": "direct"
}
```

### `POST /api/chat`

Send a message and run the agent pipeline.

```bash
curl -X POST http://127.0.0.1:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id-from-upload",
    "message": "Summarize this document"
  }'
```

**Response:**
```json
{
  "session_id": "uuid",
  "status": "complete",
  "intent": "summarize",
  "intent_confidence": 0.95,
  "execution_plan": ["Load extracted text", "Run summarizer tool", "Format output"],
  "result": "ONE-LINE SUMMARY: ...\n\nKEY POINTS:\n• ...",
  "logs": ["[intent_node] intent=summarize confidence=0.95 (0.8s)"],
  "estimated_cost": 0.000234
}
```

**Status values:**
| Status | Meaning |
|---|---|
| `complete` | Task finished, `result` contains the output |
| `need_clarification` | Intent was ambiguous, `follow_up_question` is set |
| `error` | Pipeline failed |

### `GET /api/chat/session/{session_id}`

Get session metadata.

### `DELETE /api/chat/session/{session_id}`

Clear session and free memory.

### `GET /health`

```json
{ "status": "ok", "version": "1.0.0", "uptime_seconds": 120.5 }
```

---

## 🧪 Running Tests

```bash
cd DSAI/agentic-app

# Run all 48 tests
py -m pytest backend/tests/ -v

# Run specific test file
py -m pytest backend/tests/test_api.py -v
py -m pytest backend/tests/test_extraction.py -v
```

**Test coverage:**

| Test Class | What it tests |
|---|---|
| `TestHealthEndpoint` | `/health` liveness check |
| `TestUploadEndpoint` | File upload for all types, MIME rejection |
| `TestChatEndpoint` | Full pipeline: summarize, clarification, session CRUD |
| `TestIntentAgent` | Intent routing logic, JSON parsing, fallback |
| `TestPlannerAgent` | Execution plan generation per intent |
| `TestSummarizer` | Summary tool output format |
| `TestSentimentAnalyzer` | Sentiment label + evidence |
| `TestCodeExplainer` | Code analysis + complexity |
| `TestRAGPipeline` | Chunker, FAISS index, RAG QA end-to-end |
| `TestPDFParserGraceful` | Tesseract-absent graceful degradation |

---

## 🗂 Project Structure

```
DSAI/
├── agentic-app/
│   ├── backend/
│   │   ├── agents/
│   │   │   ├── intent_agent.py      # Gemini intent classifier (9 labels)
│   │   │   ├── followup_agent.py    # Clarification question generator
│   │   │   ├── planner_agent.py     # Execution plan + cost estimation
│   │   │   └── graph.py             # LangGraph StateGraph definition
│   │   ├── extraction/
│   │   │   ├── pdf_parser.py        # PyMuPDF + Tesseract OCR fallback
│   │   │   ├── image_ocr.py         # Tesseract image extraction
│   │   │   ├── audio_transcriber.py # Whisper transcription
│   │   │   └── youtube_fetcher.py   # YouTube transcript API
│   │   ├── tools/
│   │   │   ├── summarizer.py        # 3-tier structured summaries
│   │   │   ├── sentiment_analyzer.py# Sentiment + emotion detection
│   │   │   ├── code_explainer.py    # Code analysis + Big-O
│   │   │   ├── qa_rag.py            # FAISS-backed Q&A
│   │   │   └── conversational.py    # Multi-turn chat
│   │   ├── rag/
│   │   │   ├── chunker.py           # Recursive text splitter
│   │   │   ├── embedder.py          # all-MiniLM-L6-v2 embeddings
│   │   │   └── vector_store.py      # In-memory FAISS per session
│   │   ├── routers/
│   │   │   ├── upload.py            # POST /api/upload
│   │   │   ├── chat.py              # POST /api/chat + session management
│   │   │   └── health.py            # GET /health
│   │   ├── schemas/                 # Pydantic request/response models
│   │   ├── utils/
│   │   │   ├── llm_factory.py       # Centralised LLM builder
│   │   │   ├── logger.py            # Structured JSON logging
│   │   │   └── file_handler.py      # Upload file management
│   │   └── config.py                # Pydantic Settings (reads .env)
│   ├── frontend/
│   │   ├── templates/index.html     # Chat UI
│   │   └── static/
│   │       ├── css/style.css        # Dark glassmorphism design
│   │       └── js/app.js            # Upload, chat, session management
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
└── .gitignore
```

---

## ⚙️ Configuration

All settings are read from `.env` (override via environment variables):

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Get free at aistudio.google.com |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Any model from ListModels |
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `UPLOAD_DIR` | `./uploads` | Where uploaded files are saved |
| `RAG_CHUNK_SIZE` | `512` | Token chunk size for RAG |
| `RAG_TOP_K` | `5` | Number of chunks retrieved per query |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` |

---

## 🐳 Docker (Optional)

```bash
cd DSAI/agentic-app

# Set your API key
echo "GEMINI_API_KEY=your_key" > .env

# Build and run
docker-compose up --build
```

App available at `http://localhost:8000`

---

## 📋 Requirements

Key dependencies (see `requirements.txt` for full list):

```
fastapi / uvicorn          — API server
langgraph                  — Agent orchestration
langchain-google-genai     — Gemini integration
PyMuPDF                    — PDF text extraction
pytesseract / Pillow       — Image OCR
openai-whisper             — Audio transcription
faiss-cpu                  — Vector similarity search
sentence-transformers      — Local text embeddings
youtube-transcript-api     — YouTube captions
pydantic-settings          — Config management
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "feat: your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using FastAPI · LangGraph · Gemini · FAISS
</div>
