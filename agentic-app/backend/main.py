"""
main.py — FastAPI application entry point.

Wires together:
  - Logging setup
  - CORS middleware
  - All APIRouters (health, upload, chat)
  - Static files and Jinja2 templates
  - Startup / shutdown lifecycle events
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.config import get_settings
from backend.utils.logger import setup_logging, get_logger
from backend.routers.health import router as health_router

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise logging, ensure upload dir exists.
    Shutdown: nothing to clean up (Whisper model is GC'd).
    """
    setup_logging()
    logger = get_logger("main")
    settings = get_settings()

    # Ensure upload directory exists
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    logger.info("app_startup", upload_dir=settings.upload_dir)

    yield  # ← application runs here

    logger.info("app_shutdown")


# ── App factory ───────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="Agentic Multi-Modal AI",
    description=(
        "A production-grade agentic system that accepts Text, Images, PDFs, "
        "and Audio, extracts content, classifies intent, and performs the "
        "correct task — all with text-only output."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health_router)

from backend.routers.upload import router as upload_router
from backend.routers.chat import router as chat_router

app.include_router(upload_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

# ── Static files & Templates ──────────────────────────────────────────────────

_base = Path(__file__).parent.parent  # project root

static_dir = _base / "frontend" / "static"
templates_dir = _base / "frontend" / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(templates_dir))


# ── Root route ────────────────────────────────────────────────────────────────

from fastapi import Request
from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    """Serve the chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})
