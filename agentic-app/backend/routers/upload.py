"""
upload.py — POST /upload endpoint.
Accepts a file, detects type, runs extraction, stores result in session store.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from backend.schemas.request import UploadResponse
from backend.utils.file_handler import save_upload_file
from backend.dependencies import get_session_store
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# MIME → file_type mapping
MIME_TO_TYPE = {
    # Images
    "image/jpeg": "image", "image/png": "image", "image/bmp": "image",
    "image/tiff": "image", "image/webp": "image", "image/gif": "image",
    # PDFs
    "application/pdf": "pdf",
    # Audio
    "audio/mpeg": "audio", "audio/mp3": "audio", "audio/wav": "audio",
    "audio/x-wav": "audio", "audio/mp4": "audio", "audio/m4a": "audio",
    "audio/ogg": "audio", "audio/flac": "audio", "audio/webm": "audio",
    "audio/aac": "audio",
    # Text
    "text/plain": "text", "text/csv": "text", "text/markdown": "text",
}


@router.post("/upload", response_model=UploadResponse, tags=["upload"])
async def upload_file(
    file: UploadFile = File(...),
    session_store: dict = Depends(get_session_store),
) -> UploadResponse:
    """
    Upload a file (image, PDF, audio, or text) for processing.
    Runs extraction immediately and stores result in the session store.
    Returns file_id and session_id for subsequent /chat calls.
    """
    content_type = file.content_type or ""
    file_type = MIME_TO_TYPE.get(content_type)

    # Explicitly reject browser/web types
    REJECTED_TYPES = {"text/html", "text/xml", "application/xml", "application/json",
                      "application/javascript", "text/javascript", "application/zip",
                      "application/x-zip-compressed"}
    if content_type in REJECTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Supported: image, PDF, audio, text."
        )

    # Extension-based fallback
    if not file_type and file.filename:
        ext = Path(file.filename).suffix.lower()
        ext_map = {
            ".jpg": "image", ".jpeg": "image", ".png": "image", ".bmp": "image",
            ".tiff": "image", ".webp": "image", ".gif": "image",
            ".pdf": "pdf",
            ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
            ".ogg": "audio", ".flac": "audio", ".aac": "audio", ".webm": "audio",
            ".txt": "text", ".md": "text", ".csv": "text",
        }
        file_type = ext_map.get(ext, "text")

    if not file_type:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Supported: image, PDF, audio, text."
        )

    # Generate IDs
    file_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    logger.info("upload_received", file_id=file_id, filename=file.filename, file_type=file_type)

    # Save file to disk
    file_path = await save_upload_file(file, file_id)

    # Run extraction based on type
    extraction_result = {}
    try:
        if file_type == "image":
            from backend.extraction.image_ocr import extract_text_from_image
            extraction_result = extract_text_from_image(file_path)

        elif file_type == "pdf":
            from backend.extraction.pdf_parser import extract_text_from_pdf
            extraction_result = extract_text_from_pdf(file_path)

        elif file_type == "audio":
            from backend.extraction.audio_transcriber import transcribe_audio
            from backend.dependencies import get_whisper_model
            model = get_whisper_model()
            extraction_result = transcribe_audio(file_path, model=model)

        elif file_type == "text":
            raw = file_path.read_text(errors="replace")
            extraction_result = {
                "text": raw,
                "confidence": 1.0,
                "word_count": len(raw.split()),
                "method": "direct_read",
            }

    except Exception as e:
        logger.error("extraction_failed", file_id=file_id, error=str(e))
        extraction_result = {
            "text": "",
            "confidence": 0.0,
            "word_count": 0,
            "method": "failed",
            "warning": f"Extraction error: {str(e)}",
        }

    # Store session data
    session_store[session_id] = {
        "file_id": file_id,
        "file_type": file_type,
        "file_path": str(file_path),
        "filename": file.filename,
        "extracted_text": extraction_result.get("text", ""),
        "extraction_confidence": extraction_result.get("confidence", 1.0),
        "file_metadata": {
            "page_count": extraction_result.get("page_count"),
            "duration_seconds": extraction_result.get("duration_seconds"),
            "word_count": extraction_result.get("word_count", 0),
            "method": extraction_result.get("method"),
            "language": extraction_result.get("language"),
            "segments": extraction_result.get("segments", []),
        },
        "messages": [],
    }

    logger.info(
        "upload_complete",
        session_id=session_id,
        file_type=file_type,
        word_count=extraction_result.get("word_count", 0),
    )

    return UploadResponse(
        file_id=file_id,
        session_id=session_id,
        file_type=file_type,
        extracted_text=extraction_result.get("text", "")[:500] or None,  # preview only
        confidence=extraction_result.get("confidence"),
        page_count=extraction_result.get("page_count"),
        duration_seconds=extraction_result.get("duration_seconds"),
        word_count=extraction_result.get("word_count"),
        method=extraction_result.get("method"),
        warning=extraction_result.get("warning"),
    )
