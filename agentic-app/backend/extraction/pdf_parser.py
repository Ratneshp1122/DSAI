"""
pdf_parser.py — PDF text extraction using PyMuPDF.

Strategy:
  1. Try direct text extraction with PyMuPDF (fast, layout-preserving)
  2. If extracted text < 50 chars (scanned/image-only PDF):
     a. If Tesseract is installed → render pages at 300-DPI + OCR
     b. If Tesseract is NOT installed → return what was found + clear warning
  3. Return structured result with text, metadata, method used
"""
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
import io

from backend.extraction.image_ocr import _preprocess_image
from backend.utils.logger import get_logger

logger = get_logger(__name__)

DIRECT_TEXT_THRESHOLD = 10   # Chars needed to consider PDF text-based (was 50)
OCR_RENDER_DPI = 300


def _tesseract_available() -> bool:
    """Check if Tesseract binary is installed and reachable."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False
    except Exception:
        return False


def extract_text_from_pdf(file_path: Path) -> dict:
    """
    Extract text from a PDF file with graceful Tesseract fallback.

    Returns:
        {
          "text": str,
          "page_count": int,
          "method": "direct" | "ocr_fallback" | "direct_short",
          "confidence": float,
          "word_count": int,
          "metadata": dict,
          "warning": Optional[str]
        }
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info("pdf_parse_start", path=str(file_path))

    try:
        doc = fitz.open(str(file_path))
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {e}") from e

    page_count = len(doc)
    metadata = {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "page_count": page_count,
    }

    # ── Attempt 1: Direct text extraction ────────────────────────────────────
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    direct_text = "\n\n".join(pages_text).strip()

    if len(direct_text) >= DIRECT_TEXT_THRESHOLD:
        word_count = len(direct_text.split())
        logger.info("pdf_direct_extraction", word_count=word_count, pages=page_count)
        doc.close()
        return {
            "text": direct_text,
            "page_count": page_count,
            "method": "direct",
            "confidence": 1.0,
            "word_count": word_count,
            "metadata": metadata,
            "warning": None,
        }

    # ── Check Tesseract availability ─────────────────────────────────────────
    if not _tesseract_available():
        doc.close()
        warning = (
            "This PDF appears to be scanned/image-based. "
            "Tesseract OCR is not installed — only limited text could be extracted. "
            "Install Tesseract for full OCR: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        # Return whatever direct text was found (even if short)
        word_count = len(direct_text.split())
        logger.warning("tesseract_not_available", pdf=str(file_path))
        return {
            "text": direct_text,
            "page_count": page_count,
            "method": "direct_short",
            "confidence": 0.5 if direct_text else 0.0,
            "word_count": word_count,
            "metadata": metadata,
            "warning": warning,
        }

    # ── Attempt 2: OCR fallback (scanned PDF, Tesseract available) ───────────
    logger.info("pdf_ocr_fallback_start", pages=page_count)
    all_text = []
    all_confidences = []
    warning: Optional[str] = None

    for page_num, page in enumerate(doc):
        mat = fitz.Matrix(OCR_RENDER_DPI / 72, OCR_RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        processed = _preprocess_image(img)
        try:
            data = pytesseract.image_to_data(processed, output_type=Output.DICT)
        except pytesseract.TesseractNotFoundError:
            # Tesseract disappeared mid-run — stop OCR gracefully
            logger.warning("tesseract_disappeared_mid_run")
            break
        except Exception as e:
            logger.warning("pdf_page_ocr_failed", page=page_num + 1, error=str(e))
            continue

        confidences = [
            int(c) for c in data["conf"]
            if str(c).lstrip("-").isdigit() and int(c) > 0
        ]
        if confidences:
            all_confidences.extend(confidences)

        words = [w for w in data["text"] if w.strip()]
        all_text.append(" ".join(words))
        logger.info("pdf_page_ocr", page=page_num + 1, words=len(words))

    doc.close()
    full_text = "\n\n".join(all_text).strip()

    avg_confidence = (
        round(sum(all_confidences) / len(all_confidences) / 100, 3)
        if all_confidences else 0.0
    )

    if not full_text:
        warning = "No readable text found in PDF. The document may be image-only with poor quality."
        avg_confidence = 0.0
    elif avg_confidence < 0.4:
        warning = "Low OCR confidence on scanned PDF. Results may contain errors."

    word_count = len(full_text.split())
    logger.info("pdf_ocr_fallback_complete", word_count=word_count, confidence=avg_confidence)

    return {
        "text": full_text,
        "page_count": page_count,
        "method": "ocr_fallback",
        "confidence": avg_confidence,
        "word_count": word_count,
        "metadata": metadata,
        "warning": warning,
    }
