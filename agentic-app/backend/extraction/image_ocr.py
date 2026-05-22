"""
image_ocr.py — Tesseract OCR extraction from image files.

Pipeline:
  1. Load image with PIL
  2. Preprocess: grayscale + adaptive threshold for better OCR accuracy
  3. Run pytesseract to extract text + per-word confidence data
  4. Return structured result with text, confidence, word_count
"""
from pathlib import Path
from typing import Optional

import pytesseract
from pytesseract import Output
from PIL import Image, ImageFilter, ImageOps

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Auto-detect Tesseract path on Windows ────────────────────────────────────
import sys, os
if sys.platform == "win32":
    _tess_candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for _path in _tess_candidates:
        if os.path.isfile(_path):
            pytesseract.pytesseract.tesseract_cmd = _path
            break

# Supported image MIME types
SUPPORTED_IMAGE_MIMES = {
    "image/jpeg", "image/png", "image/bmp",
    "image/tiff", "image/webp", "image/gif",
}


def _preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR accuracy:
    - Convert to grayscale
    - Apply slight sharpening
    - Scale up if small (Tesseract performs better on larger images)
    """
    # Convert to grayscale
    img = img.convert("L")

    # Scale up small images (< 1000px wide)
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)

    return img


def extract_text_from_image(file_path: Path) -> dict:
    """
    Extract text from an image file using Tesseract OCR.

    Args:
        file_path: Path to the image file on disk.

    Returns:
        {
          "text": str,              # Extracted text
          "confidence": float,      # Average OCR confidence (0.0–1.0)
          "word_count": int,
          "method": "tesseract_ocr",
          "warning": Optional[str]  # Low confidence warning
        }

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If image cannot be opened.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    logger.info("ocr_start", path=str(file_path))

    try:
        img = Image.open(file_path)
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}") from e

    # Preprocess
    processed = _preprocess_image(img)

    # Extract text + confidence data
    try:
        data = pytesseract.image_to_data(processed, output_type=Output.DICT)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract is not installed or not found in PATH. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki"
        )

    # Calculate mean confidence from words with confidence > 0
    confidences = [
        int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) > 0
    ]
    avg_confidence = round(sum(confidences) / len(confidences) / 100, 3) if confidences else 0.0

    # Reconstruct text (join non-empty words)
    words = [w for w in data["text"] if w.strip()]
    text = " ".join(words)

    word_count = len(words)

    warning: Optional[str] = None
    if avg_confidence < 0.4:
        warning = "Low OCR confidence. Try a higher-resolution image for better results."
    elif not text.strip():
        warning = "No readable text detected in the image."
        avg_confidence = 0.0

    result = {
        "text": text,
        "confidence": avg_confidence,
        "word_count": word_count,
        "method": "tesseract_ocr",
        "warning": warning,
    }

    logger.info(
        "ocr_complete",
        word_count=word_count,
        confidence=avg_confidence,
        has_warning=warning is not None,
    )
    return result
