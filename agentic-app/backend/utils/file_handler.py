"""
file_handler.py — Temp file management utilities.
Handles saving/cleaning uploaded files in the upload directory.
"""
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def get_upload_dir() -> Path:
    """Return and ensure upload directory exists."""
    settings = get_settings()
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


async def save_upload_file(upload_file: UploadFile, file_id: str = None) -> Path:
    """
    Save an UploadFile to a temp location on disk.
    Returns the Path to the saved file.
    """
    upload_dir = get_upload_dir()
    suffix = Path(upload_file.filename).suffix if upload_file.filename else ""
    unique_name = f"{file_id or uuid.uuid4()}{suffix}"
    dest_path = upload_dir / unique_name

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    logger.info("file_saved", path=str(dest_path), original_name=upload_file.filename)
    return dest_path


def delete_file(file_path: Path) -> None:
    """Delete a file safely, ignoring missing-file errors."""
    try:
        file_path.unlink(missing_ok=True)
        logger.info("file_deleted", path=str(file_path))
    except Exception as e:
        logger.warning("file_delete_failed", path=str(file_path), error=str(e))
