"""
audio_transcriber.py — Speech-to-text using faster-whisper.

Uses the faster-whisper library (CTranslate2 backend) which:
- Runs locally (no API cost, no privacy concerns)
- Is 4× faster than original Whisper with same accuracy
- Supports automatic language detection
- Returns word-level timestamps via segments

Model is loaded once at startup via the singleton in dependencies.py.
"""
from pathlib import Path
from typing import Optional
import tempfile
import shutil

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Supported audio MIME types
SUPPORTED_AUDIO_MIMES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/mp4", "audio/m4a", "audio/ogg", "audio/flac",
    "audio/webm", "audio/aac",
}


def _convert_to_wav_if_needed(file_path: Path) -> tuple[Path, bool]:
    """
    Convert audio to WAV if pydub is needed.
    Returns (path_to_use, was_converted).
    WAV/MP3 are passed directly to Whisper; others may need conversion.
    """
    suffix = file_path.suffix.lower()
    # faster-whisper / ffmpeg handles most formats natively
    # We only convert if the format is unusual
    if suffix in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"):
        return file_path, False

    # For unusual formats, convert to WAV via pydub
    try:
        from pydub import AudioSegment
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        audio = AudioSegment.from_file(str(file_path))
        audio.export(tmp.name, format="wav")
        logger.info("audio_converted", original=suffix, converted="wav")
        return Path(tmp.name), True
    except Exception as e:
        logger.warning("audio_conversion_failed", error=str(e))
        return file_path, False


def transcribe_audio(file_path: Path, model=None) -> dict:
    """
    Transcribe an audio file using faster-whisper.

    Args:
        file_path: Path to the audio file on disk.
        model: faster-whisper WhisperModel instance (loaded by dependency).
               If None, loads the 'base' model inline (for testing).

    Returns:
        {
          "text": str,                  # Full transcript
          "language": str,              # Detected language code
          "language_probability": float,
          "duration_seconds": float,
          "word_count": int,
          "segments": list[dict],       # [{start, end, text}]
          "method": "faster_whisper",
          "warning": Optional[str]
        }

    Raises:
        FileNotFoundError: If file_path does not exist.
        RuntimeError: If transcription fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    logger.info("transcription_start", path=str(file_path))

    # Convert if necessary
    work_path, was_converted = _convert_to_wav_if_needed(file_path)
    converted_temp = work_path if was_converted else None

    try:
        # Lazy-load model if not provided
        if model is None:
            from faster_whisper import WhisperModel
            logger.info("whisper_model_lazy_load", model="base")
            model = WhisperModel("base", compute_type="int8")

        segments_gen, info = model.transcribe(
            str(work_path),
            beam_size=5,
            word_timestamps=False,
        )

        # Collect segments
        segments = []
        full_text_parts = []
        last_end = 0.0

        for seg in segments_gen:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())
            last_end = seg.end

        full_text = " ".join(full_text_parts)
        word_count = len(full_text.split())
        duration = round(info.duration, 2) if hasattr(info, "duration") else round(last_end, 2)

        warning: Optional[str] = None
        if not full_text.strip():
            warning = "No speech detected in the audio file."
        elif duration > 300:
            warning = f"Long audio ({duration:.0f}s). Processing may have taken extra time."

        result = {
            "text": full_text,
            "language": info.language if hasattr(info, "language") else "unknown",
            "language_probability": round(
                info.language_probability if hasattr(info, "language_probability") else 1.0, 3
            ),
            "duration_seconds": duration,
            "word_count": word_count,
            "segments": segments,
            "method": "faster_whisper",
            "warning": warning,
        }

        logger.info(
            "transcription_complete",
            language=result["language"],
            duration=duration,
            word_count=word_count,
        )
        return result

    except Exception as e:
        logger.error("transcription_failed", error=str(e))
        raise RuntimeError(f"Transcription failed: {e}") from e

    finally:
        # Clean up converted temp file
        if converted_temp and converted_temp.exists():
            converted_temp.unlink(missing_ok=True)
