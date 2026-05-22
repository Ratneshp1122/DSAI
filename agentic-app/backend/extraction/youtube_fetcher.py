"""
youtube_fetcher.py — YouTube transcript fetching via youtube-transcript-api v1.x.

Strategy:
  1. Detect YouTube URL in user input via regex
  2. Extract video ID
  3. Fetch auto-generated or manual captions using instance API
  4. Join segments into clean text
  5. Fallback: clear error message if transcript unavailable

youtube-transcript-api v1.x uses an instance-based API:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    transcript = transcript_list.find_transcript(['en'])
    fetched = transcript.fetch()
"""
import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Regex to extract YouTube video ID from various URL formats
YOUTUBE_URL_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)"
    r"([a-zA-Z0-9_-]{11})"
)


def extract_youtube_video_id(text: str) -> Optional[str]:
    """
    Extract a YouTube video ID from a string containing a URL.

    Returns:
        11-character video ID string, or None if no YouTube URL found.
    """
    match = YOUTUBE_URL_PATTERN.search(text)
    return match.group(1) if match else None


def fetch_youtube_transcript(text_or_url: str) -> dict:
    """
    Fetch the transcript of a YouTube video.

    Args:
        text_or_url: User input or URL string containing a YouTube URL.

    Returns:
        {
          "text": str,                      # Full transcript joined
          "video_id": str,
          "language": str,                  # Transcript language code
          "available_languages": list[str],
          "segment_count": int,
          "method": "youtube_transcript_api",
          "error": Optional[str]
        }
    """
    video_id = extract_youtube_video_id(text_or_url)

    if not video_id:
        return {
            "text": "",
            "video_id": None,
            "language": None,
            "available_languages": [],
            "segment_count": 0,
            "method": "youtube_transcript_api",
            "error": "No YouTube URL detected in the input.",
        }

    logger.info("youtube_fetch_start", video_id=video_id)

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        available_langs = [t.language_code for t in transcript_list]

        # Prefer manually created transcripts; fall back to auto-generated
        try:
            transcript = transcript_list.find_manually_created_transcript(available_langs)
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(available_langs)

        fetched = transcript.fetch()
        language = transcript.language_code

        # FetchedTranscript is iterable; each item has .text attribute
        texts = []
        count = 0
        for snippet in fetched:
            # Support both dict-style and object-style (v1.x returns objects)
            text = snippet.get("text", "") if isinstance(snippet, dict) else getattr(snippet, "text", "")
            if text.strip():
                texts.append(text.strip())
            count += 1

        full_text = " ".join(texts)

        result = {
            "text": full_text,
            "video_id": video_id,
            "language": language,
            "available_languages": available_langs,
            "segment_count": count,
            "method": "youtube_transcript_api",
            "error": None,
        }

        logger.info(
            "youtube_fetch_complete",
            video_id=video_id,
            language=language,
            segments=count,
            word_count=len(full_text.split()),
        )
        return result

    except TranscriptsDisabled:
        msg = "Transcripts are disabled for this video. Auto-captions may be turned off by the creator."
    except VideoUnavailable:
        msg = "This video is unavailable or private."
    except NoTranscriptFound:
        msg = "No transcript found for this video in any language."
    except Exception as e:
        msg = f"Failed to fetch transcript: {str(e)}"

    logger.warning("youtube_fetch_failed", video_id=video_id, reason=msg)
    return {
        "text": "",
        "video_id": video_id,
        "language": None,
        "available_languages": [],
        "segment_count": 0,
        "method": "youtube_transcript_api",
        "error": msg,
    }
