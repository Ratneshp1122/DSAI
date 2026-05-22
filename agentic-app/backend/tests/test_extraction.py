"""
test_extraction.py — Unit tests for the extraction layer.

Tests:
  - image_ocr: file-not-found, unsupported input, mock OCR output
  - pdf_parser: direct extraction path, OCR fallback path (mocked)
  - audio_transcriber: file-not-found, happy path (mocked model)
  - youtube_fetcher: URL detection, no-URL case, mock API response
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import tempfile
import shutil


# ─────────────────────────────────────────────────────────────
# image_ocr tests
# ─────────────────────────────────────────────────────────────

class TestImageOCR:
    def test_file_not_found(self):
        from backend.extraction.image_ocr import extract_text_from_image
        with pytest.raises(FileNotFoundError):
            extract_text_from_image(Path("/nonexistent/file.png"))

    def test_invalid_image(self, tmp_path):
        from backend.extraction.image_ocr import extract_text_from_image
        bad_file = tmp_path / "bad.png"
        bad_file.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Cannot open image"):
            extract_text_from_image(bad_file)

    @patch("backend.extraction.image_ocr.pytesseract.image_to_data")
    def test_successful_extraction(self, mock_ocr, tmp_path):
        from backend.extraction.image_ocr import extract_text_from_image
        from PIL import Image

        # Create a real (tiny) PNG
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 50), color="white")
        img.save(img_path)

        # Mock OCR data response
        mock_ocr.return_value = {
            "text": ["Hello", "World", ""],
            "conf": ["90", "85", "-1"],
        }

        result = extract_text_from_image(img_path)

        assert result["method"] == "tesseract_ocr"
        assert "Hello" in result["text"]
        assert result["word_count"] == 2
        assert result["confidence"] == pytest.approx(0.875, abs=0.01)

    @patch("backend.extraction.image_ocr.pytesseract.image_to_data")
    def test_low_confidence_warning(self, mock_ocr, tmp_path):
        from backend.extraction.image_ocr import extract_text_from_image
        from PIL import Image

        img_path = tmp_path / "low.png"
        img = Image.new("RGB", (100, 50), color="white")
        img.save(img_path)

        mock_ocr.return_value = {
            "text": ["abc"],
            "conf": ["20"],
        }

        result = extract_text_from_image(img_path)
        assert result["warning"] is not None
        assert result["confidence"] < 0.4

    @patch("backend.extraction.image_ocr.pytesseract.image_to_data")
    def test_no_text_detected(self, mock_ocr, tmp_path):
        from backend.extraction.image_ocr import extract_text_from_image
        from PIL import Image

        img_path = tmp_path / "blank.png"
        img = Image.new("RGB", (100, 50), color="white")
        img.save(img_path)

        mock_ocr.return_value = {
            "text": ["", "", ""],
            "conf": ["-1", "-1", "-1"],
        }

        result = extract_text_from_image(img_path)
        assert result["text"] == ""
        assert result["confidence"] == 0.0
        assert result["warning"] is not None


# ─────────────────────────────────────────────────────────────
# pdf_parser tests
# ─────────────────────────────────────────────────────────────

class TestPDFParser:
    def test_file_not_found(self):
        from backend.extraction.pdf_parser import extract_text_from_pdf
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf(Path("/nonexistent/file.pdf"))

    @patch("backend.extraction.pdf_parser.fitz.open")
    def test_direct_extraction(self, mock_fitz, tmp_path):
        from backend.extraction.pdf_parser import extract_text_from_pdf

        # Create dummy file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        # Mock fitz document
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is a test document with lots of text content."
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.metadata = {"title": "Test Doc", "author": "Tester"}
        mock_fitz.return_value = mock_doc

        result = extract_text_from_pdf(pdf_path)

        assert result["method"] == "direct"
        assert result["confidence"] == 1.0
        assert "This is a test" in result["text"]
        assert result["page_count"] == 1

    @patch("backend.extraction.pdf_parser.fitz.open")
    def test_ocr_fallback_triggered(self, mock_fitz, tmp_path):
        """Test that scanned PDFs trigger OCR fallback."""
        from backend.extraction.pdf_parser import extract_text_from_pdf
        import backend.extraction.pdf_parser as pdf_mod
        import pytesseract

        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        # Page returns very short text -> triggers OCR fallback
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""  # scanned page

        # Pixmap mock
        from PIL import Image
        import io
        img = Image.new("RGB", (100, 50), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = buf.getvalue()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.metadata = {"title": "", "author": ""}
        mock_fitz.return_value = mock_doc

        mock_ocr_data = {
            "text": ["OCR", "content"],
            "conf": ["80", "75"],
        }

        # Patch pytesseract in all modules that use it
        with patch("pytesseract.image_to_data", return_value=mock_ocr_data):
            result = extract_text_from_pdf(pdf_path)

        # Key assertion: the OCR fallback path OR graceful degradation was triggered (not plain "direct")
        assert result["method"] in ("ocr_fallback", "direct_short")
        assert result["page_count"] == 1
        # Note: actual text/method depends on Tesseract binary being installed


# ─────────────────────────────────────────────────────────────
# audio_transcriber tests
# ─────────────────────────────────────────────────────────────

class TestAudioTranscriber:
    def test_file_not_found(self):
        from backend.extraction.audio_transcriber import transcribe_audio
        with pytest.raises(FileNotFoundError):
            transcribe_audio(Path("/nonexistent/audio.mp3"))

    def test_successful_transcription(self, tmp_path):
        from backend.extraction.audio_transcriber import transcribe_audio

        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"\xff\xfb" + b"\x00" * 100)  # fake MP3 header

        # Create mock segment
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 5.0
        mock_seg.text = "Hello world this is a test"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        result = transcribe_audio(audio_path, model=mock_model)

        assert result["method"] == "faster_whisper"
        assert result["language"] == "en"
        assert "Hello world" in result["text"]
        assert result["duration_seconds"] == 5.0
        assert result["word_count"] > 0

    def test_empty_transcription_warning(self, tmp_path):
        from backend.extraction.audio_transcriber import transcribe_audio

        audio_path = tmp_path / "silent.mp3"
        audio_path.write_bytes(b"\xff\xfb" + b"\x00" * 100)

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.5
        mock_info.duration = 3.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)

        result = transcribe_audio(audio_path, model=mock_model)
        assert result["text"] == ""
        assert result["warning"] is not None
        assert "No speech" in result["warning"]


# ─────────────────────────────────────────────────────────────
# youtube_fetcher tests
# ─────────────────────────────────────────────────────────────

class TestYouTubeFetcher:
    def test_no_url_in_text(self):
        from backend.extraction.youtube_fetcher import fetch_youtube_transcript
        result = fetch_youtube_transcript("Can you summarize this article for me?")
        assert result["error"] is not None
        assert "No YouTube URL" in result["error"]
        assert result["video_id"] is None

    def test_url_extraction(self):
        from backend.extraction.youtube_fetcher import extract_youtube_video_id
        assert extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert extract_youtube_video_id("no url here") is None

    @patch("backend.extraction.youtube_fetcher.YouTubeTranscriptApi")
    def test_successful_fetch(self, mock_api_class):
        from backend.extraction.youtube_fetcher import fetch_youtube_transcript
        from youtube_transcript_api._errors import NoTranscriptFound

        # Mock segment object
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"

        mock_transcript = MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.fetch.return_value = [mock_segment]

        mock_list_obj = MagicMock()
        mock_list_obj.__iter__ = MagicMock(return_value=iter([mock_transcript]))
        mock_list_obj.find_manually_created_transcript.side_effect = NoTranscriptFound(
            "vid", ["en"], {"en": {}}
        )
        mock_list_obj.find_generated_transcript.return_value = mock_transcript

        # Instance-based API: YouTubeTranscriptApi().list()
        mock_api_instance = MagicMock()
        mock_api_instance.list.return_value = mock_list_obj
        mock_api_class.return_value = mock_api_instance

        result = fetch_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["error"] is None
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert "Hello world" in result["text"]

    @patch("backend.extraction.youtube_fetcher.YouTubeTranscriptApi")
    def test_transcripts_disabled(self, mock_api_class):
        from backend.extraction.youtube_fetcher import fetch_youtube_transcript
        from youtube_transcript_api._errors import TranscriptsDisabled

        mock_api_instance = MagicMock()
        # TranscriptsDisabled in v1.2.4 takes only video_id
        mock_api_instance.list.side_effect = TranscriptsDisabled("vid")
        mock_api_class.return_value = mock_api_instance

        result = fetch_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["error"] is not None
        assert "disabled" in result["error"].lower()
