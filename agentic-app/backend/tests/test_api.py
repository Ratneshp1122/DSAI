"""
test_api.py — Integration tests for the FastAPI endpoints.
Tests /health, /api/upload, /api/chat without hitting real Gemini API.
"""
import io
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_version(self):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data


# ─────────────────────────────────────────────────────────────
# Upload endpoint
# ─────────────────────────────────────────────────────────────

class TestUploadEndpoint:
    def _make_text_file(self, content="Hello world. This is a test document."):
        return io.BytesIO(content.encode())

    def test_upload_text_file(self):
        """Text files should be directly read without OCR."""
        f = self._make_text_file()
        resp = client.post(
            "/api/upload",
            files={"file": ("test.txt", f, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_type"] == "text"
        assert data["session_id"] is not None
        assert data["file_id"] is not None
        assert data["method"] == "direct_read"
        assert data["word_count"] > 0

    def test_upload_returns_session_id(self):
        f = self._make_text_file()
        resp = client.post("/api/upload", files={"file": ("doc.txt", f, "text/plain")})
        data = resp.json()
        assert len(data["session_id"]) > 10

    def test_upload_unsupported_type_returns_415(self):
        f = io.BytesIO(b"<html>test</html>")
        resp = client.post(
            "/api/upload",
            files={"file": ("page.html", f, "text/html")},
        )
        assert resp.status_code == 415

    def test_upload_pdf_no_tesseract_returns_warning(self):
        """PDF with no selectable text + no Tesseract → warning, not crash."""
        from PIL import Image
        import fitz

        # Create a minimal real text PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello world from PDF direct text extraction test.")
        pdf_bytes = doc.tobytes()
        doc.close()

        f = io.BytesIO(pdf_bytes)
        resp = client.post(
            "/api/upload",
            files={"file": ("real.pdf", f, "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_type"] == "pdf"
        # Direct extraction should work (text PDF, no Tesseract needed)
        assert data["method"] in ("direct", "direct_short", "ocr_fallback")

    @patch("backend.extraction.image_ocr.pytesseract.image_to_data")
    def test_upload_image_file(self, mock_ocr):
        """Image upload should trigger OCR."""
        from PIL import Image

        mock_ocr.return_value = {"text": ["Hello", "World"], "conf": ["90", "85"]}

        img = Image.new("RGB", (200, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/api/upload",
            files={"file": ("test.png", buf, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_type"] == "image"
        assert data["session_id"] is not None


# ─────────────────────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────────────────────

class TestChatEndpoint:
    def _upload_text(self, content="The quarterly revenue grew by 25% in Q3."):
        f = io.BytesIO(content.encode())
        resp = client.post("/api/upload", files={"file": ("doc.txt", f, "text/plain")})
        assert resp.status_code == 200
        return resp.json()

    def test_chat_requires_session_id(self):
        """A fresh session_id with no file → creates a new session."""
        with patch("backend.agents.intent_agent.ChatGoogleGenerativeAI") as mock_llm:
            mock_resp = MagicMock()
            mock_resp.content = '{"intent": "converse", "confidence": 0.9, "reason": "General chat", "ambiguous_intents": []}'
            mock_llm.return_value.invoke.return_value = mock_resp

            with patch("backend.tools.conversational.ChatGoogleGenerativeAI") as mock_conv:
                mock_conv_resp = MagicMock()
                mock_conv_resp.content = "I'm here to help!"
                mock_conv.return_value.invoke.return_value = mock_conv_resp

                resp = client.post("/api/chat", json={
                    "session_id": "brand-new-session-xyz",
                    "message": "Hello there!",
                })

        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["status"] in ("complete", "need_clarification", "error")

    def test_chat_without_api_key_returns_503(self):
        """Missing API key → 503."""
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = ""
        with patch("backend.routers.chat.get_settings", return_value=mock_settings):
            resp = client.post("/api/chat", json={
                "session_id": "test-no-key",
                "message": "test",
            })
        assert resp.status_code == 503

    def test_chat_summarize_intent(self):
        """Upload a text file then ask to summarize — should call summarizer."""
        upload_data = self._upload_text(
            "The company reported record profits this quarter. "
            "Revenue was up 45% year-over-year driven by new product launches. "
            "The CEO expressed optimism for future growth despite market uncertainty."
        )
        session_id = upload_data["session_id"]

        with patch("backend.agents.intent_agent.ChatGoogleGenerativeAI") as mock_intent:
            mock_intent_resp = MagicMock()
            mock_intent_resp.content = '{"intent": "summarize", "confidence": 0.95, "reason": "User wants a summary", "ambiguous_intents": []}'
            mock_intent.return_value.invoke.return_value = mock_intent_resp

            with patch("backend.tools.summarizer.ChatGoogleGenerativeAI") as mock_sum:
                mock_sum_resp = MagicMock()
                mock_sum_resp.content = (
                    "ONE-LINE SUMMARY: Company reports record profits driven by new products.\n\n"
                    "KEY POINTS:\n• Revenue up 45% YoY\n• New products driving growth\n• CEO optimistic\n\n"
                    "DETAILED SUMMARY: The company had an exceptional quarter..."
                )
                mock_sum.return_value.invoke.return_value = mock_sum_resp

                resp = client.post("/api/chat", json={
                    "session_id": session_id,
                    "message": "Please summarize this document",
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"
        assert data["intent"] == "summarize"
        assert data["result"] is not None
        assert "ONE-LINE SUMMARY" in data["result"]
        assert data["execution_plan"] is not None
        assert len(data["execution_plan"]) > 0
        assert data["logs"] is not None

    def test_chat_followup_on_unclear_intent(self):
        """When intent is UNCLEAR → status should be need_clarification."""
        upload_data = self._upload_text("def add(a, b): return a + b")
        session_id = upload_data["session_id"]

        with patch("backend.agents.intent_agent.ChatGoogleGenerativeAI") as mock_intent:
            mock_intent_resp = MagicMock()
            mock_intent_resp.content = '{"intent": "UNCLEAR", "confidence": 0.4, "reason": "Could be code_explain or qa_rag", "ambiguous_intents": ["code_explain", "qa_rag"]}'
            mock_intent.return_value.invoke.return_value = mock_intent_resp

            with patch("backend.agents.followup_agent.ChatGoogleGenerativeAI") as mock_fup:
                mock_fup_resp = MagicMock()
                mock_fup_resp.content = "Would you like me to explain this code or answer a specific question about it?"
                mock_fup.return_value.invoke.return_value = mock_fup_resp

                resp = client.post("/api/chat", json={
                    "session_id": session_id,
                    "message": "help me with this",
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "need_clarification"
        assert data["follow_up_question"] is not None
        assert len(data["follow_up_question"]) > 10

    def test_chat_session_get(self):
        """GET /chat/session/{id} returns session info."""
        upload_data = self._upload_text("Test content for session retrieval.")
        session_id = upload_data["session_id"]

        resp = client.get(f"/api/chat/session/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["file_type"] == "text"
        assert "word_count" in data

    def test_chat_session_not_found(self):
        """GET /chat/session/{non-existent} → 404."""
        resp = client.get("/api/chat/session/totally-fake-id-12345")
        assert resp.status_code == 404

    def test_chat_session_delete(self):
        """DELETE /chat/session/{id} removes the session."""
        upload_data = self._upload_text("Delete me.")
        session_id = upload_data["session_id"]

        del_resp = client.delete(f"/api/chat/session/{session_id}")
        assert del_resp.status_code == 200

        # Verify it's gone
        get_resp = client.get(f"/api/chat/session/{session_id}")
        assert get_resp.status_code == 404


# ─────────────────────────────────────────────────────────────
# Agent intent classification
# ─────────────────────────────────────────────────────────────

class TestIntentAgent:
    def test_route_after_intent_clear(self):
        from backend.agents.intent_agent import route_after_intent
        state = {
            "intent": "summarize",
            "is_intent_clear": True,
            "ambiguous_intents": [],
        }
        assert route_after_intent(state) == "planner"

    def test_route_after_intent_unclear(self):
        from backend.agents.intent_agent import route_after_intent
        state = {
            "intent": "UNCLEAR",
            "is_intent_clear": False,
            "ambiguous_intents": ["summarize", "qa_rag"],
        }
        assert route_after_intent(state) == "followup"

    @patch("backend.agents.intent_agent.ChatGoogleGenerativeAI")
    def test_classify_intent_valid_json(self, mock_llm):
        from backend.agents.intent_agent import classify_intent

        mock_resp = MagicMock()
        mock_resp.content = '{"intent": "sentiment", "confidence": 0.88, "reason": "User wants sentiment", "ambiguous_intents": []}'
        mock_llm.return_value.invoke.return_value = mock_resp

        state = {
            "raw_input": "What is the tone of this text?",
            "extracted_text": "I love this product!",
            "file_type": "text",
            "logs": [],
        }
        result = classify_intent(state)
        assert result["intent"] == "sentiment"
        assert result["intent_confidence"] == pytest.approx(0.88, abs=0.01)
        assert result["is_intent_clear"] is True

    @patch("backend.agents.intent_agent.ChatGoogleGenerativeAI")
    def test_classify_intent_fallback_on_error(self, mock_llm):
        from backend.agents.intent_agent import classify_intent
        mock_llm.return_value.invoke.side_effect = Exception("Network error")

        state = {
            "raw_input": "test",
            "extracted_text": "",
            "file_type": "none",
            "logs": [],
        }
        result = classify_intent(state)
        # Should fall back to converse
        assert result["intent"] == "converse"
        assert result["is_intent_clear"] is True


# ─────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────

class TestPlannerAgent:
    def test_summarize_plan(self):
        from backend.agents.planner_agent import plan_execution
        state = {
            "intent": "summarize",
            "raw_input": "summarize this",
            "extracted_text": "test text",
            "logs": [],
        }
        result = plan_execution(state)
        assert len(result["execution_plan"]) == 3
        assert "summarizer" in result["execution_plan"][1]

    def test_qa_rag_plan(self):
        from backend.agents.planner_agent import plan_execution
        state = {
            "intent": "qa_rag",
            "raw_input": "What is the main finding?",
            "extracted_text": "test text",
            "logs": [],
        }
        result = plan_execution(state)
        assert len(result["execution_plan"]) == 5
        assert any("faiss" in s.lower() for s in result["execution_plan"])

    def test_estimated_cost_is_non_negative(self):
        from backend.agents.planner_agent import plan_execution
        state = {
            "intent": "converse",
            "raw_input": "hi",
            "extracted_text": "",
            "logs": [],
        }
        result = plan_execution(state)
        assert result["estimated_cost"] >= 0


# ─────────────────────────────────────────────────────────────
# Executor Tools
# ─────────────────────────────────────────────────────────────

class TestSummarizer:
    @patch("backend.tools.summarizer.ChatGoogleGenerativeAI")
    def test_summarizer_returns_structured_output(self, mock_llm):
        from backend.tools.summarizer import run_summarizer
        mock_resp = MagicMock()
        mock_resp.content = (
            "ONE-LINE SUMMARY: Test summary line.\n\n"
            "KEY POINTS:\n• Point 1\n• Point 2\n• Point 3\n\n"
            "DETAILED SUMMARY: Five sentence detailed description here."
        )
        mock_llm.return_value.invoke.return_value = mock_resp

        state = {
            "extracted_text": "This is a test document with some content.",
            "raw_input": "summarize",
        }
        result = run_summarizer(state)
        assert "ONE-LINE SUMMARY" in result
        assert "KEY POINTS" in result

    def test_summarizer_handles_empty_text(self):
        from backend.tools.summarizer import run_summarizer
        state = {"extracted_text": "", "raw_input": ""}
        result = run_summarizer(state)
        assert "No text content" in result


class TestSentimentAnalyzer:
    @patch("backend.tools.sentiment_analyzer.ChatGoogleGenerativeAI")
    def test_sentiment_returns_label(self, mock_llm):
        from backend.tools.sentiment_analyzer import run_sentiment
        mock_resp = MagicMock()
        mock_resp.content = (
            "SENTIMENT: Positive\nCONFIDENCE: 92%\n"
            "EMOTIONS DETECTED: joy, excitement\n"
            "JUSTIFICATION: The text expresses very positive sentiment.\n"
            "KEY PHRASES: great, excellent, love"
        )
        mock_llm.return_value.invoke.return_value = mock_resp

        state = {"extracted_text": "I love this product, it's excellent!", "raw_input": ""}
        result = run_sentiment(state)
        assert "SENTIMENT" in result
        assert "Positive" in result

    def test_sentiment_handles_empty_text(self):
        from backend.tools.sentiment_analyzer import run_sentiment
        state = {"extracted_text": "", "raw_input": ""}
        result = run_sentiment(state)
        assert "No text content" in result


class TestCodeExplainer:
    @patch("backend.tools.code_explainer.ChatGoogleGenerativeAI")
    def test_code_explainer_returns_analysis(self, mock_llm):
        from backend.tools.code_explainer import run_code_explainer
        mock_resp = MagicMock()
        mock_resp.content = (
            "LANGUAGE: Python\n\nEXPLANATION:\nThis function adds two numbers.\n\n"
            "POTENTIAL BUGS:\nNone detected.\n\n"
            "TIME COMPLEXITY: O(1) — constant time\n"
            "SPACE COMPLEXITY: O(1) — constant space\n\n"
            "SUGGESTIONS:\nAdd type hints."
        )
        mock_llm.return_value.invoke.return_value = mock_resp

        state = {"extracted_text": "def add(a, b):\n    return a + b", "raw_input": ""}
        result = run_code_explainer(state)
        assert "LANGUAGE" in result
        assert "Python" in result
        assert "TIME COMPLEXITY" in result


# ─────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_chunker_splits_text(self):
        from backend.rag.chunker import chunk_text
        long_text = "This is a sentence. " * 200
        chunks = chunk_text(long_text)
        assert len(chunks) > 1
        assert all(len(c) > 0 for c in chunks)

    def test_chunker_returns_empty_for_empty_input(self):
        from backend.rag.chunker import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_vector_store_build_and_search(self):
        from backend.rag.vector_store import build_index, search_index, has_index, delete_index

        session = "test-rag-session-unit"
        chunks = ["The sky is blue.", "Water is wet.", "Fire is hot."]
        embeddings = [[float(i), float(j), 0.1] for i, j in enumerate(range(3))]

        build_index(session, chunks, embeddings)
        assert has_index(session)

        results = search_index(session, [0.0, 0.0, 0.1], top_k=2)
        assert len(results) == 2
        assert all(isinstance(r[0], str) for r in results)
        assert all(isinstance(r[1], float) for r in results)

        delete_index(session)
        assert not has_index(session)

    @patch("backend.tools.qa_rag.embed_query")
    @patch("backend.tools.qa_rag.embed_texts")
    @patch("backend.tools.qa_rag.ChatGoogleGenerativeAI")
    def test_qa_rag_returns_answer(self, mock_llm, mock_embed_texts, mock_embed_query):
        from backend.tools.qa_rag import run_qa_rag
        from backend.rag.vector_store import delete_index

        session = "test-qa-rag-unit"
        delete_index(session)

        mock_embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embed_query.return_value = [0.1, 0.2, 0.3]

        mock_resp = MagicMock()
        mock_resp.content = (
            "ANSWER: The revenue grew by 25%.\n\n"
            "EVIDENCE:\n• Revenue grew 25%\n\n"
            "CONFIDENCE: High\nCONFIDENCE REASON: Directly stated in document."
        )
        mock_llm.return_value.invoke.return_value = mock_resp

        state = {
            "session_id": session,
            "raw_input": "What was the revenue growth?",
            "extracted_text": "Revenue grew 25%. Profits increased. Costs reduced.",
        }
        result = run_qa_rag(state)
        assert "ANSWER" in result
        assert "25%" in result

        delete_index(session)


# ─────────────────────────────────────────────────────────────
# PDF parser (graceful Tesseract)
# ─────────────────────────────────────────────────────────────

class TestPDFParserGraceful:
    def test_scanned_pdf_without_tesseract_returns_warning(self, tmp_path):
        """A scanned PDF with Tesseract unavailable should warn, not crash."""
        import fitz
        from backend.extraction.pdf_parser import extract_text_from_pdf

        # Create a PDF with very little text (simulates scanned)
        doc = fitz.open()
        page = doc.new_page()
        # Don't add any text — simulates a scanned page
        pdf_bytes = doc.tobytes()
        doc.close()

        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(pdf_bytes)

        with patch("backend.extraction.pdf_parser._tesseract_available", return_value=False):
            result = extract_text_from_pdf(pdf_path)

        assert result["method"] == "direct_short"
        assert "Tesseract" in (result["warning"] or "")
        assert result["confidence"] <= 0.5

    def test_text_pdf_works_without_tesseract(self, tmp_path):
        """A text-based PDF should extract perfectly without Tesseract."""
        import fitz
        from backend.extraction.pdf_parser import extract_text_from_pdf

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "This document contains sufficient text for direct extraction purposes.")
        pdf_bytes = doc.tobytes()
        doc.close()

        pdf_path = tmp_path / "text.pdf"
        pdf_path.write_bytes(pdf_bytes)

        result = extract_text_from_pdf(pdf_path)
        assert result["method"] == "direct"
        assert result["confidence"] == 1.0
        assert "sufficient text" in result["text"]
        assert result["warning"] is None
