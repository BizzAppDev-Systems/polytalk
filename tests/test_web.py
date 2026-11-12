# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for web router.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestWebRouter:
    """Test web router endpoints."""

    def test_home_page(self, client):
        """Test home page rendering."""
        with patch("app.routers.web.get_config") as mock_config:
            mock_config.return_value.app = {
                "auto_play_audio": True,
                "name": "PolyTalk",
            }
            mock_config.return_value.translation = {
                "custom_instruction_max_chars": 250,
            }

            response = client.get("/")

            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_home_page_includes_custom_instruction_setting(self, client):
        """Settings modal exposes bounded custom AI instructions."""
        with patch("app.routers.web.get_config") as mock_config:
            mock_config.return_value.app = {"auto_play_audio": True}
            mock_config.return_value.translation = {
                "custom_instruction_max_chars": 123,
            }

            response = client.get("/")

        assert response.status_code == 200
        assert 'id="custom-instruction-input"' in response.text
        assert 'maxlength="123"' in response.text
        assert 'aria-describedby="custom-instruction-count"' in response.text
        assert 'id="custom-instruction-count"' in response.text
        assert "Translation Instructions" in response.text

    def test_home_page_includes_session_translation_pacing(self, client):
        """Advanced settings render bounded pacing from the server default."""
        with patch("app.routers.web.get_config") as mock_config:
            mock_config.return_value.app = {
                "auto_play_audio": True,
                "translation_flush_seconds": "4.5",
            }
            mock_config.return_value.translation = {
                "custom_instruction_max_chars": 250,
            }

            response = client.get("/")

        assert response.status_code == 200
        assert 'id="translation-buffer-seconds"' in response.text
        assert 'min="1.0"' in response.text
        assert 'max="10.0"' in response.text
        assert 'step="0.5"' in response.text
        assert 'value="4.5"' in response.text
        assert "The live transcript remains immediate." in response.text

    def test_home_page_includes_typed_text_translation(self, client):
        """The OSS UI exposes Unicode text translation with on-demand speech."""
        with patch("app.routers.web.get_config") as mock_config:
            mock_config.return_value.app = {"auto_play_audio": True}
            mock_config.return_value.translation = {
                "custom_instruction_max_chars": 250,
            }

            response = client.get("/")

        assert response.status_code == 200
        assert 'id="text-source-input"' in response.text
        assert 'id="text-mode-btn"' in response.text
        assert 'aria-controls="text-translation-card"' in response.text
        assert 'data-max-characters="5000"' in response.text
        assert 'dir="auto"' in response.text
        assert 'id="text-audio-grid"' in response.text
        assert 'id="text-source-speak-btn"' in response.text
        assert 'id="text-source-audio-player"' in response.text
        assert 'id="text-source-audio-panel"' in response.text
        assert 'id="text-translation-audio-panel"' in response.text
        assert 'id="text-speak-btn"' in response.text
        assert 'id="text-audio-player"' in response.text
        assert "action-icon-text-translation" in response.text
        assert (
            '<span class="btn-text action-label">Text Translation</span>'
            not in response.text
        )
        assert "/static/js/text-translation.js" in response.text

        script = (
            Path(__file__).parents[1] / "app" / "static" / "js" / "text-translation.js"
        ).read_text()
        assert "fetch('/api/text/translations'" in script
        assert "fetch('/api/text/speech'" in script
        assert "kind === 'source' ? sourceText : translatedText" in script
        assert (
            "kind === 'source' ? sourceLanguage.value : targetLanguage.value" in script
        )
        assert "syncAudioGridVisibility()" in script
        assert "showOnlyAudioPanel(audio)" in script
        assert "candidate.panel.hidden = true" in script
        assert "audio.panel.hidden = false" in script
        assert "if (missingRequiredElement || incompleteAudioTrack)" in script
        assert "Missing required DOM element" in script
        assert "Math.min(characters, maxCharacters)" in script
        assert "[PolyTalk] Audio playback failed:" in script
        assert "[...segmenter.segment(str)]" in script
        assert "segments.slice(0, max).map(s => s.segment).join('')" in script
        assert "URL.revokeObjectURL" in script
        assert "text-mode-active" in script
        assert "setTextModeOpen(false)" in script
        css = (
            Path(__file__).parents[1]
            / "app"
            / "static"
            / "css"
            / "text-translation.css"
        ).read_text()
        assert "body.text-mode-active .live-results" in css
        assert "body.text-mode-active #audio-container" in css
        assert ".text-audio-grid" in css
        assert "grid-template-columns: minmax(0, 1fr)" in css
        assert "#text-source-audio-panel" in css
        assert "#text-translation-audio-panel" in css
        assert ".text-audio-panel > audio" in css

    def test_frontend_persists_and_sends_custom_instruction(self):
        """Frontend stores instructions and encodes them on translation sessions."""
        script = (
            Path(__file__).resolve().parents[1] / "app/static/js/polytalk-app.js"
        ).read_text()

        assert "polytalk_custom_instruction" in script
        assert "buildTranslationWebSocketUrl" in script
        assert "new URLSearchParams" in script
        assert "params.set('custom_instruction', customInstruction)" in script
