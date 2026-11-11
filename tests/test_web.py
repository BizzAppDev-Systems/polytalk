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

    def test_frontend_persists_and_sends_custom_instruction(self):
        """Frontend stores instructions and encodes them on translation sessions."""
        script = (
            Path(__file__).resolve().parents[1] / "app/static/js/polytalk-app.js"
        ).read_text()

        assert "polytalk_custom_instruction" in script
        assert "buildTranslationWebSocketUrl" in script
        assert "new URLSearchParams" in script
        assert "params.set('custom_instruction', customInstruction)" in script

    def test_frontend_buffers_audio_until_translation_socket_opens(self):
        """Startup PCM is retained and flushed instead of being silently dropped."""
        script = (
            Path(__file__).resolve().parents[1] / "app/static/js/polytalk-app.js"
        ).read_text()

        assert "beginStartupAudioBuffering()" in script
        assert "sendOrBufferAudioChunk(chunk)" in script
        assert "flushStartupAudioBuffer()" in script
        assert "this.maxStartupAudioBufferBytes = 16000 * 2 * 5" in script
        assert script.count("this.sendOrBufferAudioChunk(chunk);") == 2
        assert script.count("this.flushStartupAudioBuffer();") >= 2
