# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for TTS service methods.
"""

from pathlib import Path
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.tts_service import TTSService
from app.utils.sanitize import normalize_tts_text


class TestTTSServiceInit:
    """Test TTS service initialization."""

    def test_tts_init_default(self):
        """Test TTS service initialization with defaults."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": True,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}
            service = TTSService()
            assert service.enabled is True
            assert service.mock_mode is True
            assert service.provider == "piper"
            assert service.base_url == "http://localhost:5000"
            assert service.voice == "en_US-lessac-medium"
            assert service.timeout == 15

    def test_tts_init_disabled(self):
        """Test TTS service initialization when disabled."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": False,
                "mock_mode": True,
            }
            mock_config.return_value.app = {}
            service = TTSService()
            assert service.enabled is False

    def test_tts_init_custom_provider(self):
        """Test TTS service initialization with custom provider."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "openai",
                "base_url": "https://api.openai.com",
                "voice": "alloy",
            }
            mock_config.return_value.app = {}
            service = TTSService()
            assert service.provider == "openai"


class TestTTSProviderRouting:
    """Test language-specific TTS provider routing."""

    def test_language_provider_routing(self):
        """Test configured languages route to Supertonic."""
        supertonic_languages = {
            "ar",
            "bg",
            "hr",
            "et",
            "hi",
            "ru",
            "id",
            "vi",
            "pt",
            "pl",
            "uk",
            "sv",
            "da",
            "fi",
            "cs",
            "el",
            "hu",
            "lv",
            "lt",
            "sk",
            "sl",
            "ko",
            "ja",
        }
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://piper:5000",
                "language_providers": {
                    language: "supertonic" for language in supertonic_languages
                },
            }
            mock_config.return_value.app = {}
            service = TTSService()

            for language in supertonic_languages:
                assert service._get_provider_for_language(language) == "supertonic"
            assert service._get_provider_for_language("ko-KR") == "supertonic"
            assert service._get_provider_for_language("ja-JP") == "supertonic"
            assert service._get_provider_for_language("zh") == "piper"
            assert service._get_provider_for_language("en") == "piper"

    def test_supertonic_voice_routing(self):
        """Test Supertonic can use per-language voices."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "providers": {
                    "supertonic": {
                        "voice": "M1",
                        "voices": {"ko": "F1", "ja": "M2"},
                    }
                },
            }
            mock_config.return_value.app = {}
            service = TTSService()

            assert service._get_supertonic_voice_for_language("ko-KR") == "F1"
            assert service._get_supertonic_voice_for_language("ja-JP") == "M2"
            assert service._get_supertonic_voice_for_language("en") == "M1"


class TestTTSSynthesize:
    """Test TTS synthesis methods."""

    @pytest.mark.asyncio
    async def test_synthesize_supertonic_success(self, tmp_path):
        """Test synthesis through Supertonic provider routing."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://piper:5000",
                "timeout_seconds": 10,
                "language_providers": {"ko": "supertonic"},
                "providers": {
                    "supertonic": {
                        "base_url": "http://supertonic-tts:7788",
                        "voice": "M1",
                        "voices": {"ko": "F1"},
                        "steps": 6,
                        "speed": 1.1,
                        "response_format": "wav",
                    }
                },
            }
            mock_config.return_value.app = {}
            mock_config.return_value.media_output_dir = tmp_path
            service = TTSService()

            response = httpx.Response(
                200,
                content=b"RIFFtest",
                headers={"X-Audio-Duration": "1.25"},
                request=httpx.Request("POST", "http://supertonic-tts:7788/v1/tts"),
            )

            class MockStream:
                async def __aenter__(self):
                    return response

                async def __aexit__(self, exc_type, exc, traceback):
                    return None

            service._http_client.stream = MagicMock(return_value=MockStream())

            output_path = tmp_path / "ko.wav"
            result = await service.synthesize("안녕하세요", "ko-KR", output_path)

            assert result.success is True
            assert result.audio_path == output_path
            assert result.duration == 1.25
            assert output_path.read_bytes() == b"RIFFtest"
            service._http_client.stream.assert_called_once_with(
                "POST",
                "http://supertonic-tts:7788/v1/tts",
                json={
                    "text": "안녕하세요",
                    "voice": "F1",
                    "lang": "ko",
                    "steps": 6,
                    "speed": 1.1,
                    "response_format": "wav",
                },
            )

    @pytest.mark.asyncio
    async def test_synthesize_service_disabled(self):
        """Test synthesis when service is disabled."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": False,
                "mock_mode": True,
            }
            mock_config.return_value.app = {}
            service = TTSService()
            result = await service.synthesize("Hello", "en")

            assert result.success is False
            assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_synthesize_mock_mode(self):
        """Test synthesis in mock mode."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": True,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)
                service = TTSService()
                result = await service.synthesize("Hello world", "en")

                assert result.success is True
                assert result.audio_url is not None


class TestTTSMockSynthesize:
    """Test mock synthesis methods."""

    @pytest.mark.asyncio
    async def test_mock_synthesize_default_path(self):
        """Test mock synthesis with default path."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": True,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)
                service = TTSService()
                result = await service._mock_synthesize("Hello", "en")

                assert result.success is True
                assert result.audio_path is not None
                assert result.audio_url is not None

    @pytest.mark.asyncio
    async def test_mock_synthesize_custom_path(self):
        """Test mock synthesis with custom path."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": True,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)
                service = TTSService()
                custom_path = Path(tmpdir) / "custom.wav"
                result = await service._mock_synthesize(
                    "Hello", "en", output_path=custom_path
                )

                assert result.success is True
                assert result.audio_path == custom_path


class TestTTSServiceGetVoiceForLanguage:
    """Test TTS voice selection for languages."""

    @pytest.mark.asyncio
    async def test_get_voice_default_config(self):
        """Test getting voice from default config."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "voice": "en_US-lessac-medium",
                    "default_voices": {"en": "en_US-lessac-medium"},
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                with patch.object(
                    service,
                    "_fetch_voices",
                    return_value={"en_US-lessac-medium.onnx": {}},
                ):
                    voice = await service._get_voice_for_language("en")

                assert voice == "en_US-lessac-medium"


class TestTTSServicePiperSynthesize:
    """Test Piper TTS synthesis."""

    @pytest.mark.asyncio
    async def test_piper_synthesize_success(self):
        """Test Piper synthesis success."""
        import tempfile
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "voice": "en_US-lessac-medium",
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b"fake_audio_data"
                mock_response.raise_for_status.return_value = None

                with patch.object(
                    service._http_client, "post", return_value=mock_response
                ):
                    with patch.object(
                        service, "_get_voice_for_language", return_value="en_US-test"
                    ):
                        result = await service._piper_synthesize("Hello", "en")

                        assert result.success is True
                        assert result.audio_path is not None

    @pytest.mark.asyncio
    async def test_piper_synthesize_timeout(self):
        """Test Piper synthesis timeout."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "timeout_seconds": 15,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                with patch.object(
                    service._http_client, "post", new_callable=AsyncMock
                ) as mock_post:
                    mock_post.side_effect = httpx.TimeoutException("Timeout")

                    with patch.object(
                        service, "_get_voice_for_language", return_value="en_US-test"
                    ):
                        result = await service._piper_synthesize("Hello", "en")

                        assert result.success is False
                        assert "timeout" in result.error.lower()


class TestTTSServiceOpenaiSynthesize:
    """Test OpenAI TTS synthesis."""

    @pytest.mark.asyncio
    async def test_openai_synthesize_success(self):
        """Test OpenAI synthesis success."""
        import tempfile
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "openai",
                    "base_url": "https://api.openai.com",
                    "voice": "alloy",
                    "api_key": "test_key",
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b"fake_mp3_data"
                mock_response.raise_for_status.return_value = None

                with patch.object(
                    service._http_client, "post", return_value=mock_response
                ):
                    result = await service._openai_synthesize("Hello", "en")

                    assert result.success is True
                    assert result.audio_path is not None


class TestTTSServiceClose:
    """Test TTS service close."""

    @pytest.mark.asyncio
    async def test_close_service(self):
        """Test closing TTS service."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": True,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                with patch.object(
                    service._http_client, "aclose", new_callable=AsyncMock
                ):
                    await service.close()


class TestTTSPiperSynthesizeErrorHandling:
    """Test Piper TTS error handling."""

    @pytest.mark.asyncio
    async def test_piper_synthesize_http_400_error(self):
        """Test Piper synthesis with HTTP 400 error."""
        import tempfile
        import httpx

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "timeout_seconds": 15,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                with patch.object(
                    service._http_client, "post", new_callable=AsyncMock
                ) as mock_post:
                    mock_post.side_effect = httpx.HTTPError("HTTP 400")

                    with patch.object(
                        service, "_get_voice_for_language", return_value="en_US-test"
                    ):
                        result = await service._piper_synthesize("Hello", "en")

                        assert result.success is False
                        assert "http error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_piper_synthesize_http_500_error(self):
        """Test Piper synthesis with HTTP 500 error."""
        import tempfile
        import httpx

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                with patch.object(
                    service._http_client, "post", new_callable=AsyncMock
                ) as mock_post:
                    mock_post.side_effect = httpx.HTTPStatusError(
                        "Server Error",
                        request=MagicMock(),
                        response=MagicMock(status_code=500),
                    )

                    with patch.object(
                        service, "_get_voice_for_language", return_value="en_US-test"
                    ):
                        result = await service._piper_synthesize("Hello", "en")

                        assert result.success is False


class TestTTSOpenaiSynthesizeErrorHandling:
    """Test OpenAI TTS error handling."""

    @pytest.mark.asyncio
    async def test_openai_synthesize_http_error(self):
        """Test OpenAI synthesis with HTTP error."""
        import tempfile
        import httpx

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "openai",
                    "base_url": "https://api.openai.com",
                    "voice": "alloy",
                    "api_key": "test_key",
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                with patch.object(
                    service._http_client, "post", new_callable=AsyncMock
                ) as mock_post:
                    mock_post.side_effect = httpx.HTTPError("API Error")

                    result = await service._openai_synthesize("Hello", "en")

                    assert result.success is False
                    assert "http error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_openai_synthesize_different_status_codes(self):
        """Test OpenAI synthesis with different HTTP status codes."""
        import tempfile
        import httpx

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "openai",
                    "base_url": "https://api.openai.com",
                    "voice": "alloy",
                    "api_key": "test_key",
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                for status_code in [401, 403, 429]:
                    with patch.object(
                        service._http_client, "post", new_callable=AsyncMock
                    ) as mock_post:
                        mock_post.side_effect = httpx.HTTPStatusError(
                            f"HTTP {status_code}",
                            request=MagicMock(),
                            response=MagicMock(status_code=status_code),
                        )

                        result = await service._openai_synthesize("Hello", "en")
                        assert result.success is False


class TestTTSMockSynthesizeErrorHandling:
    """Test mock synthesis error handling."""

    @pytest.mark.asyncio
    async def test_mock_synthesize_permission_error(self):
        """Test mock synthesis with permission error."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": True,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                with patch(
                    "wave.open", side_effect=PermissionError("Permission denied")
                ):
                    result = await service._mock_synthesize("Hello", "en")

                    assert result.success is False
                    assert "failed" in result.error.lower()


class TestTTSSynthesizeEdgeCases:
    """Test TTS synthesis edge cases."""

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self):
        """Test synthesis with empty text."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": True,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()
                result = await service.synthesize("", "en")

                assert result.success is True

    @pytest.mark.asyncio
    async def test_synthesize_very_long_text(self):
        """Test synthesis with very long text."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": True,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()
                long_text = "Hello " * 1000
                result = await service.synthesize(long_text, "en")

                assert result.success is True


class TestTTSVoiceSelection:
    """Test TTS voice selection logic."""

    @pytest.mark.asyncio
    async def test_get_voice_for_language_with_default(self):
        """Test voice selection with default configured."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "voice": "default_voice",
                    "default_voices": {"en": "custom_en_voice"},
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                with patch.object(
                    service, "_fetch_voices", return_value={"custom_en_voice.onnx": {}}
                ):
                    voice = await service._get_voice_for_language("en")

                assert voice == "custom_en_voice"

    @pytest.mark.asyncio
    async def test_get_voice_for_language_with_voices_api(self):
        """Test voice selection from voices API."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "voice": "default_voice",
                    "default_voices": {},
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                mock_voices = {"en_US-test": {}, "en_GB-other": {}}

                with patch.object(service, "_fetch_voices", return_value=mock_voices):
                    voice = await service._get_voice_for_language("en_US")
                    assert voice == "en_US-test"

    @pytest.mark.asyncio
    async def test_get_voice_for_language_base_language_fallback(self):
        """Test voice selection with base language fallback."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "voice": "default_voice",
                    "default_voices": {},
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                mock_voices = {"en_US-test": {}}

                with patch.object(service, "_fetch_voices", return_value=mock_voices):
                    voice = await service._get_voice_for_language("en-GB")
                    assert voice == "en_US-test"

    @pytest.mark.asyncio
    async def test_fetch_voices_success(self):
        """Test fetching voices from API."""
        import tempfile
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"voice1": {}, "voice2": {}}

                with patch.object(
                    service._http_client, "post", return_value=mock_response
                ):
                    with patch.object(
                        service._http_client, "get", return_value=mock_response
                    ):
                        voices = await service._fetch_voices()
                        assert len(voices) == 2

    @pytest.mark.asyncio
    async def test_fetch_voices_timeout(self):
        """Test fetching voices with timeout."""
        import tempfile
        import httpx

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                with patch.object(
                    service._http_client, "get", new_callable=AsyncMock
                ) as mock_get:
                    mock_get.side_effect = httpx.TimeoutException("Timeout")

                    voices = await service._fetch_voices()
                    assert voices == {}

    @pytest.mark.asyncio
    async def test_fetch_voices_cache_hit(self):
        """Test voices cache hit."""
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()
                service._voices_cache = {"cached": {}}
                service._voices_cache_timestamp = time.time()

                voices = await service._fetch_voices()
                assert voices == {"cached": {}}

    @pytest.mark.asyncio
    async def test_real_speak_with_http_error(self):
        """Test real speak handles HTTP error."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal server error"

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.synthesize("Hello", "en")
                assert result.success is False


class TestNormalizeTTS:
    """Test normalize_tts_text utility function."""

    def test_normalize_tts_no_change_for_ascii(self):
        """Text with only ASCII characters should remain unchanged."""
        assert normalize_tts_text("Hello World") == "Hello World"
        assert normalize_tts_text("123 abc") == "123 abc"
        assert (
            normalize_tts_text("now some of your parents") == "now some of your parents"
        )

    def test_normalize_tts_devanagari_digits(self):
        """Devanagari digits (Hindi, Marathi) should be converted to ASCII."""
        assert normalize_tts_text("कीमत ० रुपये") == "कीमत 0 रुपये"
        assert normalize_tts_text("संख्या १२३") == "संख्या 123"
        assert normalize_tts_text("अंक ०१२३४५६७८९") == "अंक 0123456789"

    def test_normalize_tts_gujarati_digits(self):
        """Gujarati digits should be converted to ASCII."""
        assert normalize_tts_text("કિંમત ૦") == "કિંમત 0"
        assert normalize_tts_text("અંક ૦૧૨૩૪૫૬૭૮૯") == "અંક 0123456789"

    def test_normalize_tts_bengali_digits(self):
        """Bengali digits should be converted to ASCII."""
        assert normalize_tts_text("মূল্য ০") == "মূল্য 0"
        assert normalize_tts_text("০১২৩৪৫৬৭৮৯") == "0123456789"

    def test_normalize_tts_gurmukhi_digits(self):
        """Gurmukhi digits should be converted to ASCII."""
        assert normalize_tts_text("ਕੀਮਤ ੦") == "ਕੀਮਤ 0"
        assert normalize_tts_text("੦੧੨੩੪੫੬੭੮੯") == "0123456789"

    def test_normalize_tts_arabic_indic_digits(self):
        """Arabic-Indic digits should be converted to ASCII."""
        assert normalize_tts_text("٠١٢٣٤٥٦٧٨٩") == "0123456789"

    def test_normalize_tts_extended_arabic_indic_digits(self):
        """Extended Arabic-Indic digits should be converted to ASCII."""
        assert normalize_tts_text("۰۱۲۳۴۵۶۷۸۹") == "0123456789"

    def test_normalize_tts_myanmar_digits(self):
        """Myanmar digits should be converted to ASCII."""
        assert normalize_tts_text("၀၁၂၃၄၅၆၇၈၉") == "0123456789"

    def test_normalize_tts_thai_digits(self):
        """Thai digits should be converted to ASCII."""
        assert normalize_tts_text("๐๑๒๓๔๕๖๗๘๙") == "0123456789"

    def test_normalize_tts_lao_digits(self):
        """Lao digits should be converted to ASCII."""
        assert normalize_tts_text("໐໑໒໓໔໕໖໗໘໙") == "0123456789"

    def test_normalize_tts_tibetan_digits(self):
        """Tibetan digits should be converted to ASCII."""
        assert normalize_tts_text("༠༡༢༣༤༥༦༧༨༩") == "0123456789"

    def test_normalize_tts_khmer_digits(self):
        """Khmer digits should be converted to ASCII."""
        assert normalize_tts_text("០១២៣៤៥៦៧៨៩") == "0123456789"

    def test_normalize_tts_mixed_content(self):
        """Mixed ASCII and non-ASCII digits should be handled correctly."""
        assert normalize_tts_text("Price is ० and 100") == "Price is 0 and 100"
        assert normalize_tts_text("०५० and 123") == "050 and 123"
        assert normalize_tts_text("०१२३ and 456 and 789") == "0123 and 456 and 789"

    def test_normalize_tts_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_tts_text("") == ""

    def test_normalize_tts_no_digits(self):
        """Text without any digits should return unchanged."""
        assert normalize_tts_text("Hello World!") == "Hello World!"
        assert normalize_tts_text("日本語テスト") == "日本語テスト"
        assert normalize_tts_text("Привет мир") == "Привет мир"


class TestSupertonicDigitNormalization:
    """Test that Supertonic synthesis normalizes non-Latin digits."""

    @pytest.mark.asyncio
    async def test_supertonic_normalizes_devanagari_digits(self, tmp_path):
        """Devanagari digits in Hindi text should be normalized for Supertonic."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://piper:5000",
                "timeout_seconds": 10,
                "language_providers": {"hi": "supertonic"},
                "providers": {
                    "supertonic": {
                        "base_url": "http://supertonic-tts:7788",
                        "voice": "M1",
                        "steps": 8,
                        "speed": 1.05,
                        "response_format": "wav",
                    }
                },
            }
            mock_config.return_value.app = {}
            mock_config.return_value.media_output_dir = tmp_path
            service = TTSService()

            response = httpx.Response(
                200,
                content=b"RIFFtest",
                headers={"X-Audio-Duration": "1.0"},
                request=httpx.Request("POST", "http://supertonic-tts:7788/v1/tts"),
            )

            class MockStream:
                async def __aenter__(self):
                    return response

                async def __aexit__(self, exc_type, exc, traceback):
                    return None

            service._http_client.stream = MagicMock(return_value=MockStream())

            result = await service.synthesize("रुपये ०", "hi-IN", tmp_path / "hi.wav")

            assert result.success is True
            call_args = service._http_client.stream.call_args
            assert call_args[1]["json"]["text"] == "रुपये 0"

    @pytest.mark.asyncio
    async def test_supertonic_preserves_meaning_with_numbers(self, tmp_path):
        """Normalized digits should not change the numeric meaning."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://piper:5000",
                "timeout_seconds": 10,
                "language_providers": {"hi": "supertonic"},
                "providers": {
                    "supertonic": {
                        "base_url": "http://supertonic-tts:7788",
                        "voice": "M1",
                        "steps": 8,
                        "speed": 1.05,
                        "response_format": "wav",
                    }
                },
            }
            mock_config.return_value.app = {}
            mock_config.return_value.media_output_dir = tmp_path
            service = TTSService()

            response = httpx.Response(
                200,
                content=b"RIFFtest",
                headers={"X-Audio-Duration": "1.0"},
                request=httpx.Request("POST", "http://supertonic-tts:7788/v1/tts"),
            )

            class MockStream:
                async def __aenter__(self):
                    return response

                async def __aexit__(self, exc_type, exc, traceback):
                    return None

            service._http_client.stream = MagicMock(return_value=MockStream())

            result = await service.synthesize(
                "संख्या ०१२३४५६७८९", "hi-IN", tmp_path / "hi.wav"
            )

            assert result.success is True
            call_args = service._http_client.stream.call_args
            assert call_args[1]["json"]["text"] == "संख्या 0123456789"

    @pytest.mark.asyncio
    async def test_synthesize_with_http_error(self):
        """Test synthesize handles HTTP error."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            with patch.object(
                service._http_client, "get", side_effect=httpx.HTTPError("HTTP error")
            ):
                result = await service.synthesize("Hello", "en")
                assert result.success is False
                assert "HTTP error" in result.error

    @pytest.mark.asyncio
    async def test_synthesize_with_timeout_error(self):
        """Test synthesize handles timeout error."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            with patch.object(
                service._http_client,
                "get",
                side_effect=httpx.TimeoutException("Timeout"),
            ):
                result = await service.synthesize("Hello", "en")
                assert result.success is False
                assert (
                    "timeout" in result.error.lower()
                    or "http error" in result.error.lower()
                )

    @pytest.mark.asyncio
    async def test_synthesize_with_request_error(self):
        """Test synthesize handles request error."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            with patch.object(
                service._http_client,
                "get",
                side_effect=httpx.RequestError("Request error"),
            ):
                result = await service.synthesize("Hello", "en")
                assert result.success is False
                assert "http error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_synthesize_with_empty_response(self):
        """Test synthesize handles empty response."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "voice": "en_US-lessac-medium",
                    "timeout_seconds": 15,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b""

                with patch.object(
                    service._http_client, "post", return_value=mock_response
                ):
                    result = await service.synthesize("Hello", "en")
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_synthesize_with_success(self):
        """Test synthesize success."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "voice": "en_US-lessac-medium",
                    "timeout_seconds": 15,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b"fake audio data"

                with patch.object(
                    service._http_client, "post", return_value=mock_response
                ):
                    result = await service.synthesize("Hello", "en")
                    assert result.success is True
                    assert result.audio_path is not None
                    assert result.audio_url is not None

    @pytest.mark.asyncio
    async def test_fetch_voices_with_cache_expired(self):
        """Test _fetch_voices with cache expired."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()
            service._voices_cache = {"test": "voice"}
            service._voices_cache_timestamp = time.time() - 3700

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"newVoice": "data"}

            with patch.object(service._http_client, "get", return_value=mock_response):
                voices = await service._fetch_voices()
                assert voices == {"newVoice": "data"}

    @pytest.mark.asyncio
    async def test_fetch_voices_with_cache_valid(self):
        """Test _fetch_voices with cache valid."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()
            service._voices_cache = {"cachedVoice": "data"}
            service._voices_cache_timestamp = time.time()

            voices = await service._fetch_voices()
            assert voices == {"cachedVoice": "data"}

    @pytest.mark.asyncio
    async def test_fetch_voices_when_disabled(self):
        """Test _fetch_voices when disabled."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": False,
                "mock_mode": True,
            }
            mock_config.return_value.app = {}

            service = TTSService()
            voices = await service._fetch_voices()
            assert voices == {}

    @pytest.mark.asyncio
    async def test_synthesize_with_invalid_voice(self):
        """Test synthesize with invalid voice."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "invalid_voice",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Invalid voice"

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.synthesize("Hello", "en")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_real_speak_with_request_error(self):
        """Test real speak handles request error."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            with patch.object(
                service._http_client,
                "post",
                side_effect=httpx.RequestError("Network error"),
            ):
                result = await service.synthesize("Hello", "en")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_real_speak_with_empty_response(self):
        """Test real speak handles empty response."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "voice": "en_US-lessac-medium",
                    "timeout_seconds": 15,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b""

                with patch.object(
                    service._http_client, "post", return_value=mock_response
                ):
                    result = await service.synthesize("Hello", "en")
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_real_speak_with_success(self):
        """Test real speak succeeds."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.tts_service.get_config") as mock_config:
                mock_config.return_value.tts = {
                    "enabled": True,
                    "mock_mode": False,
                    "provider": "piper",
                    "base_url": "http://localhost:5000",
                    "voice": "en_US-lessac-medium",
                    "timeout_seconds": 15,
                }
                mock_config.return_value.app = {}
                mock_config.return_value.media_output_dir = Path(tmpdir)

                service = TTSService()

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b"audio_data"

                with patch.object(
                    service._http_client, "post", return_value=mock_response
                ):
                    result = await service.synthesize("Hello", "en")
                    assert result.success is True
                    assert result.audio_path is not None

    @pytest.mark.asyncio
    async def test_fetch_voices_with_http_error(self):
        """Test fetch voices handles HTTP error."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = Exception(
                "500 Internal Server Error"
            )

            with patch.object(service._http_client, "get", return_value=mock_response):
                voices = await service._fetch_voices()
                assert voices == {}

    @pytest.mark.asyncio
    async def test_fetch_voices_with_json_parse_error(self):
        """Test fetch voices handles JSON parse error."""
        import json

        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("test", "doc", 0)

            with patch.object(service._http_client, "get", return_value=mock_response):
                voices = await service._fetch_voices()
                assert voices == {}

    @pytest.mark.asyncio
    async def test_fetch_voices_with_request_error(self):
        """Test fetch voices handles request error."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            with patch.object(
                service._http_client,
                "get",
                side_effect=httpx.RequestError("Network error"),
            ):
                voices = await service._fetch_voices()
                assert voices == {}

    @pytest.mark.asyncio
    async def test_fetch_voices_with_timeout_error(self):
        """Test fetch voices handles timeout error."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            with patch.object(
                service._http_client,
                "get",
                side_effect=httpx.TimeoutException("Timeout"),
            ):
                voices = await service._fetch_voices()
                assert voices == {}

    @pytest.mark.asyncio
    async def test_get_voices_with_cache_expired(self):
        """Test get voices with expired cache."""
        import time

        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()
            service._voices_cache = {"cached": {"voice1": "en"}}
            service._voices_cache_timestamp = time.time() - 3600

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"voices": [{"name": "voice2"}]}

            with patch.object(service._http_client, "get", return_value=mock_response):
                voices = await service._fetch_voices()
                assert "voice2" in str(voices)

    @pytest.mark.asyncio
    async def test_get_voices_with_cache_valid(self):
        """Test get voices with valid cache."""
        import time

        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "en_US-lessac-medium",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()
            service._voices_cache = {"cached": {"voice1": "en"}}
            service._voices_cache_timestamp = time.time()

            voices = await service._fetch_voices()
            assert voices == {"cached": {"voice1": "en"}}

    @pytest.mark.asyncio
    async def test_get_voices_when_disabled(self):
        """Test get voices when TTS is disabled."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": False,
                "mock_mode": True,
            }
            mock_config.return_value.app = {}

            service = TTSService()
            voices = await service._fetch_voices()
            assert voices == {}

    @pytest.mark.asyncio
    async def test_real_speak_with_invalid_voice(self):
        """Test real speak with invalid voice."""
        with patch("app.services.tts_service.get_config") as mock_config:
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "base_url": "http://localhost:5000",
                "voice": "invalid_voice",
                "timeout_seconds": 15,
            }
            mock_config.return_value.app = {}

            service = TTSService()

            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Invalid voice"

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.synthesize("Hello", "en")
                assert result.success is False
