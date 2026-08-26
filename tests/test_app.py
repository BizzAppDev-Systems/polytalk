"""
Comprehensive tests for PolyTalk application.
"""

import asyncio
import importlib.util
import os
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "benchmarks"))


def load_stt_main_module(env: dict[str, str] | None = None):
    """Load STT main module with faster_whisper stubbed for unit tests."""
    module_name = "polytalk_stt_main_for_tests"
    if env:
        env_suffix = "_".join(f"{key}_{value}" for key, value in sorted(env.items()))
        module_name = f"{module_name}_{env_suffix}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = object
    sys.modules.setdefault("faster_whisper", faster_whisper_stub)

    module_path = Path(__file__).parent.parent / "stt" / "app" / "main.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec and spec.loader
    old_env = {}
    if env:
        for key, value in env.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
    try:
        spec.loader.exec_module(module)
    finally:
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
    return module


class TestConfig:
    """Tests for configuration loading."""

    def test_config_singleton(self):
        """Test that config returns singleton instance."""
        from app.config import get_config

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_config_has_whisper_section(self):
        """Test that config has whisper section."""
        from app.config import get_config

        config = get_config()
        assert hasattr(config, "whisper")
        assert isinstance(config.whisper, dict)

    def test_config_has_translation_section(self):
        """Test that config has translation section."""
        from app.config import get_config

        config = get_config()
        assert hasattr(config, "translation")
        assert isinstance(config.translation, dict)

    def test_config_has_tts_section(self):
        """Test that config has tts section."""
        from app.config import get_config

        config = get_config()
        assert hasattr(config, "tts")
        assert isinstance(config.tts, dict)

    def test_config_mock_mode_default_behavior(self):
        """Test that mock_mode can be any boolean value and config loads correctly."""
        from app.config import get_config

        config = get_config()
        # Config should load successfully with any mock_mode value (true/false)
        # This ensures tests work regardless of production configuration
        assert "mock_mode" in config.translation
        assert "mock_mode" in config.tts
        assert "mock_mode" in config.whisper
        # All mock_mode values should be boolean
        assert isinstance(config.translation["mock_mode"], bool)
        assert isinstance(config.tts["mock_mode"], bool)
        assert isinstance(config.whisper["mock_mode"], bool)

    def test_config_has_app_section(self):
        """Test that config has app section."""
        from app.config import get_config

        config = get_config()
        assert hasattr(config, "app")
        assert isinstance(config.app, dict)

    def test_config_media_output_dir(self):
        """Test media output directory configuration."""
        from app.config import get_config

        config = get_config()
        assert hasattr(config, "media_output_dir")
        from pathlib import Path

        assert isinstance(config.media_output_dir, Path)

    def test_config_debug_mode(self):
        """Test debug mode configuration."""
        from app.config import get_config

        config = get_config()
        assert hasattr(config, "debug")
        assert isinstance(config.debug, bool)

    def test_config_host(self):
        """Test host configuration."""
        from app.config import get_config

        config = get_config()
        assert hasattr(config, "host")
        assert isinstance(config.host, str)

    def test_config_port(self):
        """Test port configuration."""
        import os

        from app.config import Config, get_config

        with patch.dict(os.environ, {"APP_PORT": "8000"}):
            Config._instance = None
            config = get_config()
            assert hasattr(config, "port")
            assert isinstance(config.port, int)

    def test_config_env_expansion(self):
        """Test environment variable expansion in config."""
        import os

        from app.config import Config

        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            Config._instance = None
            config = Config()
            result = config._expand_string("${TEST_VAR}")
            assert result == "test_value"

    def test_config_type_conversion(self):
        """Test automatic type conversion in config."""
        import os

        from app.config import Config

        os.environ["APP_PORT"] = "8000"
        os.environ["APP_DEBUG"] = "true"
        os.environ["APP_HOST"] = "0.0.0.0"
        Config._instance = None
        config = Config()
        result_true = config._expand_string("true")
        assert result_true is True
        assert isinstance(result_true, bool)
        result_false = config._expand_string("false")
        assert result_false is False
        assert isinstance(result_false, bool)
        assert config._expand_string("123") == 123
        assert isinstance(config._expand_string("123"), int)
        assert config._expand_string("12.5") == 12.5
        assert isinstance(config._expand_string("12.5"), float)
        assert config._expand_string("hello") == "hello"
        assert isinstance(config._expand_string("hello"), str)
        assert config._expand_string("${UNSET_VAR}") == "${UNSET_VAR}"


class TestServices:
    """Tests for service classes."""

    def test_whisper_service_init(self):
        """Test Whisper service initialization."""
        from app.services.whisper_service import WhisperService

        service = WhisperService()
        assert isinstance(service.mock_mode, bool)

    def test_whisper_service_config(self):
        """Test Whisper service configuration."""
        from app.services.whisper_service import WhisperService

        service = WhisperService()
        assert service.enabled is True
        assert service.model == "whisper-1"
        assert service.timeout == 60

    @pytest.mark.asyncio
    async def test_whisper_transcribe_with_mock(self):
        """Test Whisper transcription with mock mode enabled."""
        from app.services.whisper_service import WhisperService

        service = WhisperService()
        service.mock_mode = True
        service.enabled = True

        audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1e\x00\x00\x40\x1e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        result = await anext(service.stream_transcribe(audio_bytes, language="en"))
        assert result.success is True
        assert result.text != ""
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_whisper_transcribe_different_languages_mock(self):
        """Test Whisper transcription with different languages in mock mode."""
        from app.services.whisper_service import WhisperService

        service = WhisperService()
        service.mock_mode = True
        service.enabled = True

        audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1e\x00\x00\x40\x1e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        for lang in ["en", "gu", "hi", "es", "fr", "de"]:
            result = await anext(service.stream_transcribe(audio_bytes, language=lang))
            assert result.success is True
            assert result.language == lang

    @pytest.mark.asyncio
    async def test_whisper_transcribe_unknown_language_mock(self):
        """Test Whisper transcription with unknown language in mock mode."""
        from app.services.whisper_service import WhisperService

        service = WhisperService()
        service.mock_mode = True
        service.enabled = True

        audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1e\x00\x00\x40\x1e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        result = await anext(service.stream_transcribe(audio_bytes, language="unknown"))
        assert result.success is True
        assert result.language == "unknown"

    @pytest.mark.asyncio
    async def test_whisper_stream_transcribe_mock_mode(self):
        """Test Whisper streaming transcription in mock mode."""
        from app.services.whisper_service import WhisperService

        service = WhisperService()
        service.mock_mode = True
        service.enabled = True

        async def audio_generator():
            yield b"chunk1"
            yield b"chunk2"
            yield b"__END_SIGNAL__"

        results = []
        async for result in service.stream_transcribe(audio_generator(), language="en"):
            results.append(result)

        assert len(results) > 0
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_whisper_stream_closes_stt_when_audio_generator_ends_without_signal(
        self,
    ):
        """Test upstream disconnect closes the downstream STT WebSocket."""
        from app.services.whisper_service import WhisperService

        class FakeWebSocket:
            def __init__(self):
                self.sent = []
                self.close_calls = 0
                self._closed_event = asyncio.Event()

            async def send(self, message):
                self.sent.append(message)

            async def recv(self):
                await self._closed_event.wait()
                raise RuntimeError("closed")

            async def close(self):
                self.close_calls += 1
                self._closed_event.set()

        fake_ws = FakeWebSocket()

        async def fake_connect(*args, **kwargs):
            return fake_ws

        async def audio_generator():
            yield b"chunk1"

        service = WhisperService()
        service.enabled = True
        service.mock_mode = False
        service.max_reconnect_attempts = 1

        async def consume_stream():
            results = []
            async for result in service.stream_transcribe(
                audio_generator(),
                language="en",
            ):
                results.append(result)
            return results

        with patch("app.services.whisper_service.websockets.connect", fake_connect):
            results = await asyncio.wait_for(consume_stream(), timeout=2.0)

        assert results == []
        assert b"chunk1" in fake_ws.sent
        assert fake_ws.close_calls >= 1

    def test_translation_service_init(self):
        """Test Translation service initialization."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        assert isinstance(service.mock_mode, bool)

    def test_translation_config_value_falls_back_for_missing_or_unexpanded(self):
        """Test missing and unresolved env config values use safe defaults."""
        from app.services.translation_service import _config_value

        assert _config_value(None, "openai_chat") == "openai_chat"
        assert (
            _config_value("${TRANSLATION_API_FORMAT}", "openai_chat") == "openai_chat"
        )
        assert _config_value("openai_responses", "openai_chat") == "openai_responses"

    def test_translation_prompt_uses_language_names(self):
        """Test translation prompts use readable language names, not raw codes."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.system_prompt_template = (
            "Translate from {source_language} ({source_language_code}) "
            "to {target_language} ({target_language_code})."
        )

        prompt = service._build_system_prompt("de", "gu")

        assert "German (de)" in prompt
        assert "Gujarati (gu)" in prompt

    def test_translation_builds_openai_chat_request(self):
        """Test OpenAI Chat Completions request construction."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"
        service.base_url = "https://api.openai.com"
        service.endpoint = "/v1/chat/completions"
        service.api_key = "test-key"
        service.model = "gpt-4o-mini"
        service.temperature = 0.0
        service.max_tokens = 123
        service.system_prompt_template = (
            "Translate {source_language} to {target_language}."
        )

        url, headers, payload = service._build_translation_request("Hello", "en", "gu")

        assert url == "https://api.openai.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert payload["max_tokens"] == 123
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1] == {"role": "user", "content": "Hello"}

    def test_translation_builds_contextual_openai_chat_request(self):
        """Test context is included only as read-only translation history."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"
        service.base_url = "https://api.openai.com"
        service.endpoint = "/v1/chat/completions"
        service.api_key = "test-key"
        service.model = "gpt-4o-mini"
        service.temperature = 0.0
        service.max_tokens = 123
        service.context_enabled = True
        service.system_prompt_template = (
            "Translate {source_language} to {target_language}."
        )

        _, _, payload = service._build_translation_request(
            "It is ready.",
            "en",
            "gu",
            context=[{"source": "The order arrived.", "target": "ઓર્ડર આવી ગયો."}],
        )

        system_prompt = payload["messages"][0]["content"]
        user_text = payload["messages"][1]["content"]

        assert "previous conversation context" in user_text.lower()
        assert "The order arrived." in user_text
        assert "ઓર્ડર આવી ગયો." in user_text
        assert "It is ready." in user_text
        assert "Translate only the current text" in system_prompt

    def test_translation_visual_context_is_prompted_as_hint_only(self):
        """Test shared tab visual context is included as a non-translated hint."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"
        service.base_url = "https://api.openai.com"
        service.endpoint = "/v1/chat/completions"
        service.api_key = "test-key"
        service.model = "gpt-4o-mini"
        service.temperature = 0.0
        service.max_tokens = 123
        service.context_enabled = True
        service.system_prompt_template = (
            "Translate {source_language} to {target_language}."
        )

        _, _, payload = service._build_translation_request(
            "Move it to stock.",
            "en",
            "de",
            visual_context="The shared tab shows an Odoo Inventory Overview page.",
        )

        system_prompt = payload["messages"][0]["content"]
        user_text = payload["messages"][1]["content"]

        assert "shared visual context" in system_prompt.lower()
        assert "spoken source text wins" in system_prompt
        assert "Shared tab/page visual context hint" in user_text
        assert "Odoo Inventory Overview" in user_text
        assert "Move it to stock." in user_text

    def test_translation_custom_instruction_is_prompted_and_bounded(self):
        """Test user translation guidance is added to the system prompt only."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"
        service.base_url = "https://api.openai.com"
        service.endpoint = "/v1/chat/completions"
        service.api_key = "test-key"
        service.model = "gpt-4o-mini"
        service.temperature = 0.0
        service.max_tokens = 123
        service.context_enabled = True
        service.system_prompt_template = (
            "Translate {source_language} to {target_language}."
        )
        instruction = "Keep   product names in English.\n" + ("x" * 260)

        _, _, payload = service._build_translation_request(
            "Move it to stock.",
            "en",
            "de",
            custom_instruction=instruction,
        )

        system_prompt = payload["messages"][0]["content"]
        user_text = payload["messages"][1]["content"]

        assert "User translation instruction (high priority)" in system_prompt
        assert "speaker gender, persona, or voice" in system_prompt
        assert "target-language grammar and verb forms" in system_prompt
        assert "Keep product names in English." in system_prompt
        assert (
            len(service._sanitize_custom_instruction(instruction))
            == service.custom_instruction_max_chars
        )
        assert instruction not in user_text

    def test_translation_warns_for_large_context_payload(self, caplog):
        """Test large contextual payloads log sizes without prompt contents."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"
        service.base_url = "https://api.openai.com"
        service.endpoint = "/v1/chat/completions"
        service.api_key = "test-key"
        service.model = "gpt-4o-mini"
        service.temperature = 0.0
        service.max_tokens = 123
        service.context_enabled = True
        service.context_payload_warn_chars = 10
        service.system_prompt_template = (
            "Translate {source_language} to {target_language}."
        )

        with caplog.at_level("WARNING"):
            service._build_translation_request(
                "It is ready.",
                "en",
                "gu",
                context=[
                    {
                        "source": "The order arrived.",
                        "target": "ઓર્ડર આવી ગયો.",
                    }
                ],
            )

        assert "Translation prompt payload is large" in caplog.text
        assert "system_chars=" in caplog.text
        assert "user_chars=" in caplog.text
        assert "The order arrived." not in caplog.text

    def test_translation_omits_context_when_disabled(self):
        """Test disabled context preserves the plain provider payload."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"
        service.base_url = "https://api.openai.com"
        service.endpoint = "/v1/chat/completions"
        service.api_key = "test-key"
        service.model = "gpt-4o-mini"
        service.temperature = 0.0
        service.max_tokens = 123
        service.context_enabled = False
        service.system_prompt_template = (
            "Translate {source_language} to {target_language}."
        )

        _, _, payload = service._build_translation_request(
            "It is ready.",
            "en",
            "gu",
            context=[{"source": "The order arrived.", "target": "ઓર્ડર આવી ગયો."}],
        )

        assert payload["messages"][1] == {
            "role": "user",
            "content": "It is ready.",
        }
        assert "previous context" not in payload["messages"][0]["content"].lower()

    def test_translation_builds_openai_responses_request(self):
        """Test OpenAI Responses request construction."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_responses"
        service.base_url = "https://api.openai.com"
        service.endpoint = "/v1/responses"
        service.api_key = "test-key"
        service.model = "gpt-4o-mini"
        service.temperature = 0.0
        service.max_tokens = 123
        service.system_prompt_template = (
            "Translate {source_language} to {target_language}."
        )

        url, headers, payload = service._build_translation_request("Hello", "en", "gu")

        assert url == "https://api.openai.com/v1/responses"
        assert headers["Authorization"] == "Bearer test-key"
        assert payload["input"] == "Hello"
        assert payload["max_output_tokens"] == 123
        assert "English" in payload["instructions"]
        assert "Gujarati" in payload["instructions"]

    def test_translation_builds_anthropic_messages_request(self):
        """Test Anthropic Messages request construction."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.config = {"anthropic_version": "2023-06-01"}
        service.api_format = "anthropic_messages"
        service.base_url = "https://api.anthropic.com"
        service.endpoint = "/v1/messages"
        service.api_key = "test-key"
        service.model = "claude-3-5-haiku-latest"
        service.temperature = 0.0
        service.max_tokens = 123
        service.system_prompt_template = (
            "Translate {source_language} to {target_language}."
        )

        url, headers, payload = service._build_translation_request("Hello", "en", "gu")

        assert url == "https://api.anthropic.com/v1/messages"
        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert payload["max_tokens"] == 123
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert "English" in payload["system"]

    def test_translation_builds_gemini_generate_content_request(self):
        """Test Gemini generateContent request construction."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "gemini_generate_content"
        service.base_url = "https://generativelanguage.googleapis.com/v1beta"
        service.endpoint = "/models/{model}:generateContent"
        service.api_key = "test-key"
        service.model = "gemini-1.5-flash"
        service.temperature = 0.0
        service.max_tokens = 123
        service.system_prompt_template = (
            "Translate {source_language} to {target_language}."
        )

        url, headers, payload = service._build_translation_request("Hello", "en", "gu")

        assert url.endswith("/models/gemini-1.5-flash:generateContent")
        assert headers["x-goog-api-key"] == "test-key"
        assert payload["contents"][0]["parts"][0]["text"] == "Hello"
        assert payload["generationConfig"]["maxOutputTokens"] == 123
        assert "English" in payload["systemInstruction"]["parts"][0]["text"]

    def test_translation_parses_supported_api_responses(self):
        """Test response text extraction for supported API formats."""
        from app.services.translation_service import TranslationService

        service = TranslationService()

        service.api_format = "openai_chat"
        text, quality = service._parse_translation_response(
            {
                "choices": [{"message": {"content": "નમસ્તે"}}],
            }
        )
        assert text == "નમસ્તે"
        assert quality == 0.5

        service.api_format = "openai_responses"
        text, quality = service._parse_translation_response({"output_text": "નમસ્તે"})
        assert text == "નમસ્તે"
        assert quality == 0.5

        service.api_format = "anthropic_messages"
        text, quality = service._parse_translation_response(
            {
                "content": [{"type": "text", "text": "નમસ્તે"}],
            }
        )
        assert text == "નમસ્તે"
        assert quality == 0.5

        service.api_format = "gemini_generate_content"
        text, quality = service._parse_translation_response(
            {
                "candidates": [{"content": {"parts": [{"text": "નમસ્તે"}]}}],
            }
        )
        assert text == "નમસ્તે"
        assert quality == 0.5

    def test_translation_confidence_parsing_with_separator(self):
        """Test confidence extraction when ###TRANSLATION_END### separator is present."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"

        # Happy path: separator with valid JSON
        text, quality = service._parse_translation_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'નમસ્તે###TRANSLATION_END###{"translation_quality": 0.85}'
                        }
                    }
                ],
            }
        )
        assert text == "નમસ્તે"
        assert quality == 0.85

        # Valid JSON with different quality value
        text, quality = service._parse_translation_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'Hello world###TRANSLATION_END###{"translation_quality": 1.0}'
                        }
                    }
                ],
            }
        )
        assert text == "Hello world"
        assert quality == 1.0

        # Invalid JSON after separator: returns text with None quality
        text, quality = service._parse_translation_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Hello world###TRANSLATION_END###not valid json"
                        }
                    }
                ],
            }
        )
        assert text == "Hello world"
        assert quality == 0.5

    def test_translation_confidence_fallback_json_on_own_line(self):
        """Test fallback confidence extraction when separator is omitted but JSON is on its own line."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"

        # JSON on its own last line (separator omitted)
        text, quality = service._parse_translation_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'Hello world\n{"translation_quality": 0.75}'
                        }
                    }
                ],
            }
        )
        assert text == "Hello world"
        assert quality == 0.75

        # Multi-line translation with JSON fallback
        text, quality = service._parse_translation_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'First sentence\nSecond sentence\n{"translation_quality": 0.9}'
                        }
                    }
                ],
            }
        )
        assert text == "First sentence\nSecond sentence"
        assert quality == 0.9

        # JSON embedded in text (not on its own line): should NOT strip
        text, quality = service._parse_translation_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'See { "translation_quality": 0.5 } for details.'
                        }
                    }
                ],
            }
        )
        assert text == 'See { "translation_quality": 0.5 } for details.'
        assert quality == 0.5

        # JSON not on own line (mid-sentence): should NOT strip
        text, quality = service._parse_translation_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'Note: {"translation_quality": 0.5} is important.'
                        }
                    }
                ],
            }
        )
        assert text == 'Note: {"translation_quality": 0.5} is important.'
        assert quality == 0.5

    def test_translation_confidence_no_data_returns_default(self):
        """Test that missing confidence data returns 0.5 (50%%) as default."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"

        text, quality = service._parse_translation_response(
            {
                "choices": [{"message": {"content": "Hello world"}}],
            }
        )
        assert text == "Hello world"
        assert quality == 0.5

    def test_translation_allows_openai_responses_without_api_key(self):
        """Test OpenAI-compatible responses can run without auth."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_responses"
        service.base_url = "http://localhost:8000"
        service.endpoint = "/v1/responses"
        service.api_key = ""

        _, headers, _ = service._build_translation_request("Hello", "en", "gu")

        assert "Authorization" not in headers

    def test_translation_allows_openai_chat_without_api_key(self):
        """Test OpenAI-compatible self-hosted chat can run without auth."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"
        service.base_url = "http://localhost:11434"
        service.endpoint = "/v1/chat/completions"
        service.api_key = ""

        _, headers, _ = service._build_translation_request("Hello", "en", "gu")

        assert "Authorization" not in headers

    def test_translation_parses_empty_gemini_edge_cases_as_empty_text(self):
        """Test Gemini edge cases raise ValueError for empty responses."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "gemini_generate_content"

        for payload in [
            {"candidates": []},
            {"candidates": [{}]},
            {"candidates": [{"content": {}}]},
        ]:
            with pytest.raises(
                ValueError, match="Empty gemini_generate_content translation response"
            ):
                service._parse_translation_response(payload)

    def test_translation_rejects_unknown_api_format(self):
        """Test unsupported API formats fail before making a request."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "unknown"

        with pytest.raises(ValueError, match="Unsupported translation api_format"):
            service._build_translation_request("Hello", "en", "gu")

    def test_translation_rejects_malformed_response(self):
        """Test malformed provider responses raise clear parsing errors."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.api_format = "openai_chat"

        with pytest.raises(
            ValueError, match="Invalid openai_chat translation response"
        ):
            service._parse_translation_response({"choices": []})

    def test_translation_service_config(self):
        """Test Translation service configuration."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        assert service.enabled is True
        assert service.temperature == 0.0
        # max_tokens comes from config, verify it's set to a reasonable default
        assert isinstance(service.max_tokens, int)
        assert service.max_tokens > 0

    @pytest.mark.asyncio
    async def test_translation_translate_mock(self):
        """Test translation in mock mode."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.mock_mode = True
        service.enabled = True

        result = await service.translate(
            "Hello, how are you?", source_language="en", target_language="gu"
        )
        assert result.success is True
        assert result.text != ""
        assert result.source_language == "en"
        assert result.target_language == "gu"

    @pytest.mark.asyncio
    async def test_translation_translate_disabled(self):
        """Test translation when service is disabled."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.enabled = False
        service.mock_mode = False

        result = await service.translate(
            "test", source_language="en", target_language="gu"
        )
        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_translation_mock_all_combinations(self):
        """Test mock translation for all language combinations."""
        from app.services.translation_service import TranslationService

        service = TranslationService()
        service.mock_mode = True
        service.enabled = True

        test_cases = [
            ("en", "gu"),
            ("en", "hi"),
            ("en", "es"),
            ("gu", "en"),
            ("hi", "en"),
        ]

        for source, target in test_cases:
            result = await service.translate(
                "Test text", source_language=source, target_language=target
            )
            assert result.success is True
            assert result.text != ""

    def test_tts_service_init(self):
        """Test TTS service initialization."""
        from app.services.tts_service import TTSService

        service = TTSService()
        assert isinstance(service.mock_mode, bool)

    def test_tts_regional_default_voices_take_precedence(self):
        """Test regional language codes use exact configured voice defaults."""
        from app.services.tts_service import TTSService

        service = TTSService()

        async def fake_fetch_voices():
            return {
                "es_ES-davefx-medium": {},
                "es_MX-claude-high": {},
                "nl_NL-ronnie-medium": {},
                "nl_BE-nathalie-medium": {},
            }

        service._fetch_voices = fake_fetch_voices

        assert (
            asyncio.run(service._get_voice_for_language("es_MX")) == "es_MX-claude-high"
        )
        assert (
            asyncio.run(service._get_voice_for_language("nl_BE"))
            == "nl_BE-nathalie-medium"
        )

    def test_tts_regional_defaults_fall_back_when_unavailable(self):
        """Test missing regional voices fall back to available base language voices."""
        from app.services.tts_service import TTSService

        service = TTSService()

        async def fake_fetch_voices():
            return {
                "es_ES-davefx-medium": {},
                "nl_NL-ronnie-medium": {},
            }

        service._fetch_voices = fake_fetch_voices

        assert (
            asyncio.run(service._get_voice_for_language("es_MX"))
            == "es_ES-davefx-medium"
        )
        assert (
            asyncio.run(service._get_voice_for_language("nl_BE"))
            == "nl_NL-ronnie-medium"
        )

    def test_tts_language_length_scale_defaults(self):
        """Test Piper length scale can be tuned per language."""
        from app.services.tts_service import TTSService

        service = TTSService()
        service.length_scales = {
            "default": 0.9,
            "gu": 0.75,
            "gu_IN-aikosh_female-medium": 0.8,
        }

        assert (
            service._get_length_scale_for_language("gu", "gu_IN-aikosh_female-medium")
            == 0.8
        )
        assert service._get_length_scale_for_language("gu_IN", "other-voice") == 0.75
        assert service._get_length_scale_for_language("en", "en_GB-jenny") == 0.9

    def test_whisper_normalizes_regional_language_hints(self):
        """Test regional UI language codes are normalized for ASR hints."""
        from app.services.whisper_service import WhisperService

        service = WhisperService()

        assert service._normalize_language_hint("es_MX") == "es"
        assert service._normalize_language_hint("nl-BE") == "nl"
        assert service._normalize_language_hint(None) is None

    def test_pipeline_preserves_regional_source_for_translation(self):
        """Test ASR can use base hints without dropping regional translation source."""
        from app.services.base import TranscriptionResult, TranslationResult, TTSResult
        from app.services.pipeline_service import TranslationPipelineService

        class FakeWhisperService:
            mock_mode = True

            def __init__(self):
                self.language_hints = []

            async def stream_transcribe(
                self, audio_generator, language=None, **_kwargs
            ):
                self.language_hints.append(language)
                yield TranscriptionResult(
                    text="hallo allemaal.",
                    language="nl",
                    success=True,
                    is_partial=False,
                )

            async def close(self):
                return None

        class FakeTranslationService:
            mock_mode = True

            def __init__(self):
                self.calls = []

            async def translate(
                self,
                text,
                source_language,
                target_language,
                context=None,
                visual_context=None,
                custom_instruction=None,
            ):
                self.calls.append((text, source_language, target_language))
                return TranslationResult(
                    text="hello everyone.",
                    source_language=source_language,
                    target_language=target_language,
                    success=True,
                )

            async def close(self):
                return None

        class FakeTTSService:
            mock_mode = True

            async def synthesize(self, text, language, output_path=None):
                return TTSResult(audio_url="/fake.wav", success=True)

            async def close(self):
                return None

        async def audio_generator():
            yield b"audio"

        async def run_pipeline():
            whisper = FakeWhisperService()
            translation = FakeTranslationService()
            pipeline = TranslationPipelineService(
                whisper_service=whisper,
                translation_service=translation,
                tts_service=FakeTTSService(),
                warm_connections=False,
            )
            async for _ in pipeline.process_streaming(
                audio_generator(),
                source_language="nl_BE",
                target_language="en",
                save_media=False,
            ):
                pass
            return whisper.language_hints, translation.calls

        language_hints, translation_calls = asyncio.run(run_pipeline())

        assert language_hints == ["nl_BE"]
        assert translation_calls
        assert translation_calls[0][1:] == ("nl_BE", "en")

    def test_pipeline_flushes_short_delta_on_stt_pause(self):
        """Test final short pause-flushed fragments are translated."""
        from app.services.base import TranscriptionResult, TranslationResult, TTSResult
        from app.services.pipeline_service import TranslationPipelineService

        class FakeWhisperService:
            mock_mode = True

            async def stream_transcribe(
                self, audio_generator, language=None, **_kwargs
            ):
                yield TranscriptionResult(
                    text="Working on",
                    language="en",
                    success=True,
                    is_partial=True,
                    metrics={"force_emit": True},
                )

            async def close(self):
                return None

        class FakeTranslationService:
            mock_mode = True

            def __init__(self):
                self.calls = []

            async def translate(
                self,
                text,
                source_language,
                target_language,
                context=None,
                visual_context=None,
                custom_instruction=None,
            ):
                self.calls.append((text, source_language, target_language))
                return TranslationResult(
                    text="Werken aan",
                    source_language=source_language,
                    target_language=target_language,
                    success=True,
                )

            async def close(self):
                return None

        class FakeTTSService:
            mock_mode = True

            async def synthesize(self, text, language, output_path=None):
                return TTSResult(audio_url="/fake.wav", success=True)

            async def close(self):
                return None

        async def audio_generator():
            yield b"audio"

        async def run_pipeline():
            translation = FakeTranslationService()
            pipeline = TranslationPipelineService(
                whisper_service=FakeWhisperService(),
                translation_service=translation,
                tts_service=FakeTTSService(),
                warm_connections=False,
            )
            async for _ in pipeline.process_streaming(
                audio_generator(),
                source_language="en",
                target_language="nl",
                save_media=False,
            ):
                pass
            return translation.calls

        translation_calls = asyncio.run(run_pipeline())

        assert translation_calls == [("Working on", "en", "nl")]

    def test_pipeline_passes_prior_translation_context(self):
        """Test later translation chunks receive bounded prior context."""
        from app.services.base import TranscriptionResult, TranslationResult, TTSResult
        from app.services.pipeline_service import TranslationPipelineService

        class FakeWhisperService:
            mock_mode = True

            async def stream_transcribe(
                self, audio_generator, language=None, **_kwargs
            ):
                yield TranscriptionResult(
                    text="The order arrived.",
                    language="en",
                    success=True,
                    is_partial=True,
                )
                yield TranscriptionResult(
                    text="The order arrived. It is ready.",
                    language="en",
                    success=True,
                    is_partial=False,
                )

            async def close(self):
                return None

        class FakeTranslationService:
            mock_mode = True

            def __init__(self):
                self.calls = []

            async def translate(
                self,
                text,
                source_language,
                target_language,
                context=None,
                visual_context=None,
                custom_instruction=None,
            ):
                copied_context = [dict(item) for item in context] if context else None
                self.calls.append(
                    (
                        text,
                        source_language,
                        target_language,
                        copied_context,
                        visual_context,
                    )
                )
                translated = {
                    "The order arrived.": "ઓર્ડર આવી ગયો.",
                    "It is ready.": "તે તૈયાર છે.",
                }[text]
                return TranslationResult(
                    text=translated,
                    source_language=source_language,
                    target_language=target_language,
                    success=True,
                )

            async def close(self):
                return None

        class FakeTTSService:
            mock_mode = True

            async def synthesize(self, text, language, output_path=None):
                return TTSResult(audio_url="/fake.wav", success=True)

            async def close(self):
                return None

        async def audio_generator():
            yield b"audio"

        async def run_pipeline():
            translation = FakeTranslationService()
            pipeline = TranslationPipelineService(
                whisper_service=FakeWhisperService(),
                translation_service=translation,
                tts_service=FakeTTSService(),
                warm_connections=False,
            )
            async for _ in pipeline.process_streaming(
                audio_generator(),
                source_language="en",
                target_language="gu",
                save_media=False,
            ):
                pass
            return translation.calls

        translation_calls = asyncio.run(run_pipeline())

        assert translation_calls[0] == ("The order arrived.", "en", "gu", None, None)
        assert translation_calls[1] == (
            "It is ready.",
            "en",
            "gu",
            [{"source": "The order arrived.", "target": "ઓર્ડર આવી ગયો."}],
            None,
        )

    def test_pipeline_passes_visual_context_to_translation(self):
        """Test visual context summaries are passed to translation calls."""
        import asyncio

        from app.services.base import TranscriptionResult, TranslationResult, TTSResult
        from app.services.pipeline_service import TranslationPipelineService

        class FakeWhisperService:
            mock_mode = True

            async def stream_transcribe(
                self, audio_generator, language=None, **_kwargs
            ):
                async for _ in audio_generator:
                    break
                yield TranscriptionResult(
                    text="Move it to stock.",
                    language="en",
                    success=True,
                    is_partial=False,
                )

            async def close(self):
                return None

        class FakeTranslationService:
            mock_mode = True

            def __init__(self):
                self.calls = []

            async def translate(
                self,
                text,
                source_language,
                target_language,
                context=None,
                visual_context=None,
                custom_instruction=None,
            ):
                self.calls.append(visual_context)
                return TranslationResult(
                    text="Ins Lager verschieben.",
                    source_language=source_language,
                    target_language=target_language,
                    success=True,
                )

            async def close(self):
                return None

        class FakeTTSService:
            mock_mode = True

            async def synthesize(self, text, language, output_path=None):
                return TTSResult(audio_url="/fake.wav", success=True)

            async def close(self):
                return None

        async def audio_generator():
            yield b"audio"

        async def run_pipeline():
            visual_context_queue = asyncio.Queue()
            visual_context_queue.put_nowait("The shared tab shows Odoo Inventory.")
            translation = FakeTranslationService()
            pipeline = TranslationPipelineService(
                whisper_service=FakeWhisperService(),
                translation_service=translation,
                tts_service=FakeTTSService(),
                warm_connections=False,
            )
            async for _ in pipeline.process_streaming(
                audio_generator(),
                source_language="en",
                target_language="de",
                save_media=False,
                visual_context_queue=visual_context_queue,
            ):
                pass
            return translation.calls

        translation_calls = asyncio.run(run_pipeline())

        assert translation_calls == ["The shared tab shows Odoo Inventory."]

    def test_pipeline_extracts_simple_cumulative_delta(self):
        """Test cumulative STT growth returns only newly appended text."""
        from app.services.pipeline_service import TranslationPipelineService

        delta = TranslationPipelineService._extract_new_transcript_text(
            "eins zwei drei vier",
            "eins zwei",
        )

        assert delta == "drei vier"

    def test_pipeline_extracts_delta_after_asr_correction_overlap(self):
        """Test ASR rewrites use suffix overlap instead of full transcript."""
        from app.services.pipeline_service import TranslationPipelineService

        delta = TranslationPipelineService._extract_new_transcript_text(
            "eins zwei korrigiert drei vier funf sechs",
            "eins zwei drei vier",
        )

        assert delta == "funf sechs"

    def test_pipeline_skips_ambiguous_rewrite_delta(self):
        """Test ambiguous ASR rewrites are not treated as all-new speech."""
        from app.services.pipeline_service import TranslationPipelineService

        delta = TranslationPipelineService._extract_new_transcript_text(
            "komplett anderer text ohne klare uberlappung",
            "eins zwei drei vier",
        )

        assert delta == ""

    def test_pipeline_skips_shorter_correction_delta(self):
        """Test shorter ASR corrections do not retrigger translation/TTS."""
        from app.services.pipeline_service import TranslationPipelineService

        delta = TranslationPipelineService._extract_new_transcript_text(
            "eins zwei",
            "eins zwei drei vier",
        )

        assert delta == ""

    @pytest.mark.asyncio
    async def test_tts_synthesize_mock_mode(self):
        """Test TTS synthesis in mock mode."""
        from pathlib import Path

        from app.services.tts_service import TTSService

        service = TTSService()
        service.mock_mode = True
        service.enabled = True
        service.media_dir = Path("/tmp")

        result = await service.synthesize("Hello world", language="en")
        assert result.success is True
        assert result.audio_path is not None
        assert result.audio_url is not None

    @pytest.mark.asyncio
    async def test_tts_synthesize_with_custom_path_mock(self):
        """Test TTS synthesis with custom output path in mock mode."""
        from pathlib import Path

        from app.services.tts_service import TTSService

        service = TTSService()
        service.mock_mode = True
        service.enabled = True

        custom_path = Path("/tmp/test_tts.wav")

        result = await service.synthesize(
            "Hello", language="en", output_path=custom_path
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_tts_synthesize_disabled(self):
        """Test TTS when service is disabled."""
        from app.services.tts_service import TTSService

        service = TTSService()
        service.enabled = False
        service.mock_mode = False

        result = await service.synthesize("Hello", language="en")
        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_tts_voice_selection(self):
        """Test TTS voice selection for different languages."""
        from app.services.tts_service import TTSService

        service = TTSService()

        for lang in ["en", "gu", "hi"]:
            voice = await service._get_voice_for_language(lang)
            assert voice is not None
            assert isinstance(voice, str)


class TestMainApp:
    """Tests for main FastAPI application."""

    def test_create_app(self):
        """Test creating FastAPI application."""
        from app.main import create_app
        from app.version import __version__

        app = create_app()
        assert app is not None
        assert app.title == "PolyTalk"
        assert app.version == __version__

    def test_app_has_routers(self):
        """Test that app includes routers."""
        from app.main import create_app

        create_app()


class TestSTTCadence:
    """Tests for STT stream cadence and overlap helpers."""

    def test_stt_pause_flush_seconds_default(self):
        """Test STT pause flushing defaults to a short speech-boundary delay."""
        stt_main = load_stt_main_module()

        assert stt_main.PAUSE_FLUSH_SECONDS == 1.0

    def test_stt_pause_flush_seconds_env_override(self):
        """Test STT pause flushing is configurable."""
        stt_main = load_stt_main_module({"STT_PAUSE_FLUSH_SECONDS": "1.75"})

        assert stt_main.PAUSE_FLUSH_SECONDS == 1.75

    def test_stt_leading_silence_preroll_default(self):
        """Test leading silence keeps only a short default pre-roll."""
        stt_main = load_stt_main_module()

        assert stt_main.LEADING_SILENCE_PREROLL_SECONDS == 0.2

    def test_stt_leading_silence_preroll_env_override(self):
        """Test leading silence pre-roll is configurable."""
        stt_main = load_stt_main_module({"STT_LEADING_SILENCE_PREROLL_SECONDS": "0.35"})

        assert stt_main.LEADING_SILENCE_PREROLL_SECONDS == 0.35

    def test_stt_leading_silence_preroll_keeps_recent_audio_only(self):
        """Test startup silence is bounded before first detected speech."""
        stt_main = load_stt_main_module()
        chunks = []

        total_bytes = 0
        total_bytes = stt_main._append_bounded_audio_preroll(
            chunks, b"a" * 4, max_bytes=10, current_total_bytes=total_bytes
        )
        total_bytes = stt_main._append_bounded_audio_preroll(
            chunks, b"b" * 4, max_bytes=10, current_total_bytes=total_bytes
        )
        total_bytes = stt_main._append_bounded_audio_preroll(
            chunks, b"c" * 4, max_bytes=10, current_total_bytes=total_bytes
        )

        assert b"".join(chunks) == b"aabbbbcccc"
        assert total_bytes == 10

    def test_stt_leading_silence_preroll_can_be_disabled(self):
        """Test zero pre-roll discards all leading silence."""
        stt_main = load_stt_main_module()
        chunks = [b"silence"]

        total_bytes = stt_main._append_bounded_audio_preroll(
            chunks, b"more", max_bytes=0, current_total_bytes=len(b"silence")
        )

        assert chunks == []
        assert total_bytes == 0

    def test_stt_stream_prepends_leading_silence_preroll_before_first_voice(self):
        """Test stream transcription prepends bounded startup silence to speech."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "0.3",
                "STT_LEADING_SILENCE_PREROLL_SECONDS": "0.2",
                "STT_EMIT_MIN_CHARS": "1",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        captured_audio = []
        silence = b"\x00\x00" * 2
        voice = (10000).to_bytes(2, "little", signed=True) * 2

        def fake_process_transcribe_job(model, job):
            captured_audio.append(job.audio_bytes)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="hello",
                has_speech=True,
                detected_language="en",
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_bytes(silence)
                websocket.send_bytes(voice)
                assert websocket.receive_json()["text"] == "hello"

        assert captured_audio == [silence + voice]

    def test_stt_stream_prepends_partial_leading_silence_preroll(self):
        """Test speech receives partial preroll when voice starts before it fills."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "0.4",
                "STT_LEADING_SILENCE_PREROLL_SECONDS": "0.4",
                "STT_EMIT_MIN_CHARS": "1",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        captured_audio = []
        silence = b"\x00\x00" * 1
        voice = (10000).to_bytes(2, "little", signed=True) * 3

        def fake_process_transcribe_job(model, job):
            captured_audio.append(job.audio_bytes)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="hello",
                has_speech=True,
                detected_language="en",
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_bytes(silence)
                websocket.send_bytes(voice)
                assert websocket.receive_json()["text"] == "hello"

        assert captured_audio == [silence + voice]

    def test_stt_stream_skips_preroll_when_voice_starts_immediately(self):
        """Test stream transcription does not prepend silence without startup silence."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "0.2",
                "STT_LEADING_SILENCE_PREROLL_SECONDS": "0.2",
                "STT_EMIT_MIN_CHARS": "1",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        captured_audio = []
        voice = (10000).to_bytes(2, "little", signed=True) * 2

        def fake_process_transcribe_job(model, job):
            captured_audio.append(job.audio_bytes)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="hello",
                has_speech=True,
                detected_language="en",
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_bytes(voice)
                assert websocket.receive_json()["text"] == "hello"

        assert captured_audio == [voice]

    def test_stt_stream_pause_flush_still_trims_trailing_silence(self):
        """Test leading-silence preroll does not break pause-flush trimming."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "10",
                "STT_PAUSE_FLUSH_SECONDS": "0.2",
                "STT_VAD_SPEECH_PAD_MS": "0",
                "STT_EMIT_MIN_CHARS": "1",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        captured_audio = []
        voice = (10000).to_bytes(2, "little", signed=True) * 2
        silence = b"\x00\x00" * 2

        def fake_process_transcribe_job(model, job):
            captured_audio.append(job.audio_bytes)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="hello",
                has_speech=True,
                detected_language="en",
                force_emit=job.force_emit,
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_bytes(voice)
                websocket.send_bytes(silence)
                result = websocket.receive_json()

        assert result["metrics"]["force_emit"] is True
        assert captured_audio == [voice]

    def test_stt_explicit_flush_emits_pending_audio_without_ending_stream(self):
        """An enterprise acoustic-pause control force-emits the current window."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "1.0",
                "STT_EMIT_MIN_CHARS": "1",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        voice = (10000).to_bytes(2, "little", signed=True) * 2
        captured_force_emit = []

        def fake_process_transcribe_job(model, job):
            captured_force_emit.append(job.force_emit)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="hello",
                has_speech=True,
                detected_language="en",
                force_emit=job.force_emit,
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_bytes(voice)
                websocket.send_text('{"type":"flush"}')
                result = websocket.receive_json()
                websocket.send_text('{"type":"end"}')

        assert result["text"] == "hello"
        assert result["metrics"]["force_emit"] is True
        assert captured_force_emit == [True]

    def test_stt_emit_interval_controls_live_inference_window(self):
        """Session context pacing gives STT the same larger audio window."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "0.2",
                "STT_EMIT_MIN_CHARS": "1",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        captured_audio = []
        first_voice = (10000).to_bytes(2, "little", signed=True) * 2
        remaining_voice = (10000).to_bytes(2, "little", signed=True) * 8

        def fake_process_transcribe_job(model, job):
            captured_audio.append(job.audio_bytes)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="complete statement",
                has_speech=True,
                detected_language="en",
                force_emit=job.force_emit,
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_text('{"emit_interval_seconds":1.0}')
                assert websocket.receive_json() == {
                    "type": "emit_interval_ack",
                    "emit_interval_seconds": 1.0,
                }
                websocket.send_bytes(first_voice)
                websocket.send_bytes(remaining_voice)
                result = websocket.receive_json()

        assert result["text"] == "complete statement"
        assert captured_audio == [first_voice + remaining_voice]

    def test_stt_explicit_flush_preempts_configured_inference_window(self):
        """An acoustic pause still processes speech before the context interval."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "0.2",
                "STT_EMIT_MIN_CHARS": "1",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        captured_audio = []
        voice = (10000).to_bytes(2, "little", signed=True) * 2

        def fake_process_transcribe_job(model, job):
            captured_audio.append(job.audio_bytes)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="paused statement",
                has_speech=True,
                detected_language="en",
                force_emit=job.force_emit,
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_text('{"emit_interval_seconds":1.0}')
                websocket.receive_json()
                websocket.send_bytes(voice)
                websocket.send_text('{"type":"flush"}')
                result = websocket.receive_json()

        assert result["metrics"]["force_emit"] is True
        assert captured_audio == [voice]

    def test_stt_pause_emit_policy_ignores_regular_chunk_size_flush(self):
        """Test pause mode does not emit just because the stream window is full."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "0.2",
                "STT_IDLE_FLUSH_SECONDS": "0.05",
                "STT_EMIT_MIN_CHARS": "1",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        voice = (10000).to_bytes(2, "little", signed=True) * 4
        captured_force_emit = []

        def fake_process_transcribe_job(model, job):
            captured_force_emit.append(job.force_emit)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="hello",
                has_speech=True,
                detected_language="en",
                force_emit=job.force_emit,
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_text('{"emit_policy":"pause"}')
                assert websocket.receive_json() == {
                    "type": "emit_policy_ack",
                    "emit_policy": "pause",
                }
                websocket.send_bytes(voice)
                result = websocket.receive_json()

        assert result["text"] == "hello"
        assert result["metrics"]["force_emit"] is True
        assert captured_force_emit == [True]

    def test_stt_acknowledges_candidate_languages_control_message(self):
        """Test STT normalizes and acknowledges candidate language controls."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {"STT_PRELOAD_MODEL": "false", "STT_TRANSCRIBE_WORKERS": "1"}
        )
        stt_main._get_model = lambda: object()

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_text(
                    '{"candidate_languages":["en-US","hi","en","",null]}'
                )
                result = websocket.receive_json()

        assert result == {
            "type": "candidate_languages_ack",
            "candidate_languages": ["en", "hi"],
        }

    def test_stt_idle_flush_emits_when_audio_chunks_stop_after_voice(self):
        """Test active speech is flushed if no more audio arrives after a short idle gap."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "10",
                "STT_IDLE_FLUSH_SECONDS": "0.05",
                "STT_VAD_SPEECH_PAD_MS": "0",
                "STT_EMIT_MIN_CHARS": "100",
                "STT_EMIT_INTERVAL_SECONDS": "999",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        voice = (10000).to_bytes(2, "little", signed=True) * 4
        captured_force_emit = []

        def fake_process_transcribe_job(model, job):
            captured_force_emit.append(job.force_emit)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="hello",
                has_speech=True,
                detected_language="en",
                force_emit=job.force_emit,
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_text('{"emit_policy":"pause"}')
                assert websocket.receive_json() == {
                    "type": "emit_policy_ack",
                    "emit_policy": "pause",
                }
                websocket.send_bytes(voice)
                result = websocket.receive_json()

        assert result["text"] == "hello"
        assert result["metrics"]["force_emit"] is True
        assert captured_force_emit == [True]

    def test_stt_idle_flush_does_not_change_live_emit_cadence(self):
        """Test idle flush does not force live-mode STT emissions."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "0.8",
                "STT_IDLE_FLUSH_SECONDS": "0.05",
                "STT_VAD_SPEECH_PAD_MS": "0",
                "STT_EMIT_MIN_CHARS": "1",
                "STT_EMIT_INTERVAL_SECONDS": "999",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        voice = (10000).to_bytes(2, "little", signed=True) * 4
        captured_force_emit = []

        def fake_process_transcribe_job(model, job):
            captured_force_emit.append(job.force_emit)
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="hello",
                has_speech=True,
                detected_language="en",
                force_emit=job.force_emit,
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_bytes(voice)
                time.sleep(0.1)
                websocket.send_bytes(voice)
                result = websocket.receive_json()

        assert result["text"] == "hello"
        assert result["metrics"]["force_emit"] is False
        assert captured_force_emit == [False]

    def test_stt_pause_emit_policy_flushes_pending_transcript_without_new_text(self):
        """Test pause-only mode emits held text when pause flush adds no new words."""
        from fastapi.testclient import TestClient

        stt_main = load_stt_main_module(
            {
                "STT_SAMPLE_RATE": "10",
                "STT_SAMPLE_WIDTH_BYTES": "2",
                "STT_STREAM_CHUNK_SECONDS": "0.2",
                "STT_CHUNK_OVERLAP_SECONDS": "0.2",
                "STT_PAUSE_FLUSH_SECONDS": "0.2",
                "STT_VAD_SPEECH_PAD_MS": "0",
                "STT_EMIT_MIN_CHARS": "100",
                "STT_EMIT_INTERVAL_SECONDS": "999",
                "STT_TRANSCRIBE_WORKERS": "1",
            }
        )
        voice = (10000).to_bytes(2, "little", signed=True) * 4
        silence = b"\x00\x00" * 2

        def fake_process_transcribe_job(model, job):
            return stt_main.TranscribeResult(
                sequence=job.sequence,
                transcript="hello",
                has_speech=True,
                detected_language="en",
                force_emit=job.force_emit,
            )

        stt_main._get_model = lambda: object()
        stt_main._process_transcribe_job = fake_process_transcribe_job

        with TestClient(stt_main.app) as client:
            with client.websocket_connect("/v1/stream/transcriptions") as websocket:
                websocket.send_text('{"emit_policy":"pause"}')
                assert websocket.receive_json() == {
                    "type": "emit_policy_ack",
                    "emit_policy": "pause",
                }
                websocket.send_bytes(voice)
                websocket.send_bytes(silence)
                result = websocket.receive_json()

        assert result["text"] == "hello"
        assert result["metrics"]["force_emit"] is True

    def test_stt_transcribe_job_preserves_pause_force_emit(self):
        """Test pause-flushed jobs remain emit-eligible after inference."""
        stt_main = load_stt_main_module()

        def fake_transcribe_audio(
            model, wav_buffer, language, task, candidate_languages=()
        ):
            return "short phrase", True, "en"

        stt_main._transcribe_audio = fake_transcribe_audio
        job = stt_main.TranscribeJob(
            sequence=1,
            audio_bytes=(1000).to_bytes(2, "little", signed=True) * 1000,
            language="en",
            candidate_languages=(),
            task="transcribe",
            queued_at=1.0,
            queue_depth_at_enqueue=0,
            force_emit=True,
        )

        result = stt_main._process_transcribe_job(object(), job)

        assert result.transcript == "short phrase"
        assert result.force_emit is True

    def test_stt_candidate_language_selection_excludes_unrelated_language(self):
        """Candidate languages constrain the explicit Whisper decode language."""
        from types import SimpleNamespace

        stt_main = load_stt_main_module({"STT_PRELOAD_MODEL": "false"})

        class FakeModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, wav_buffer, **kwargs):
                self.calls.append(kwargs["language"])
                info = SimpleNamespace(
                    language="ko",
                    all_language_probs=[("ko", 0.6), ("en", 0.3), ("hi", 0.1)],
                )
                if kwargs["language"] is None:
                    return iter(()), info
                segment = SimpleNamespace(
                    text="hello", no_speech_prob=0.0, avg_logprob=0.0
                )
                return iter((segment,)), SimpleNamespace(language=kwargs["language"])

        model = FakeModel()
        result = stt_main._transcribe_audio(
            model,
            stt_main._build_wav_buffer(b"\\x01\\x00" * 100),
            None,
            "transcribe",
            ("en", "hi"),
        )

        assert result == ("hello", True, "en")
        assert model.calls == [None, "en", "hi"]

    def test_stt_candidate_language_selection_uses_script_for_ambiguous_audio(self):
        """Ambiguous Gujarati audio is not forced into plausible English text."""
        from types import SimpleNamespace

        stt_main = load_stt_main_module({"STT_PRELOAD_MODEL": "false"})

        class FakeModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, wav_buffer, **kwargs):
                language = kwargs["language"]
                self.calls.append(language)
                if language is None:
                    return iter(()), SimpleNamespace(
                        language="hi",
                        all_language_probs=[
                            ("hi", 0.47),
                            ("en", 0.30),
                            ("gu", 0.23),
                        ],
                    )
                text = (
                    "What are you doing?" if language == "en" else "તમે શું કરી રહ્યા છો?"
                )
                segment = SimpleNamespace(
                    text=text, no_speech_prob=0.0, avg_logprob=-0.2
                )
                return iter((segment,)), SimpleNamespace(language=language)

        model = FakeModel()
        result = stt_main._transcribe_audio(
            model,
            stt_main._build_wav_buffer(b"\\x01\\x00" * 100),
            None,
            "transcribe",
            ("gu", "en"),
        )

        assert result == ("તમે શું કરી રહ્યા છો?", True, "gu")
        assert model.calls == [None, "gu", "en"]

    def test_stt_candidate_language_selection_caps_explicit_decodes(self):
        """Ambiguous windows decode only the first configured candidates."""
        from types import SimpleNamespace

        stt_main = load_stt_main_module(
            {
                "STT_PRELOAD_MODEL": "false",
                "STT_MAX_CANDIDATE_DECODES": "2",
            }
        )

        class FakeModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, wav_buffer, **kwargs):
                language = kwargs["language"]
                self.calls.append(language)
                if language is None:
                    return iter(()), SimpleNamespace(
                        language="ko",
                        all_language_probs=[
                            ("gu", 0.15),
                            ("en", 0.55),
                            ("hi", 0.25),
                            ("fr", 0.05),
                        ],
                    )
                segment = SimpleNamespace(
                    text=f"text-{language}",
                    no_speech_prob=0.0,
                    avg_logprob=0.0,
                )
                return iter((segment,)), SimpleNamespace(language=language)

        model = FakeModel()
        result = stt_main._transcribe_audio(
            model,
            stt_main._build_wav_buffer(b"\x01\x00" * 100),
            None,
            "transcribe",
            ("gu", "en", "hi", "fr"),
        )

        assert result[1] is True
        assert result[2] == "en"
        assert model.calls == [None, "gu", "en"]

    def test_stt_candidate_tuning_defaults_are_bounded_and_documented(self):
        """Candidate fallback defaults cap inference and retain script scoring."""
        with patch.dict(os.environ, {}, clear=True):
            stt_main = load_stt_main_module(
                {
                    "STT_PRELOAD_MODEL": "false",
                    "TEST_VARIANT": "candidate_tuning_defaults",
                }
            )

        assert stt_main.MAX_CANDIDATE_DECODES == 3
        assert stt_main.CANDIDATE_SCRIPT_SCORE_WEIGHT == 1.50

    def test_stt_candidate_language_selection_keeps_confident_fast_path(self):
        """A confident selected-language detection avoids duplicate decoding."""
        from types import SimpleNamespace
        from unittest.mock import Mock

        stt_main = load_stt_main_module({"STT_PRELOAD_MODEL": "false"})
        segment = SimpleNamespace(text="hello", no_speech_prob=0.0, avg_logprob=-0.1)
        model = SimpleNamespace()
        model.transcribe = Mock(
            return_value=(
                iter((segment,)),
                SimpleNamespace(language="en", all_language_probs=[("en", 0.91)]),
            )
        )

        result = stt_main._transcribe_audio(
            model,
            stt_main._build_wav_buffer(b"\\x01\\x00" * 100),
            None,
            "transcribe",
            ("gu", "en"),
        )

        assert result == ("hello", True, "en")
        assert model.transcribe.call_count == 1

    def test_stt_silence_job_preserves_pause_force_emit(self):
        """Test pause-flushed silence jobs preserve force_emit."""
        stt_main = load_stt_main_module()
        job = stt_main.TranscribeJob(
            sequence=1,
            audio_bytes=b"\x00\x00" * 100,
            language="en",
            candidate_languages=(),
            task="transcribe",
            queued_at=1.0,
            queue_depth_at_enqueue=0,
            force_emit=True,
        )

        result = stt_main._process_transcribe_job(object(), job)

        assert result.skipped_silence is True
        assert result.force_emit is True

    def test_stt_metrics_include_pause_force_emit(self):
        """Test STT metrics expose pause-forced emissions to the pipeline."""
        stt_main = load_stt_main_module()
        result = stt_main.TranscribeResult(sequence=1, force_emit=True)

        metrics = stt_main._result_metrics(result)

        assert metrics["force_emit"] is True

    def test_stt_pause_flush_trims_trailing_silence_before_inference(self):
        """Test pause flush keeps speech pad but removes excess silence."""
        stt_main = load_stt_main_module({"STT_VAD_SPEECH_PAD_MS": "200"})
        bytes_per_second = 1000
        speech = b"a" * 100
        silence = b"b" * 600

        trimmed = stt_main._trim_pause_flush_audio(
            speech + silence,
            trailing_silence_bytes=len(silence),
            bytes_per_second=bytes_per_second,
        )

        assert trimmed == speech + (b"b" * 200)

    def test_stt_pause_flush_keeps_audio_when_silence_is_within_pad(self):
        """Test pause flush does not trim short trailing pads."""
        stt_main = load_stt_main_module({"STT_VAD_SPEECH_PAD_MS": "200"})
        audio = b"a" * 100 + b"b" * 100

        trimmed = stt_main._trim_pause_flush_audio(
            audio,
            trailing_silence_bytes=100,
            bytes_per_second=1000,
        )

        assert trimmed == audio

    def test_stt_overlap_ignores_boundary_punctuation(self):
        """Test overlapped chunk text does not repeat punctuation variants."""
        stt_main = load_stt_main_module()

        new_text = stt_main._get_new_transcript_text(
            "Ich meine nicht ein paar Minuten.",
            "Minuten zu spät zu sein.",
        )

        assert new_text == "zu spät zu sein."

    def test_stt_overlap_ignores_boundary_case(self):
        """Test overlapped chunk text does not repeat case variants."""
        stt_main = load_stt_main_module()

        new_text = stt_main._get_new_transcript_text(
            "Und dann sitzt du da.",
            "Da mit diesem schweren Gefühl.",
        )

        assert new_text == "mit diesem schweren Gefühl."

    def test_stt_overlap_preserves_legitimate_new_text_without_overlap(self):
        """Test unrelated text is still treated as new speech."""
        stt_main = load_stt_main_module()

        new_text = stt_main._get_new_transcript_text(
            "Erster Satz.",
            "Ganz anderer Satz.",
        )

        assert new_text == "Ganz anderer Satz."

    def test_stt_overlap_matches_punctuation_only_tokens(self):
        """Test punctuation-only overlap tokens compare after normalization."""
        stt_main = load_stt_main_module()

        assert stt_main._overlap_words_match(["."], ["!"]) is True

    def test_stt_overlap_handles_punctuation_only_words(self):
        """Test punctuation-only overlap words do not duplicate text."""
        stt_main = load_stt_main_module()

        new_text = stt_main._get_new_transcript_text(
            "Ich sage .",
            ". zu dir",
        )

        assert new_text == "zu dir"

    def test_stt_drops_no_speech_new_text(self):
        """Test no-speech transcript deltas are suppressed before emitting."""
        stt_main = load_stt_main_module()

        result = stt_main.TranscribeResult(
            sequence=1,
            transcript="Hast du schon einmal Alarm?",
            has_speech=False,
        )

        assert stt_main._should_drop_no_speech_new_text(result, "Alarm?") is True

    def test_stt_keeps_speech_new_text(self):
        """Test accepted speech transcript deltas remain emit-eligible."""
        stt_main = load_stt_main_module()

        result = stt_main.TranscribeResult(
            sequence=1,
            transcript="Hast du schon einmal allein gesessen?",
            has_speech=True,
        )

        assert (
            stt_main._should_drop_no_speech_new_text(result, "allein gesessen?")
            is False
        )

    def test_stt_keeps_empty_no_speech_new_text(self):
        """Test empty no-speech deltas are not treated as dropped text."""
        stt_main = load_stt_main_module()

        result = stt_main.TranscribeResult(
            sequence=1,
            transcript="   ",
            has_speech=False,
        )

        assert stt_main._should_drop_no_speech_new_text(result, "") is False

    def test_stt_collapses_repeated_word_sequence(self):
        """Test repeated ASR word loops are collapsed in transcript deltas."""
        stt_main = load_stt_main_module()

        new_text = stt_main._collapse_repeated_word_sequences(
            "Weit- und Weit- und Weit- und Weit- und Eigenschaften schließt."
        )

        assert new_text == "Weit- und Weit- und Eigenschaften schließt."

    def test_stt_trims_cross_delta_single_word_loop(self):
        """Test long repeated words do not keep growing across deltas."""
        stt_main = load_stt_main_module()

        new_text = stt_main._trim_repeated_leading_words(
            "you give up also also also also also also",
            "also also also also also That's exactly what's most important.",
        )

        assert new_text == "That's exactly what's most important."

    def test_stt_preserves_short_cross_delta_repeats(self):
        """Test boundary repeat trimming preserves normal repeated wording."""
        stt_main = load_stt_main_module()

        new_text = stt_main._trim_repeated_leading_words(
            "you give up real real",
            "real real example.",
        )

        assert new_text == "real real example."

    def test_stt_preserves_cross_delta_text_with_empty_existing_text(self):
        """Test repeat trimming is a no-op without existing transcript text."""
        stt_main = load_stt_main_module()

        new_text = stt_main._trim_repeated_leading_words(
            "",
            "also also example.",
        )

        assert new_text == "also also example."

    def test_stt_trims_punctuation_after_existing_word_loop(self):
        """Test punctuation-only leading tokens do not extend word loops."""
        stt_main = load_stt_main_module()

        new_text = stt_main._trim_repeated_leading_words(
            "word word word word word word",
            "! ! ! ! ! ! Hello",
        )

        assert new_text == "Hello"

    def test_stt_trims_punctuation_then_repeated_word_loop(self):
        """Test punctuation before repeated loop words is trimmed too."""
        stt_main = load_stt_main_module()

        new_text = stt_main._trim_repeated_leading_words(
            "word word word word word word",
            "! ! word word Hello",
        )

        assert new_text == "Hello"

    def test_stt_cross_delta_repeat_threshold_has_safe_minimum(self):
        """Test configured repeat threshold cannot become too aggressive."""
        stt_main = load_stt_main_module({"STT_MAX_CROSS_DELTA_WORD_REPEATS": "1"})

        assert stt_main.MAX_CROSS_DELTA_WORD_REPEATS == 3

    def test_stt_preserves_non_adjacent_repeated_words(self):
        """Test normal non-adjacent repeated wording is not collapsed."""
        stt_main = load_stt_main_module()

        new_text = stt_main._collapse_repeated_word_sequences(
            "Vielleicht nicht perfekt. Vielleicht nicht leicht."
        )

        assert new_text == "Vielleicht nicht perfekt. Vielleicht nicht leicht."


class TestBenchmarks:
    """Tests for benchmark helper behavior."""

    def test_stt_benchmark_missing_audio_has_clear_error(self):
        """Test missing STT benchmark audio gives a helpful error."""
        from benchmark_stt import read_pcm_wav

        missing_path = "/tmp/polytalk-missing-benchmark-audio.wav"

        try:
            read_pcm_wav(missing_path)
        except ValueError as exc:
            assert "Audio file not found" in str(exc)
            assert missing_path in str(exc)
        else:
            raise AssertionError("read_pcm_wav should reject missing files")

    def test_stt_benchmark_invalid_audio_has_clear_error(self, tmp_path):
        """Test invalid STT benchmark audio gives a helpful error."""
        from benchmark_stt import read_pcm_wav

        invalid_path = tmp_path / "invalid.wav"
        invalid_path.write_bytes(b"not a wav")

        try:
            read_pcm_wav(str(invalid_path))
        except ValueError as exc:
            assert "Invalid WAV file" in str(exc)
            assert str(invalid_path) in str(exc)
        else:
            raise AssertionError("read_pcm_wav should reject invalid WAV files")

    def test_pipeline_benchmark_missing_audio_has_clear_error(self):
        """Test missing pipeline benchmark audio gives a helpful error."""
        from benchmark_pipeline import read_pcm_wav

        missing_path = "/tmp/polytalk-missing-pipeline-audio.wav"

        try:
            read_pcm_wav(missing_path)
        except ValueError as exc:
            assert "Audio file not found" in str(exc)
            assert missing_path in str(exc)
        else:
            raise AssertionError("read_pcm_wav should reject missing files")

    def test_pipeline_benchmark_invalid_audio_has_clear_error(self, tmp_path):
        """Test invalid pipeline benchmark audio gives a helpful error."""
        from benchmark_pipeline import read_pcm_wav

        invalid_path = tmp_path / "invalid.wav"
        invalid_path.write_bytes(b"not a wav")

        try:
            read_pcm_wav(str(invalid_path))
        except ValueError as exc:
            assert "Invalid WAV file" in str(exc)
            assert str(invalid_path) in str(exc)
        else:
            raise AssertionError("read_pcm_wav should reject invalid WAV files")

    def test_translation_benchmark_api_key_uses_environment(self, monkeypatch):
        """Test translation benchmark reads API key from the environment."""
        import argparse

        from benchmark_translation import resolve_api_key

        monkeypatch.setenv("TRANSLATION_API_KEY", "env-key")
        args = argparse.Namespace(api_key="")

        assert resolve_api_key(args) == "env-key"

    def test_translation_benchmark_cli_api_key_warns(self, capsys):
        """Test CLI API key emits a security warning."""
        import argparse

        from benchmark_translation import resolve_api_key

        args = argparse.Namespace(api_key="cli-key")

        assert resolve_api_key(args) == "cli-key"
        captured = capsys.readouterr()
        assert "--api-key may be visible" in captured.err
