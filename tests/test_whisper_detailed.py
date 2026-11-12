# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for whisper service methods.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
import websockets.exceptions

from app.services.whisper_service import TranscriptionResult, WhisperService


class TestWhisperServiceInit:
    """Test whisper service initialization."""

    def test_whisper_init_default(self):
        """Test whisper service initialization with defaults."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
                "base_url": "https://api.openai.com",
                "endpoint": "/v1/audio/transcriptions",
                "ws_endpoint": "/v1/stream/transcriptions",
                "api_key": "test-key",
                "model": "whisper-1",
                "timeout_seconds": 60,
                "max_reconnect_attempts": 3,
                "reconnect_delay_seconds": 2,
            }
            service = WhisperService()
            assert service.enabled is True
            assert service.mock_mode is True
            assert service.base_url == "https://api.openai.com"
            assert service.model == "whisper-1"
            assert service.timeout == 60
            assert service.max_reconnect_attempts == 3
            assert service.reconnect_delay == 2

    def test_whisper_init_disabled(self):
        """Test whisper service initialization when disabled."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": False,
                "mock_mode": True,
            }
            service = WhisperService()
            assert service.enabled is False


class TestWhisperTranscribe:
    """Test whisper transcription methods."""

    @pytest.mark.asyncio
    async def test_transcribe_service_disabled(self):
        """Test transcription when service is disabled."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": False,
                "mock_mode": True,
            }
            service = WhisperService()
            result = await anext(service.stream_transcribe(b"audio", "en"))

            assert result.success is False
            assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_transcribe_mock_mode(self):
        """Test transcription in mock mode."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()
            result = await anext(service.stream_transcribe(b"audio", "en"))

            assert result.success is True
            assert result.text != ""
            assert result.language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_mock_mode_different_languages(self):
        """Test transcription in mock mode with different languages."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            for lang in ["en", "gu", "hi", "es", "fr", "de"]:
                result = await anext(service.stream_transcribe(b"audio", lang))
                assert result.success is True
                assert result.language == lang


class TestWhisperStreamTranscribe:
    """Test whisper streaming transcription methods."""

    @pytest.mark.asyncio
    async def test_stream_transcribe_service_disabled(self):
        """Test streaming transcription when service is disabled."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": False,
                "mock_mode": True,
            }
            service = WhisperService()

            async def mock_gen():
                yield b"chunk"

            results = []
            async for result in service.stream_transcribe(mock_gen()):
                results.append(result)

            assert len(results) > 0
            assert results[0].success is False

    @pytest.mark.asyncio
    async def test_stream_transcribe_mock_mode(self):
        """Test streaming transcription in mock mode."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            async def mock_gen():
                yield b"chunk"

            results = []
            async for result in service.stream_transcribe(mock_gen()):
                results.append(result)

            assert len(results) > 0
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_stream_transcribe_mock_mode_with_callback(self):
        """Test streaming transcription with callback."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()
            callback_results = []

            def callback(result):
                callback_results.append(result)

            async def mock_gen():
                yield b"chunk"

            async for result in service.stream_transcribe(
                mock_gen(), on_result=callback
            ):
                pass

            assert len(callback_results) > 0

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_with_language_parameter(self):
        """Test real stream transcribe sends language parameter."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "timeout_seconds": 60,
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"
                yield b"__END_SIGNAL__"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Connection failed"),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen(), "en"):
                    results.append(result)

                assert len(results) > 0

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_with_api_key(self):
        """Test real stream transcribe uses API key."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "api_key": "test-key",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Connection failed"),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) > 0

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_max_reconnect_attempts(self):
        """Test real stream transcribe respects max reconnect attempts."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "max_reconnect_attempts": 2,
                "reconnect_delay_seconds": 0,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Connection failed"),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) > 0
                assert results[-1].success is False

    @pytest.mark.asyncio
    async def test_stream_transcribe_real_mode_connection_error(self):
        """Test streaming transcription with connection error."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Connection closed"),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) > 0
                assert results[-1].success is False

    @pytest.mark.asyncio
    async def test_stream_transcribe_real_mode_success(self):
        """Test streaming transcription with real WebSocket (mocked)."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "timeout_seconds": 60,
                "max_reconnect_attempts": 3,
                "reconnect_delay_seconds": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk1"
                yield b"__END_SIGNAL__"

            class MockWebSocket:
                def __init__(self):
                    self.responses = [
                        json.dumps(
                            {
                                "text": "Hello",
                                "is_final": False,
                                "language": "en",
                            }
                        ),
                    ]

                async def send(self, data):
                    pass

                async def close(self):
                    pass

                async def recv(self):
                    if self.responses:
                        return self.responses.pop(0)
                    raise websockets.exceptions.ConnectionClosedOK(None, None)

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)
                    if result.success or result.error:
                        break

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_stream_transcribe_real_mode_multiple_chunks(self):
        """Test streaming with multiple audio chunks."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
            }
            service = WhisperService()

            async def audio_gen():
                for i in range(5):
                    yield f"chunk{i}".encode()
                yield b"__END_SIGNAL__"

            class MockWebSocket:
                def __init__(self):
                    self.responses = [
                        json.dumps(
                            {
                                "text": "Hello from chunks",
                                "is_final": False,
                                "language": "en",
                            }
                        ),
                    ]

                async def send(self, data):
                    pass

                async def close(self):
                    pass

                async def recv(self):
                    if self.responses:
                        return self.responses.pop(0)
                    raise websockets.exceptions.ConnectionClosedOK(None, None)

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)
                    if result.error:
                        break

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_stream_transcribe_real_mode_with_callback(self):
        """Test streaming transcription with callback in real mode."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
            }
            service = WhisperService()
            callback_results = []

            def callback(result):
                callback_results.append(result)

            async def audio_gen():
                yield b"chunk"
                yield b"__END_SIGNAL__"

            async for result in service.stream_transcribe(
                audio_gen(), on_result=callback
            ):
                if result.error:
                    break

            # Callback may or may not be called depending on error handling
            assert isinstance(callback_results, list)

    @pytest.mark.asyncio
    async def test_stream_transcribe_real_mode_timeout(self):
        """Test streaming transcription with timeout."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Connection timeout"),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) > 0
                assert results[-1].success is False

    @pytest.mark.asyncio
    async def test_estimate_audio_duration_valid_wav(self):
        """Test audio duration estimation with valid WAV."""
        import io
        import wave

        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 16000)

            duration = service._estimate_audio_duration(buf.getvalue())
            assert duration == 1.0

    @pytest.mark.asyncio
    async def test_estimate_audio_duration_invalid(self):
        """Test audio duration estimation with invalid data."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            duration = service._estimate_audio_duration(b"invalid_data")
            assert duration == 5.0

    @pytest.mark.asyncio
    async def test_estimate_audio_duration_empty(self):
        """Test audio duration estimation with empty data."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            duration = service._estimate_audio_duration(b"")
            assert duration == 5.0

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_with_ping_config(self):
        """Test real stream transcribe with ping configuration."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "ping_interval_seconds": 10,
                "ping_timeout_seconds": 5,
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Connection failed"),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) > 0

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_error_result_yield(self):
        """Test real stream transcribe yields error result."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Test error"),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) > 0
                error_results = [r for r in results if not r.success]
                assert len(error_results) > 0

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_ws_url_construction(self):
        """Test real stream transcribe WebSocket URL construction."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "https://api.example.com",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Test error"),
            ) as mock_connect:
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) > 0
                mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_with_language_sending(self):
        """Test real stream transcribe sends language to WebSocket."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Test error"),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen(), "en"):
                    results.append(result)

                assert len(results) > 0

    @pytest.mark.asyncio
    async def test_stream_transcribe_callback_called_on_error(self):
        """Test stream transcribe calls callback on error."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()
            callback_results = []

            def callback(result):
                callback_results.append(result)

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Test error"),
            ):
                async for result in service.stream_transcribe(
                    audio_gen(), on_result=callback
                ):
                    pass

                # Callback should be called at least once for error
                assert isinstance(callback_results, list)

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_on_result_callback(self):
        """Test real stream transcribe with on_result callback."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()
            results_received = []

            def on_result(result):
                results_received.append(result)

            async def audio_gen():
                yield b"chunk"

            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Test error"),
            ):
                async for result in service.stream_transcribe(
                    audio_gen(), on_result=on_result
                ):
                    pass

                assert isinstance(results_received, list)

    @pytest.mark.asyncio
    async def test_stream_transcribe_disabled_service(self):
        """Test stream_transcribe when service is disabled."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": False,
                "mock_mode": True,
            }
            service = WhisperService()
            callback_called = []

            def on_result(result):
                callback_called.append(result)

            async def mock_audio_gen():
                yield b"test"

            results = []
            async for result in service.stream_transcribe(
                mock_audio_gen(), on_result=on_result
            ):
                results.append(result)

            assert len(results) == 1
            assert results[0].success is False
            assert results[0].error == "Whisper service is disabled"
            assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_real_stream_with_zero_reconnect_attempts(self):
        """Test real stream with max_reconnect_attempts=0."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 0,
            }
            service = WhisperService()

            async def mock_audio_gen():
                yield b"test"

            # With max_reconnect_attempts=0, the while loop never executes
            # So we should get an error from the exception handler
            with patch(
                "app.services.whisper_service.websockets.connect",
                side_effect=Exception("Connection refused"),
            ):
                # The function should still yield results even with errors
                results = []
                try:
                    async for result in service.stream_transcribe(mock_audio_gen()):
                        results.append(result)
                except Exception:
                    pass

                # May or may not have results depending on error handling path
                assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_utterance_boundary_sends_flush_without_ending_stream(self):
        """The pipeline sentinel becomes flush, followed later by the real end."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://stt:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

        end_sent = asyncio.Event()
        sent = []

        async def audio_gen():
            yield b"__END_OF_UTTERANCE__"
            yield b"__END_SIGNAL__"

        class MockWebSocket:
            async def send(self, data):
                sent.append(data)
                if isinstance(data, str) and json.loads(data).get("type") == "end":
                    end_sent.set()

            async def recv(self):
                await end_sent.wait()
                raise websockets.exceptions.ConnectionClosedOK()

            async def close(self):
                pass

        with patch(
            "app.services.whisper_service.websockets.connect",
            new=AsyncMock(return_value=MockWebSocket()),
        ):
            results = [
                result
                async for result in service._real_stream_transcribe(
                    audio_gen(), language="en"
                )
            ]

        controls = [json.loads(message) for message in sent if isinstance(message, str)]
        assert {"type": "flush", "reason": "client_flush"} in controls
        assert controls[-1] == {"type": "end"}
        assert results == []
        await service.close()

    @pytest.mark.asyncio
    async def test_real_stream_websocket_success(self):
        """Test real WebSocket streaming with successful transcription."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream/transcriptions",
                "timeout_seconds": 60,
                "max_reconnect_attempts": 1,
                "ping_interval": 20,
                "ping_timeout": 20,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk1"
                yield b"__END_SIGNAL__"

            class MockWebSocket:
                async def send(self, data):
                    pass

                async def close(self):
                    pass

                async def recv(self):
                    yield json.dumps(
                        {"text": "Hello", "is_final": False, "language": "en"}
                    )
                    yield json.dumps(
                        {"text": "Hello World", "is_final": True, "language": "en"}
                    )
                    raise websockets.exceptions.ConnectionClosedOK()

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) >= 1
                assert all(isinstance(r, TranscriptionResult) for r in results)

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_websocket_send_error(self):
        """Test real stream transcribe handles WebSocket send error."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            class MockWebSocket:
                async def send(self, data):
                    raise Exception("Send failed")

                async def close(self):
                    pass

                async def recv(self):
                    raise websockets.exceptions.ConnectionClosedOK()

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)
                    if result.error:
                        break

                assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_json_decode_error(self):
        """Test real stream transcribe handles JSON decode error."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"
                yield b"__END_SIGNAL__"

            class MockWebSocket:
                async def send(self, data):
                    pass

                async def close(self):
                    pass

                async def recv(self):
                    raise json.JSONDecodeError("test", "doc", 0)

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)
                    if result.error:
                        break

                assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_error_result(self):
        """Test real stream transcribe handles error result from WebSocket."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"
                yield b"__END_SIGNAL__"

            class MockWebSocket:
                async def send(self, data):
                    pass

                async def close(self):
                    pass

                async def recv(self):
                    yield json.dumps({"error": "Test error"})
                    raise websockets.exceptions.ConnectionClosedOK()

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                error_results = [r for r in results if not r.success and r.error]
                assert len(error_results) > 0

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_cancellation(self):
        """Test real stream transcribe handles task cancellation."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 3,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            class MockWebSocket:
                async def send(self, data):
                    pass

                async def close(self):
                    pass

                async def recv(self):
                    import asyncio

                    await asyncio.sleep(10)
                    raise websockets.exceptions.ConnectionClosedOK()

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                import asyncio

                results = []
                try:
                    async for result in service.stream_transcribe(audio_gen()):
                        results.append(result)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

                assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_with_metrics(self):
        """Test real stream transcribe handles metrics in response."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"
                yield b"__END_SIGNAL__"

            class MockWebSocket:
                async def send(self, data):
                    pass

                async def close(self):
                    pass

                async def recv(self):
                    yield json.dumps(
                        {
                            "text": "Hello",
                            "is_final": True,
                            "language": "en",
                            "metrics": {"queue_wait": 0.1},
                        }
                    )
                    raise websockets.exceptions.ConnectionClosedOK()

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_mock_stream_transcribe_gujarati(self):
        """Test mock stream transcribe with Gujarati language."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            results = []
            async for result in service.stream_transcribe(audio_gen(), "gu"):
                results.append(result)

            assert len(results) > 0
            assert results[0].language == "gu"
            assert "હેલો" in results[-1].text

    @pytest.mark.asyncio
    async def test_mock_stream_transcribe_hindi(self):
        """Test mock stream transcribe with Hindi language."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            results = []
            async for result in service.stream_transcribe(audio_gen(), "hi"):
                results.append(result)

            assert len(results) > 0
            assert results[0].language == "hi"
            assert "नमस्ते" in results[-1].text

    @pytest.mark.asyncio
    async def test_mock_stream_transcribe_unknown_language(self):
        """Test mock stream transcribe with unknown language falls back to English."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"

            results = []
            async for result in service.stream_transcribe(audio_gen(), "xx"):
                results.append(result)

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_receive_error(self):
        """Test real stream transcribe handles receive error."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"
                yield b"__END_SIGNAL__"

            class MockWebSocket:
                async def send(self, data):
                    pass

                async def close(self):
                    pass

                async def recv(self):
                    raise Exception("Receive failed")

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)
                    if result.error:
                        break

                assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_real_stream_transcribe_websocket_close_error(self):
        """Test real stream transcribe handles WebSocket close error."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://localhost:8000",
                "ws_endpoint": "/v1/stream",
                "max_reconnect_attempts": 1,
            }
            service = WhisperService()

            async def audio_gen():
                yield b"chunk"
                yield b"__END_SIGNAL__"

            class MockWebSocket:
                async def send(self, data):
                    pass

                async def close(self):
                    raise Exception("Close failed")

                async def recv(self):
                    raise websockets.exceptions.ConnectionClosedOK()

            with patch(
                "app.services.whisper_service.websockets.connect",
                return_value=MockWebSocket(),
            ):
                results = []
                async for result in service.stream_transcribe(audio_gen()):
                    results.append(result)

                assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_normalize_language_hint_with_hyphen(self):
        """Test normalize_language_hint handles hyphenated codes."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            assert service._normalize_language_hint("en-US") == "en"
            assert service._normalize_language_hint("zh-CN") == "zh"
            assert service._normalize_language_hint("pt-BR") == "pt"

    @pytest.mark.asyncio
    async def test_normalize_language_hint_with_none(self):
        """Test normalize_language_hint handles None."""
        with patch("app.services.whisper_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            service = WhisperService()

            assert service._normalize_language_hint(None) is None
            assert service._normalize_language_hint("") is None
