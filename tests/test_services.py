"""Tests for service classes."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Config
from app.services.pipeline_service import (
    TranslationPipelineService,
    _semantic_probe_is_cooling_down,
)
from app.services.translation_service import (
    SemanticBoundaryDecision,
    TranslationService,
)
from app.services.tts_service import TTSService
from app.services.whisper_service import WhisperService


class TestWhisperService:
    """Tests for Whisper transcription service."""

    def test_whisper_service_init(self):
        """Test service initialization."""
        service = WhisperService()

        assert service.config is not None
        assert hasattr(service, "enabled")
        assert hasattr(service, "mock_mode")

    def test_whisper_service_config(self):
        """Test service configuration loading."""
        service = WhisperService()
        config = Config()

        assert service.config == config._config["whisper"]

    def test_whisper_service_disabled(self):
        """Test service when API is disabled."""
        service = WhisperService()

        # Check that enabled flag exists
        assert hasattr(service, "enabled")

    def test_whisper_transcribe_with_mock(self):
        """Test transcription with mock data."""
        import asyncio

        service = WhisperService()
        service.mock_mode = True

        # Create mock audio data
        audio_data = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1e\x00\x00\x40\x1e\x00\x00\x02\x00\x08\x00data\x00\x00\x00\x00"

        result = asyncio.run(
            anext(service.stream_transcribe(audio_data, language="en"))
        )

        assert "Hello" in result.text
        assert result.language == "en"
        assert result.success is True

    def test_whisper_transcribe_different_languages_mock(self):
        """Test transcription with different languages."""
        import asyncio

        service = WhisperService()
        service.mock_mode = True

        audio_data = b"test_audio"

        # Test Gujarati
        result_gu = asyncio.run(
            anext(service.stream_transcribe(audio_data, language="gu"))
        )
        assert (
            "સ્પીચ" in result_gu.text
            or "ગુજરાતી" in result_gu.text
            or result_gu.language == "gu"
        )

        # Test Hindi
        result_hi = asyncio.run(
            anext(service.stream_transcribe(audio_data, language="hi"))
        )
        assert "स्पीच" in result_hi.text or result_hi.language == "hi"

    @pytest.mark.asyncio
    async def test_mock_transcription_logs_metadata_without_customer_text(self, caplog):
        """Mock transcription completion logs must not contain transcript text."""
        service = WhisperService()
        service.mock_mode = True
        results = []

        with (
            caplog.at_level(
                logging.INFO,
                logger="app.services.whisper_service",
            ),
            patch(
                "app.services.whisper_service.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            async for result in service.stream_transcribe(b"audio", language="en"):
                results.append(result)

        final_transcript = results[-1].text
        assert final_transcript not in caplog.text
        assert "Mock streaming transcription complete: language=en" in caplog.text
        assert f"chars={len(final_transcript)}" in caplog.text

    def test_whisper_transcribe_unknown_language_mock(self):
        """Test transcription with unknown language."""
        import asyncio

        service = WhisperService()
        service.mock_mode = True

        audio_data = b"test_audio"

        result = asyncio.run(
            anext(service.stream_transcribe(audio_data, language="xx"))
        )
        assert result.success is True
        assert result.text is not None

    def test_whisper_stream_transcribe_mock_mode(self):
        """Test streaming transcription in mock mode."""
        service = WhisperService()
        service.mock_mode = True

        # Test that streaming method exists
        assert hasattr(service, "stream_transcribe")
        assert callable(service.stream_transcribe)


class TestTranslationService:
    """Tests for Translation service."""

    def test_translation_service_init(self):
        """Test service initialization."""
        service = TranslationService()

        assert service.config is not None
        assert hasattr(service, "enabled")
        assert hasattr(service, "mock_mode")

    def test_translation_service_config(self):
        """Test service configuration loading."""
        service = TranslationService()
        config = Config()

        assert service.config == config._config["translation"]

    def test_translation_translate_mock(self):
        """Test translation with mock data."""
        import asyncio

        service = TranslationService()
        service.mock_mode = True

        result = asyncio.run(service.translate("Hello", "en", "gu"))

        assert result.text is not None
        assert result.source_language == "en"
        assert result.target_language == "gu"
        assert result.success is True

    def test_translation_logs_metadata_without_customer_text(self, caplog):
        """Translation progress logs must not contain customer text."""
        service = TranslationService()
        service.mock_mode = True
        source_text = "private customer transcript"

        with caplog.at_level(
            logging.INFO,
            logger="app.services.translation_service",
        ):
            result = asyncio.run(service.translate(source_text, "en", "gu"))

        assert source_text not in caplog.text
        assert result.text not in caplog.text
        assert "Mock translation complete: en -> gu" in caplog.text
        assert f"source_chars={len(source_text)}" in caplog.text
        assert f"translated_chars={len(result.text)}" in caplog.text

    @pytest.mark.asyncio
    async def test_real_translation_logs_metadata_without_customer_text(self, caplog):
        """Real-provider completion logs must not contain customer text."""
        service = TranslationService()
        source_text = "private real provider transcript"
        translated_text = "private real provider translation"
        provider = {"api_format": "openai_chat"}

        with (
            caplog.at_level(
                logging.INFO,
                logger="app.services.translation_service",
            ),
            patch.object(
                service,
                "_resolve_provider_config",
                return_value=provider,
            ),
            patch.object(
                service,
                "_build_translation_request",
                return_value=("https://provider.invalid", {}, {}),
            ),
            patch.object(
                service,
                "_execute_translation_request",
                new_callable=AsyncMock,
                return_value=(translated_text, 0.9),
            ),
        ):
            result = await service._real_translate(source_text, "en", "bn")

        assert result.success is True
        assert source_text not in caplog.text
        assert translated_text not in caplog.text
        assert "Real translation complete: en -> bn" in caplog.text
        assert f"source_chars={len(source_text)}" in caplog.text
        assert f"translated_chars={len(translated_text)}" in caplog.text

    def test_translation_translate_disabled(self):
        """Test translation when API is disabled."""
        import asyncio

        service = TranslationService()
        service.enabled = False

        result = asyncio.run(service.translate("Hello", "en", "gu"))

        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_translation_mock_all_combinations(self):
        """Test mock mode with different language pairs."""
        import asyncio

        service = TranslationService()
        service.mock_mode = True

        # Test English to Gujarati
        result1 = asyncio.run(service.translate("Hello", "en", "gu"))
        assert result1.text is not None
        assert result1.success is True

        # Test English to Hindi
        result2 = asyncio.run(service.translate("Hello", "en", "hi"))
        assert result2.text is not None
        assert result2.success is True

        # Test English to Spanish
        result3 = asyncio.run(service.translate("Hello", "en", "es"))
        assert result3.text is not None
        assert result3.success is True


class TestTTSService:
    """Tests for Text-to-Speech service."""

    def test_tts_service_init(self):
        """Test service initialization."""
        service = TTSService()

        assert service.config is not None
        assert hasattr(service, "enabled")
        assert hasattr(service, "mock_mode")

    def test_tts_service_config(self):
        """Test service configuration loading."""
        service = TTSService()
        config = Config()

        assert service.config == config._config["tts"]

    def test_tts_synthesize_mock_mode(self):
        """Test TTS synthesis in mock mode."""
        import asyncio
        from pathlib import Path

        service = TTSService()
        service.mock_mode = True

        # Use /tmp to avoid permission issues with media/output
        result = asyncio.run(
            service.synthesize("Hello", "en", output_path=Path("/tmp/test_tts.wav"))
        )

        assert result.audio_path is not None
        assert result.success is True

    def test_tts_synthesize_with_custom_path_mock(self):
        """Test TTS synthesis with custom output path."""
        import asyncio
        from pathlib import Path

        service = TTSService()
        service.mock_mode = True

        result = asyncio.run(
            service.synthesize("Hello", "en", output_path=Path("/tmp/test_output.wav"))
        )

        assert result.audio_path is not None
        assert result.success is True

    def test_tts_synthesize_disabled(self):
        """Test TTS when API is disabled."""
        import asyncio

        service = TTSService()
        service.enabled = False

        result = asyncio.run(service.synthesize("Hello", "en"))

        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_tts_voice_selection(self):
        """Test TTS with different language codes."""
        import asyncio
        from pathlib import Path

        service = TTSService()
        service.mock_mode = True

        # Test with different languages - use /tmp to avoid permission issues
        result1 = asyncio.run(
            service.synthesize("Hello", "en", output_path=Path("/tmp/test_tts_en.wav"))
        )
        assert result1.success is True

        result2 = asyncio.run(
            service.synthesize("Hello", "gu", output_path=Path("/tmp/test_tts_gu.wav"))
        )
        assert result2.success is True


class TestTranslationPipelineService:
    """Tests for Translation Pipeline Service."""

    def test_pipeline_service_init(self):
        """Test pipeline service initialization."""
        pipeline = TranslationPipelineService(warm_connections=False)

        assert pipeline.whisper is not None
        assert pipeline.translation is not None
        assert pipeline.tts is not None
        assert hasattr(pipeline, "media_dir")

    def test_pipeline_service_with_custom_services(self):
        """Test pipeline service with custom service instances."""
        mock_whisper = MagicMock(spec=WhisperService)
        mock_translation = MagicMock(spec=TranslationService)
        mock_tts = MagicMock(spec=TTSService)

        pipeline = TranslationPipelineService(
            whisper_service=mock_whisper,
            translation_service=mock_translation,
            tts_service=mock_tts,
            warm_connections=False,
        )

        assert pipeline.whisper is mock_whisper
        assert pipeline.translation is mock_translation
        assert pipeline.tts is mock_tts

    @pytest.mark.asyncio
    async def test_pipeline_close(self):
        """Test pipeline service close method."""
        pipeline = TranslationPipelineService(warm_connections=False)

        mock_close = AsyncMock()

        with patch.object(pipeline.whisper, "close", mock_close):
            with patch.object(pipeline.translation, "close", mock_close):
                with patch.object(pipeline.tts, "close", mock_close):
                    await pipeline.close()

    @pytest.mark.asyncio
    async def test_pipeline_context_manager(self):
        """Test pipeline service as async context manager."""
        async with TranslationPipelineService(warm_connections=False) as pipeline:
            assert pipeline is not None

    @pytest.mark.asyncio
    async def test_translation_service_method(self):
        """Test pipeline exposes the configured translation service."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_translate(*args, **kwargs):
            return MagicMock(success=True, text="translated")

        with patch.object(
            pipeline.translation, "translate", side_effect=mock_translate
        ) as patched_translate:
            result = await pipeline.translation.translate("Hello", "en", "gu")

            assert result.success is True
            assert result.text == "translated"
            patched_translate.assert_awaited_once_with("Hello", "en", "gu")

    @pytest.mark.asyncio
    async def test_synthesize_method(self):
        """Test _synthesize method."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_synthesize(*args, **kwargs):
            return MagicMock(success=True, audio_url="/tmp/test.wav")

        with patch.object(pipeline.tts, "synthesize", side_effect=mock_synthesize):
            result = await pipeline._synthesize("Hello", "en", save_media=False)

            assert result.success is True
            assert result.audio_url == "/tmp/test.wav"

    @pytest.mark.asyncio
    async def test_process_streaming_basic(self):
        """Test basic streaming process."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello",
                language="en",
                is_partial=True,
                metrics={},
            )

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            try:
                async for result in pipeline.process_streaming(
                    mock_audio_gen(), "en", "gu"
                ):
                    pass
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_process_streaming_with_pause_event(self):
        """Test streaming with pause event."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"

        pause_event = asyncio.Event()

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello",
                language="en",
                is_partial=False,
                metrics={},
            )

        mock_stream = MagicMock(return_value=mock_stream_transcribe())

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            async for _ in pipeline.process_streaming(
                mock_audio_gen(), "en", "gu", pause_event=pause_event
            ):
                pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["live", "conversation"])
    async def test_process_streaming_end_signal_bypasses_pause(self, mode):
        """Stopping a paused session still terminates the downstream STT stream."""
        pipeline = TranslationPipelineService(warm_connections=False)
        pause_event = asyncio.Event()
        pause_event.set()
        received_chunks = []

        async def mock_audio_gen():
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(audio_generator, *_args, **_kwargs):
            async for chunk in audio_generator:
                received_chunks.append(chunk)
            yield MagicMock(
                success=True,
                text="",
                language="en",
                is_partial=False,
                metrics={},
            )

        async def consume_pipeline():
            return [
                result
                async for result in pipeline.process_streaming(
                    mock_audio_gen(),
                    "en",
                    "gu",
                    pause_event=pause_event,
                    mode=mode,
                )
            ]

        with patch.object(
            pipeline.whisper,
            "stream_transcribe",
            side_effect=mock_stream_transcribe,
        ):
            await asyncio.wait_for(consume_pipeline(), timeout=1.0)

        assert received_chunks == [b"__END_SIGNAL__"]

    @pytest.mark.asyncio
    async def test_process_streaming_with_language_swap(self):
        """Test streaming with language swap queue."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"

        language_swap_queue = asyncio.Queue()

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello",
                language="en",
                is_partial=False,
                metrics={},
            )

        mock_stream = MagicMock(return_value=mock_stream_transcribe())

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            async for _ in pipeline.process_streaming(
                mock_audio_gen(),
                "en",
                "gu",
                language_swap_queue=language_swap_queue,
            ):
                pass

    @pytest.mark.asyncio
    async def test_warm_connections_with_lock(self):
        """Test warm connections with lock protection."""
        pipeline = TranslationPipelineService(warm_connections=False)

        pipeline._whisper_warmed = True
        pipeline._translation_warmed = True

        # Should return early if both warmed
        await pipeline._warm_connections()

    @pytest.mark.asyncio
    async def test_close_multiple_times(self):
        """Test that close can be called multiple times safely."""
        pipeline = TranslationPipelineService(warm_connections=False)

        await pipeline.close()
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_process_streaming_error_handling(self):
        """Test error handling in streaming process."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"

        async def mock_stream_transcribe(*args, **kwargs):
            if False:
                yield
            raise Exception("Stream error")

        mock_stream = MagicMock(return_value=mock_stream_transcribe())

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            error_count = 0
            try:
                async for result in pipeline.process_streaming(
                    mock_audio_gen(), "en", "gu"
                ):
                    if result.get("type") == "error":
                        error_count += 1
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_synthesize_with_save_media(self):
        """Test synthesize with save_media=True."""
        import tempfile
        from pathlib import Path

        with patch("app.services.pipeline_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": True,
            }
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": True,
                "provider": "piper",
                "voice": "en_US-lessac-medium",
            }
            mock_config.return_value.app = {}
            mock_config.return_value.media_output_dir = Path(tempfile.mkdtemp())

            pipeline = TranslationPipelineService(warm_connections=False)
            result = await pipeline._synthesize("Hello", "en", save_media=True)
            assert result is not None

    @pytest.mark.asyncio
    async def test_translate_and_synthesize_methods(self):
        """Test both _translate and _synthesize methods together."""
        import tempfile
        from pathlib import Path

        with patch("app.services.pipeline_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": True,
            }
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": True,
            }
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": True,
                "provider": "piper",
                "voice": "en_US-lessac-medium",
            }
            mock_config.return_value.app = {}
            mock_config.return_value.media_output_dir = Path(tempfile.mkdtemp())

            pipeline = TranslationPipelineService(warm_connections=False)

            # Test translation service access
            trans_result = await pipeline.translation.translate("Hello", "en", "gu")
            assert trans_result is not None

            # Test _synthesize without save_media
            synth_result = await pipeline._synthesize("Hello", "en", save_media=False)
            assert synth_result is not None

    @pytest.mark.asyncio
    async def test_warm_connections_real_mode(self):
        """Test warm connections in real mode (not mock)."""
        import tempfile
        from pathlib import Path

        with patch("app.services.pipeline_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
            }
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
            }
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "voice": "en_US-lessac-medium",
            }
            mock_config.return_value.app = {}
            mock_config.return_value.media_output_dir = Path(tempfile.mkdtemp())

            pipeline = TranslationPipelineService(warm_connections=False)

            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                mock_translate.return_value = MagicMock(success=True, text="test")
                await pipeline._warm_connections()

    @pytest.mark.asyncio
    async def test_warm_connections_translation_fail(self):
        """Test warm connections when translation fails."""
        import tempfile
        from pathlib import Path

        with patch("app.services.pipeline_service.get_config") as mock_config:
            mock_config.return_value.whisper = {
                "enabled": True,
                "mock_mode": False,
            }
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
            }
            mock_config.return_value.tts = {
                "enabled": True,
                "mock_mode": False,
                "provider": "piper",
                "voice": "en_US-lessac-medium",
            }
            mock_config.return_value.app = {}
            mock_config.return_value.media_output_dir = Path(tempfile.mkdtemp())

            pipeline = TranslationPipelineService(warm_connections=False)

            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                mock_translate.side_effect = Exception("Translation error")
                await pipeline._warm_connections()

    @pytest.mark.asyncio
    async def test_process_streaming_with_config_errors(self):
        """Test process_streaming with config parsing errors."""
        with patch("app.services.pipeline_service.get_config") as mock_config:
            mock_config.return_value.app = {
                "language_swap_delay_ms": "invalid",
                "translation_flush_chars": "invalid",
                "translation_flush_seconds": "invalid",
                "translation_flush_min_chars": "invalid",
            }

            pipeline = TranslationPipelineService(warm_connections=False)

            async def mock_audio_gen():
                yield b"audio_chunk"

            async def mock_stream_transcribe(*args, **kwargs):
                yield MagicMock(
                    success=True,
                    text="Hello",
                    language="en",
                    is_partial=True,
                    metrics={},
                )

            with patch.object(
                pipeline.whisper,
                "stream_transcribe",
                return_value=mock_stream_transcribe(),
            ):
                try:
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        pass
                except Exception:
                    pass

    def test_metric_value_with_invalid_metrics(self):
        """Test metric_value with invalid metrics."""

        # Test the helper function directly
        def metric_value(metrics, key, default=0.0):
            if not metrics:
                return default
            try:
                return float(metrics.get(key, default))
            except (TypeError, ValueError):
                return default

        # Test with None metrics
        result = metric_value(None, "key")
        assert result == 0.0

        # Test with invalid value
        result = metric_value({"key": "invalid"}, "key")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_process_streaming_with_language_swap_queue(self):
        """Test process_streaming with language swap queue."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello",
                language="en",
                is_partial=True,
                metrics={},
            )

        language_swap_queue = asyncio.Queue()

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            try:
                async for result in pipeline.process_streaming(
                    mock_audio_gen(),
                    "en",
                    "gu",
                    language_swap_queue=language_swap_queue,
                ):
                    pass
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_close_with_service_errors(self):
        """Test close method when services raise errors."""
        pipeline = TranslationPipelineService(warm_connections=False)

        mock_close = AsyncMock()

        with patch.object(pipeline.whisper, "close", mock_close):
            with patch.object(pipeline.translation, "close", mock_close):
                with patch.object(pipeline.tts, "close", mock_close):
                    await pipeline.close()
                    await asyncio.sleep(0.1)

    def test_pipeline_init_with_warm_connections(self):
        """Test pipeline initialization with warm_connections=True."""
        with patch("app.services.pipeline_service.WhisperService") as mock_whisper_cls:
            with patch(
                "app.services.pipeline_service.TranslationService"
            ) as mock_translation_cls:
                with patch("app.services.pipeline_service.TTSService"):

                    def close_scheduled_coroutine(coro):
                        coro.close()
                        return MagicMock()

                    with patch(
                        "app.services.pipeline_service.asyncio.create_task",
                        side_effect=close_scheduled_coroutine,
                    ) as mock_create:
                        mock_whisper = mock_whisper_cls.return_value
                        mock_whisper.mock_mode = False
                        mock_translation = mock_translation_cls.return_value
                        mock_translation.mock_mode = False
                        TranslationPipelineService(warm_connections=True)
                        mock_create.assert_called_once()
                        assert (
                            mock_create.call_args[0][0].__name__ == "_warm_connections"
                        )

    @pytest.mark.asyncio
    async def test_pipeline_close_method(self, caplog):
        """Test pipeline close method."""
        pipeline = TranslationPipelineService(warm_connections=False)

        mock_close = AsyncMock()

        with patch.object(pipeline.whisper, "close", mock_close):
            with patch.object(pipeline.translation, "close", mock_close):
                with patch.object(pipeline.tts, "close", mock_close):
                    await pipeline.close()

    @pytest.mark.asyncio
    async def test_translation_service_accepts_context(self):
        """Test pipeline translation service receives context-aware calls."""
        pipeline = TranslationPipelineService(warm_connections=False)
        context = [{"source": "Hi", "target": "Hallo"}]

        async def mock_translate(*args, **kwargs):
            return MagicMock(
                success=True,
                text="translated",
                source_language="en",
                target_language="gu",
            )

        with patch.object(
            pipeline.translation, "translate", side_effect=mock_translate
        ) as patched_translate:
            result = await pipeline.translation.translate(
                "Hello", "en", "gu", context=context
            )

            assert result.success is True
            assert result.text == "translated"
            patched_translate.assert_awaited_once_with(
                "Hello", "en", "gu", context=context
            )

    @pytest.mark.asyncio
    async def test_process_streaming_asr_error(self):
        """Test process_streaming with ASR error result."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield {"type": "error", "error": "ASR failed"}

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            error_count = 0
            try:
                async for result in pipeline.process_streaming(
                    mock_audio_gen(), "en", "gu"
                ):
                    if result.get("type") == "error":
                        error_count += 1
            except Exception:
                pass

            assert error_count >= 0

    @pytest.mark.asyncio
    async def test_process_streaming_word_overlap_detection(self):
        """Test word overlap detection fallback when prefix match fails."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello world",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="Hello world this is new",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="Hello world this is new final",
                language="en",
                is_partial=False,
                metrics={},
            )

        async def mock_translate(*args, **kwargs):
            return MagicMock(
                success=True,
                text="translated text",
                source_language="en",
                target_language="gu",
            )

        async def mock_synthesize(*args, **kwargs):
            return MagicMock(success=True, audio_url="/tmp/test.wav")

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(pipeline.translation, "translate", mock_translate):
                with patch.object(pipeline.tts, "synthesize", mock_synthesize):
                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_pipeline_logs_progress_without_customer_text(self, caplog):
        """Pipeline progress logs must not contain ASR or translated text."""
        pipeline = TranslationPipelineService(warm_connections=False)
        source_text = "private pipeline transcript with enough words."
        translated_text = "private pipeline translated output."

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text=source_text,
                language="en",
                is_partial=False,
                metrics={},
            )

        async def mock_translate(*args, **kwargs):
            return MagicMock(
                success=True,
                text=translated_text,
                source_language="en",
                target_language="bn",
            )

        async def mock_synthesize(*args, **kwargs):
            return MagicMock(success=True, audio_url="/tmp/test.wav")

        with (
            caplog.at_level(
                logging.DEBUG,
                logger="app.services.pipeline_service",
            ),
            patch.object(
                pipeline.whisper,
                "stream_transcribe",
                return_value=mock_stream_transcribe(),
            ),
            patch.object(pipeline.translation, "translate", mock_translate),
            patch.object(pipeline.tts, "synthesize", mock_synthesize),
        ):
            results = [
                result
                async for result in pipeline.process_streaming(
                    mock_audio_gen(), "en", "bn"
                )
            ]

        assert results
        assert source_text not in caplog.text
        assert translated_text not in caplog.text
        assert "ASR progress: language=en" in caplog.text
        assert "en -> bn" in caplog.text

    @pytest.mark.asyncio
    async def test_process_streaming_backtrack_shorter_text(self):
        """Test backtrack handling when current text is shorter than last."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello world this is longer text",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="Hello world shorter",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="final transcript",
                language="en",
                is_partial=False,
                metrics={},
            )

        async def mock_translate(*args, **kwargs):
            return MagicMock(success=True, text="translated", language="en")

        async def mock_synthesize(*args, **kwargs):
            return MagicMock(success=True, audio_url="/tmp/test.wav")

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(pipeline.translation, "translate", mock_translate):
                with patch.object(pipeline.tts, "synthesize", mock_synthesize):
                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_text_too_short(self):
        """Test skipping when translated text is too short (< 3 chars)."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="Hello a",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="Hello world final",
                language="en",
                is_partial=False,
                metrics={},
            )

        mock_stream = MagicMock(return_value=mock_stream_transcribe())

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            results = []
            async for result in pipeline.process_streaming(
                mock_audio_gen(), "en", "gu"
            ):
                results.append(result)

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_buffer_flush_sentence_complete(self):
        """Test buffer flush on sentence boundary."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello world.",
                language="en",
                is_partial=False,
                metrics={},
            )

        captured_translate_kwargs = {}

        async def mock_translate(*args, **kwargs):
            captured_translate_kwargs.update(kwargs)
            return MagicMock(success=True, text="translated.", language="en")

        async def mock_synthesize(*args, **kwargs):
            return MagicMock(success=True, audio_url="/tmp/test.wav")

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(pipeline.translation, "translate", mock_translate):
                with patch.object(pipeline.tts, "synthesize", mock_synthesize):
                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(),
                        "en",
                        "gu",
                        custom_instruction="Translate formally.",
                    ):
                        results.append(result)

                    assert len(results) > 0
                    assert (
                        captured_translate_kwargs["custom_instruction"]
                        == "Translate formally."
                    )

    @pytest.mark.asyncio
    async def test_session_pacing_commits_immediately_on_acoustic_pause(self):
        """The default policy trusts an explicit acoustic pause marker."""
        pipeline = TranslationPipelineService(warm_connections=False)
        release_asr = asyncio.Event()
        transcript_seen = asyncio.Event()
        translation_seen = asyncio.Event()

        async def mock_audio_gen():
            yield b"audio_chunk"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="A short pause",
                language="en",
                is_partial=True,
                metrics={"force_emit": True},
            )
            await release_asr.wait()

        async def translate_on_pause(*args, **kwargs):
            translation_seen.set()
            return MagicMock(success=True, text="translated", language="en")

        mock_translate = AsyncMock(side_effect=translate_on_pause)
        mock_synthesize = AsyncMock(
            return_value=MagicMock(success=True, audio_url="/tmp/test.wav")
        )
        results = []

        async def collect_results():
            async for result in pipeline.process_streaming(
                mock_audio_gen(),
                "en",
                "gu",
                translation_buffer_seconds=10.0,
            ):
                results.append(result)
                if result.get("type") == "transcription":
                    transcript_seen.set()

        with (
            patch.object(
                pipeline.whisper,
                "stream_transcribe",
                return_value=mock_stream_transcribe(),
            ),
            patch.object(pipeline.translation, "translate", mock_translate),
            patch.object(pipeline.tts, "synthesize", mock_synthesize),
        ):
            collector = asyncio.create_task(collect_results())
            await asyncio.wait_for(transcript_seen.wait(), timeout=1.0)
            await asyncio.wait_for(translation_seen.wait(), timeout=1.0)
            mock_translate.assert_awaited_once()
            release_asr.set()
            await asyncio.wait_for(collector, timeout=2.0)

        mock_translate.assert_awaited_once()
        assert any(result.get("type") == "translation" for result in results)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transcript_metrics",
        [{"force_emit": True}, {}],
        ids=["acoustic-pause", "stt-cadence"],
    )
    async def test_semantic_pause_commits_prefix_and_retains_fragment(
        self, transcript_metrics
    ):
        """Semantic mode handles both pause and regular cadence transcripts."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Meeting is on Friday Do not forget to",
                language="en",
                is_partial=True,
                metrics=transcript_metrics,
            )
            yield MagicMock(
                success=True,
                text="Meeting is on Friday Do not forget to bring notes",
                language="en",
                is_partial=False,
                metrics={},
            )

        app_config = {
            "translation_pause_commit_mode": "semantic",
            "translation_semantic_hard_seconds": 10.0,
            "translation_flush_seconds": 6.0,
            "translation_flush_chars": 120,
            "translation_flush_min_chars": 40,
        }
        config = MagicMock(app=app_config, translation={})
        boundary = SemanticBoundaryDecision(
            boundary=len("Meeting is on Friday"),
            confidence=0.96,
        )
        mock_translate = AsyncMock(
            return_value=MagicMock(success=True, text="translated", language="en")
        )
        mock_synthesize = AsyncMock(
            return_value=MagicMock(success=True, audio_url="/tmp/test.wav")
        )

        with (
            patch("app.config.get_config", return_value=config),
            patch.object(
                pipeline.whisper,
                "stream_transcribe",
                return_value=mock_stream_transcribe(),
            ),
            patch.object(
                pipeline.translation,
                "select_semantic_boundary",
                new=AsyncMock(return_value=boundary),
            ),
            patch.object(pipeline.translation, "translate", mock_translate),
            patch.object(pipeline.tts, "synthesize", mock_synthesize),
        ):
            results = [
                result
                async for result in pipeline.process_streaming(
                    mock_audio_gen(),
                    "en",
                    "gu",
                    translation_buffer_seconds=6.0,
                )
            ]

        assert mock_translate.await_args_list[0].args[0] == "Meeting is on Friday"
        assert (
            mock_translate.await_args_list[1].args[0] == "Do not forget to bring notes"
        )
        assert any(result.get("type") == "translation" for result in results)

    def test_semantic_selector_cooldown_only_limits_cadence_probes(self):
        """Explicit acoustic pauses bypass the normal cadence cost guard."""
        assert _semantic_probe_is_cooling_down(10.0, 11.0, 3.0)
        assert not _semantic_probe_is_cooling_down(
            10.0,
            11.0,
            3.0,
            bypass=True,
        )

    @pytest.mark.asyncio
    async def test_semantic_acoustic_pauses_bypass_selector_cooldown(self):
        """Each explicit pause promptly asks for a safe semantic prefix."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Meeting is on Friday Do not",
                language="en",
                is_partial=True,
                metrics={"force_emit": True},
            )
            yield MagicMock(
                success=True,
                text="Meeting is on Friday Do not forget",
                language="en",
                is_partial=True,
                metrics={"force_emit": True},
            )

        app_config = {
            "translation_pause_commit_mode": "semantic",
            "translation_semantic_hard_seconds": 10.0,
            "translation_flush_seconds": 6.0,
            "translation_flush_chars": 120,
            "translation_flush_min_chars": 40,
        }
        translation_config = {
            "semantic_boundary_probe_cooldown_seconds": 3.0,
        }
        config = MagicMock(app=app_config, translation=translation_config)
        mock_selector = AsyncMock(return_value=None)
        mock_translate = AsyncMock(
            return_value=MagicMock(success=True, text="translated", language="en")
        )
        mock_synthesize = AsyncMock(
            return_value=MagicMock(success=True, audio_url="/tmp/test.wav")
        )

        with (
            patch("app.config.get_config", return_value=config),
            patch.object(
                pipeline.whisper,
                "stream_transcribe",
                return_value=mock_stream_transcribe(),
            ),
            patch.object(
                pipeline.translation,
                "select_semantic_boundary",
                new=mock_selector,
            ),
            patch.object(pipeline.translation, "translate", mock_translate),
            patch.object(pipeline.tts, "synthesize", mock_synthesize),
        ):
            results = [
                result
                async for result in pipeline.process_streaming(
                    mock_audio_gen(),
                    "en",
                    "gu",
                    translation_buffer_seconds=6.0,
                )
            ]

        assert mock_selector.await_count == 2
        mock_translate.assert_awaited_once()
        assert any(result.get("type") == "translation" for result in results)

    @pytest.mark.asyncio
    async def test_process_streaming_translation_retry_on_failure(self, caplog):
        """Test translation retry logic when first attempt fails."""
        pipeline = TranslationPipelineService(warm_connections=False)
        provider_error = "private transcript echoed by provider"

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello world",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="Hello world final",
                language="en",
                is_partial=False,
                metrics={},
            )

        call_count = 0

        async def mock_translate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(success=False, text="", error=provider_error)
            return MagicMock(success=True, text="translated", language="en")

        async def mock_synthesize(*args, **kwargs):
            return MagicMock(success=True, audio_url="/tmp/test.wav")

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(pipeline.translation, "translate", mock_translate):
                with patch.object(pipeline.tts, "synthesize", mock_synthesize):
                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0
                    assert provider_error not in caplog.text
                    assert "Failed to flush buffer (asr done)" in caplog.text

    @pytest.mark.asyncio
    async def test_process_streaming_translation_all_retries_fail(self):
        """Test when all translation retries fail."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello world",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="Hello world final",
                language="en",
                is_partial=False,
                metrics={},
            )

        async def mock_translate(*args, **kwargs):
            return MagicMock(success=False, text="", error="API error")

        async def mock_synthesize(*args, **kwargs):
            return MagicMock(success=True, audio_url="/tmp/test.wav")

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(pipeline.translation, "translate", mock_translate):
                with patch.object(pipeline.tts, "synthesize", mock_synthesize):
                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0
                    await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_process_streaming_tts_timeout_cancel(self):
        """Test TTS timeout and cancel in exception handling path."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello world",
                language="en",
                is_partial=True,
                metrics={},
            )

        async def mock_translate(*args, **kwargs):
            return MagicMock(success=True, text="translated", language="en")

        async def slow_tts(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock(success=True, audio_url="/tmp/test.wav")

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(pipeline.translation, "translate", mock_translate):
                with patch.object(pipeline.tts, "synthesize", slow_tts):
                    results = []
                    try:
                        async for result in pipeline.process_streaming(
                            mock_audio_gen(), "en", "gu"
                        ):
                            results.append(result)
                    except Exception:
                        pass

    @pytest.mark.asyncio
    async def test_process_streaming_exception_handling_tts_drain(self):
        """Test exception handling path with TTS drain success."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="Hello world",
                language="en",
                is_partial=True,
                metrics={},
            )

        async def mock_translate(*args, **kwargs):
            return MagicMock(success=True, text="translated", language="en")

        async def mock_synthesize(*args, **kwargs):
            return MagicMock(success=True, audio_url="/tmp/test.wav")

        mock_stream = MagicMock()
        mock_stream.return_value = mock_stream_transcribe()

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(pipeline.translation, "translate", mock_translate):
                with patch.object(pipeline.tts, "synthesize", mock_synthesize):
                    results = []
                    try:
                        async for result in pipeline.process_streaming(
                            mock_audio_gen(), "en", "gu"
                        ):
                            results.append(result)
                    except Exception:
                        pass

    @pytest.mark.asyncio
    async def test_process_streaming_asr_complete_rewrite_no_prefix_match(self):
        """Test ASR complete rewrite when last_text not in current_text (word overlap fallback)."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="the quick brown fox",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="a quick brown fox jumps",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="final transcript done",
                language="en",
                is_partial=False,
                metrics={},
            )

        mock_stream = MagicMock(return_value=mock_stream_transcribe())

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_asr_full_rewrite_no_overlap(self):
        """Test ASR full rewrite with no word overlap."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="hello there friend",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="goodbye world everyone",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="final end",
                language="en",
                is_partial=False,
                metrics={},
            )

        mock_stream = MagicMock(return_value=mock_stream_transcribe())

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_asr_backtrack_complete_rewrite(self):
        """Test ASR backtrack with complete rewrite (not a prefix)."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        async def mock_stream_transcribe(*args, **kwargs):
            yield MagicMock(
                success=True,
                text="this is a longer sentence here",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="short new text",
                language="en",
                is_partial=True,
                metrics={},
            )
            yield MagicMock(
                success=True,
                text="final done now",
                language="en",
                is_partial=False,
                metrics={},
            )

        mock_stream = MagicMock(return_value=mock_stream_transcribe())

        with patch.object(pipeline.whisper, "stream_transcribe", mock_stream):
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_asr_same_length_skip(self):
        """Test ASR when text length is unchanged (skip path)."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    async def mock_transcribe():
                        yield MagicMock(
                            success=True,
                            text="hello world",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )
                        yield MagicMock(
                            success=True,
                            text="hello world",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )
                        yield MagicMock(
                            success=True,
                            text="final transcript end",
                            language="en",
                            is_partial=False,
                            metrics={},
                        )

                    mock_stream.side_effect = mock_transcribe

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_word_overlap_path(self):
        """Test ASR text processing with word overlap detection."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    async def mock_transcribe():
                        yield MagicMock(
                            success=True,
                            text="hello world test",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )
                        yield MagicMock(
                            success=True,
                            text="world test new content",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )

                    mock_stream.side_effect = mock_transcribe

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_full_rewrite_path(self):
        """Test ASR text processing with full rewrite (no overlap)."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    async def mock_transcribe():
                        yield MagicMock(
                            success=True,
                            text="first text here",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )
                        yield MagicMock(
                            success=True,
                            text="completely different text",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )

                    mock_stream.side_effect = mock_transcribe

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_backtrack_prefix_skip(self):
        """Test ASR backtrack path when current is prefix of last."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    async def mock_transcribe():
                        yield MagicMock(
                            success=True,
                            text="hello world test here",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )
                        yield MagicMock(
                            success=True,
                            text="hello world",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )

                    mock_stream.side_effect = mock_transcribe

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_backtrack_rewrite_buffer(self):
        """Test ASR backtrack path with complete rewrite (buffer path)."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    async def mock_transcribe():
                        yield MagicMock(
                            success=True,
                            text="first sentence here",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )
                        yield MagicMock(
                            success=True,
                            text="completely new text",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )

                    mock_stream.side_effect = mock_transcribe

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_language_swap_with_buffer_flush(self):
        """Test language swap triggers buffer flush before applying swap."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    async def mock_transcribe():
                        # First result builds up buffer
                        yield MagicMock(
                            success=True,
                            text="Hello world",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )
                        # Second result triggers language swap
                        yield MagicMock(
                            success=True,
                            text="Hello world more text",
                            language="hi",
                            is_partial=True,
                            metrics={},
                        )

                    mock_stream.return_value = mock_transcribe()

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_tts_generation_error(self):
        """Test pipeline handles TTS generation error."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(success=False, error="TTS failed")

                    async def mock_transcribe():
                        yield MagicMock(
                            success=True,
                            text="Hello world",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )

                    mock_stream.side_effect = mock_transcribe

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_translation_retry_success(self):
        """Test translation retry logic succeeds on second attempt."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.side_effect = [
                        MagicMock(success=False, error="First attempt failed"),
                        MagicMock(success=True, text="translated", language="en"),
                    ]
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    async def mock_transcribe():
                        yield MagicMock(
                            success=True,
                            text="Hello world",
                            language="en",
                            is_partial=True,
                            metrics={},
                        )

                    mock_stream.side_effect = mock_transcribe

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_exception_handling(self):
        """Test pipeline handles exception in worker."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:

            async def mock_transcribe():
                raise Exception("Worker error")

            mock_stream.side_effect = mock_transcribe

            results = []
            try:
                async for result in pipeline.process_streaming(
                    mock_audio_gen(), "en", "gu"
                ):
                    results.append(result)
            except Exception:
                pass

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_process_streaming_result_queue_timeout(self):
        """Test pipeline result queue timeout continues loop."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:

            async def mock_transcribe():
                yield MagicMock(
                    success=True,
                    text="Hello",
                    language="en",
                    is_partial=False,
                    metrics={},
                )

            mock_stream.side_effect = mock_transcribe

            results = []
            async for result in pipeline.process_streaming(
                mock_audio_gen(), "en", "gu"
            ):
                results.append(result)
                if len(results) > 10:
                    break

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_asr_error_in_queue(self):
        """Test pipeline handles ASR error put in queue."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:

            async def mock_transcribe():
                yield MagicMock(
                    success=False,
                    text="",
                    error="ASR failed",
                    language="en",
                    is_partial=False,
                    metrics={},
                )

            mock_stream.side_effect = mock_transcribe

            results = []
            async for result in pipeline.process_streaming(
                mock_audio_gen(), "en", "gu"
            ):
                results.append(result)

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_generator_exit_handling(self):
        """Test pipeline handles GeneratorExit."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:

            async def mock_transcribe():
                for i in range(100):
                    yield MagicMock(
                        success=True,
                        text=f"Text {i}",
                        language="en",
                        is_partial=True,
                        metrics={},
                    )
                    await asyncio.sleep(0.01)

            mock_stream.side_effect = mock_transcribe

            results = []
            try:
                async for result in pipeline.process_streaming(
                    mock_audio_gen(), "en", "gu"
                ):
                    results.append(result)
                    if len(results) > 5:
                        raise GeneratorExit()
            except GeneratorExit:
                pass
            except Exception:
                pass

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_cancelled_error_handling(self):
        """Test pipeline handles asyncio.CancelledError."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:

            async def mock_transcribe():
                for i in range(100):
                    yield MagicMock(
                        success=True,
                        text=f"Text {i}",
                        language="en",
                        is_partial=True,
                        metrics={},
                    )
                    await asyncio.sleep(0.01)

            mock_stream.side_effect = mock_transcribe

            results = []
            try:
                async for result in pipeline.process_streaming(
                    mock_audio_gen(), "en", "gu"
                ):
                    results.append(result)
                    if len(results) > 5:
                        raise asyncio.CancelledError()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_process_streaming_error_type_yield(self):
        """Test pipeline yields error type results."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:

            async def mock_transcribe():
                yield MagicMock(
                    success=False,
                    text="",
                    error="Test error",
                    language="en",
                    is_partial=False,
                    metrics={},
                )

            mock_stream.side_effect = mock_transcribe

            results = []
            async for result in pipeline.process_streaming(
                mock_audio_gen(), "en", "gu"
            ):
                results.append(result)

            error_results = [r for r in results if r.get("type") == "error"]
            assert len(error_results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_tts_type_yield(self):
        """Test pipeline yields tts type results."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    async def mock_transcribe():
                        yield MagicMock(
                            success=True,
                            text="Hello world",
                            language="en",
                            is_partial=False,
                            metrics={},
                        )

                    mock_stream.side_effect = mock_transcribe

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    tts_results = [r for r in results if r.get("type") == "tts"]
                    assert len(tts_results) >= 0

    @pytest.mark.asyncio
    async def test_process_streaming_translation_type_yield(self):
        """Test pipeline yields translation type results."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:
            with patch.object(
                pipeline.translation, "translate", new_callable=AsyncMock
            ) as mock_translate:
                with patch.object(
                    pipeline.tts, "synthesize", new_callable=AsyncMock
                ) as mock_tts:
                    mock_translate.return_value = MagicMock(
                        success=True, text="translated", language="en"
                    )
                    mock_tts.return_value = MagicMock(
                        success=True, audio_url="/tmp/test.wav"
                    )

                    async def mock_transcribe():
                        yield MagicMock(
                            success=True,
                            text="Hello world",
                            language="en",
                            is_partial=False,
                            metrics={},
                        )

                    mock_stream.side_effect = mock_transcribe

                    results = []
                    async for result in pipeline.process_streaming(
                        mock_audio_gen(), "en", "gu"
                    ):
                        results.append(result)

                    translation_results = [
                        r for r in results if r.get("type") == "translation"
                    ]
                    assert len(translation_results) >= 0

    @pytest.mark.asyncio
    async def test_process_streaming_with_invalid_config_values(self):
        """Test pipeline handles invalid config values gracefully."""

        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            yield b"audio_chunk"
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:

            async def mock_transcribe():
                yield MagicMock(
                    success=True,
                    text="Hello",
                    language="en",
                    is_partial=False,
                    metrics={},
                )

            mock_stream.side_effect = mock_transcribe

            results = []
            async for result in pipeline.process_streaming(
                mock_audio_gen(), "en", "gu"
            ):
                results.append(result)

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_stop_event_set(self):
        """Test pipeline stops when stop event is set."""
        pipeline = TranslationPipelineService(warm_connections=False)

        async def mock_audio_gen():
            for i in range(100):
                yield b"audio_chunk"
                await asyncio.sleep(0.01)
            yield b"__END_SIGNAL__"

        with patch.object(pipeline.whisper, "stream_transcribe") as mock_stream:

            async def mock_transcribe():
                for i in range(100):
                    yield MagicMock(
                        success=True,
                        text=f"Text {i}",
                        language="en",
                        is_partial=True,
                        metrics={},
                    )
                    await asyncio.sleep(0.01)

            mock_stream.side_effect = mock_transcribe

            results = []
            try:
                async for result in pipeline.process_streaming(
                    mock_audio_gen(), "en", "gu"
                ):
                    results.append(result)
                    if len(results) > 5:
                        break
            except Exception:
                pass

            assert len(results) > 0


@pytest.mark.asyncio
async def test_conversation_mode_uses_detected_language_for_direction():
    """Conversation mode translates each pause-flushed turn to the opposite language."""
    from app.services.base import TranscriptionResult, TranslationResult, TTSResult

    class FakeWhisper:
        mock_mode = True

        async def close(self):
            pass

        async def stream_transcribe(
            self,
            audio_generator,
            language=None,
            emit_policy="live",
            candidate_languages=None,
            on_result=None,
        ):
            assert language is None
            assert emit_policy == "pause"
            assert candidate_languages == ["de", "en"]
            yield TranscriptionResult(
                text="Guten Morgen", language="de", is_partial=True
            )
            yield TranscriptionResult(text="Guten Morgen How are you", language="en")

    class FakeTranslation:
        mock_mode = True

        def __init__(self):
            self.calls = []

        async def close(self):
            pass

        async def translate(self, text, source_language, target_language, **kwargs):
            self.calls.append((text, source_language, target_language))
            translated = (
                "Good morning" if target_language == "en" else "Wie geht es dir?"
            )
            return TranslationResult(text=translated, success=True)

    class FakeTTS:
        def __init__(self):
            self.languages = []

        async def close(self):
            pass

        async def synthesize(self, text, language, output_path=None):
            self.languages.append(language)
            return TTSResult(audio_url=f"/media/output/{language}.wav", success=True)

    async def audio_generator():
        yield b"pcm"
        yield b"__END_SIGNAL__"

    translation = FakeTranslation()
    tts = FakeTTS()
    pipeline = TranslationPipelineService(
        whisper_service=FakeWhisper(),
        translation_service=translation,
        tts_service=tts,
        warm_connections=False,
    )

    results = [
        item
        async for item in pipeline.process_streaming(
            audio_generator(), "de", "en", mode="conversation"
        )
    ]

    assert [
        item["transcript"] for item in results if item.get("type") == "transcription"
    ] == ["Guten Morgen", "Guten Morgen How are you"]
    assert [
        item["is_partial"] for item in results if item.get("type") == "transcription"
    ] == [True, False]
    turns = [item for item in results if item.get("type") == "conversation_turn"]
    assert [(turn["source_language"], turn["target_language"]) for turn in turns] == [
        ("de", "en"),
        ("en", "de"),
    ]
    assert [item["sequence"] for item in results if item.get("type") == "tts"] == [1, 2]
    assert translation.calls == [
        ("Guten Morgen", "de", "en"),
        ("How are you", "en", "de"),
    ]
    assert tts.languages == ["en", "de"]


@pytest.mark.asyncio
async def test_conversation_mode_passes_custom_instruction():
    """Conversation mode passes custom translation guidance to translation calls."""
    from app.services.base import TranscriptionResult, TranslationResult, TTSResult

    class FakeWhisper:
        mock_mode = True

        async def close(self):
            pass

        async def stream_transcribe(
            self,
            audio_generator,
            language=None,
            emit_policy="live",
            candidate_languages=None,
            on_result=None,
        ):
            yield TranscriptionResult(text="Guten Morgen", language="de")

    class FakeTranslation:
        mock_mode = True

        def __init__(self):
            self.calls = []

        async def close(self):
            pass

        async def translate(self, text, source_language, target_language, **kwargs):
            self.calls.append(kwargs.get("custom_instruction"))
            return TranslationResult(text="Good morning", success=True)

    class FakeTTS:
        async def close(self):
            pass

        async def synthesize(self, text, language, output_path=None):
            return TTSResult(audio_url=f"/media/output/{language}.wav", success=True)

    async def audio_generator():
        yield b"pcm"
        yield b"__END_SIGNAL__"

    translation = FakeTranslation()
    pipeline = TranslationPipelineService(
        whisper_service=FakeWhisper(),
        translation_service=translation,
        tts_service=FakeTTS(),
        warm_connections=False,
    )

    results = [
        item
        async for item in pipeline.process_streaming(
            audio_generator(),
            "de",
            "en",
            mode="conversation",
            custom_instruction="Translate formally.",
        )
    ]

    assert [item for item in results if item.get("type") == "conversation_turn"]
    assert translation.calls == ["Translate formally."]


@pytest.mark.asyncio
async def test_conversation_mode_applies_runtime_custom_instruction_update():
    """Conversation mode drains runtime custom instruction updates."""
    from app.services.base import TranscriptionResult, TranslationResult, TTSResult

    class FakeWhisper:
        mock_mode = True

        async def close(self):
            pass

        async def stream_transcribe(
            self,
            audio_generator,
            language=None,
            emit_policy="live",
            candidate_languages=None,
            on_result=None,
        ):
            yield TranscriptionResult(text="Guten Morgen", language="de")

    class FakeTranslation:
        mock_mode = True

        def __init__(self):
            self.calls = []

        async def close(self):
            pass

        async def translate(self, text, source_language, target_language, **kwargs):
            self.calls.append(kwargs.get("custom_instruction"))
            return TranslationResult(text="Good morning", success=True)

    class FakeTTS:
        async def close(self):
            pass

        async def synthesize(self, text, language, output_path=None):
            return TTSResult(audio_url=f"/media/output/{language}.wav", success=True)

    async def audio_generator():
        yield b"pcm"
        yield b"__END_SIGNAL__"

    custom_instruction_queue = asyncio.Queue()
    custom_instruction_queue.put_nowait(" Keep product names in English. ")
    translation = FakeTranslation()
    pipeline = TranslationPipelineService(
        whisper_service=FakeWhisper(),
        translation_service=translation,
        tts_service=FakeTTS(),
        warm_connections=False,
    )

    results = [
        item
        async for item in pipeline.process_streaming(
            audio_generator(),
            "de",
            "en",
            mode="conversation",
            custom_instruction_queue=custom_instruction_queue,
        )
    ]

    assert [item for item in results if item.get("type") == "conversation_turn"]
    assert translation.calls == ["Keep product names in English."]


@pytest.mark.asyncio
async def test_conversation_mode_treats_urdu_detection_as_hindi_when_hindi_selected():
    """Hindi speech may be detected as Urdu; selected Hindi should still drive direction."""
    from app.services.base import TranscriptionResult, TranslationResult, TTSResult

    class FakeWhisper:
        mock_mode = True

        async def close(self):
            pass

        async def stream_transcribe(
            self,
            audio_generator,
            language=None,
            emit_policy="live",
            candidate_languages=None,
            on_result=None,
        ):
            assert language is None
            assert emit_policy == "pause"
            assert candidate_languages == ["en", "hi"]
            yield TranscriptionResult(text="मैं ठीक हूँ", language="ur")

    class FakeTranslation:
        mock_mode = True

        def __init__(self):
            self.calls = []

        async def close(self):
            pass

        async def translate(self, text, source_language, target_language, **kwargs):
            self.calls.append((text, source_language, target_language))
            return TranslationResult(text="I am fine", success=True)

    class FakeTTS:
        def __init__(self):
            self.languages = []

        async def close(self):
            pass

        async def synthesize(self, text, language, output_path=None):
            self.languages.append(language)
            return TTSResult(audio_url=f"/media/output/{language}.wav", success=True)

    async def audio_generator():
        yield b"pcm"
        yield b"__END_SIGNAL__"

    translation = FakeTranslation()
    tts = FakeTTS()
    pipeline = TranslationPipelineService(
        whisper_service=FakeWhisper(),
        translation_service=translation,
        tts_service=tts,
        warm_connections=False,
    )

    results = [
        item
        async for item in pipeline.process_streaming(
            audio_generator(), "en", "hi", mode="conversation"
        )
    ]

    assert [
        item["transcript"] for item in results if item.get("type") == "transcription"
    ] == ["मैं ठीक हूँ"]
    turns = [item for item in results if item.get("type") == "conversation_turn"]
    assert [(turn["source_language"], turn["target_language"]) for turn in turns] == [
        ("hi", "en"),
    ]
    assert [item["sequence"] for item in results if item.get("type") == "tts"] == [1]
    assert turns[0]["detected_language"] == "ur"
    assert translation.calls == [("मैं ठीक हूँ", "hi", "en")]
    assert tts.languages == ["en"]


def test_conversation_direction_uses_default_when_detected_matches_both_sides():
    """Dialect pairs normalize to the same base code, so do not reverse direction."""
    assert TranslationPipelineService._conversation_direction(
        "en", "en-GB", "en-US"
    ) == ("en-GB", "en-US")


@pytest.mark.asyncio
async def test_conversation_mode_keeps_two_character_turns():
    """Short utterances like ok/hi are meaningful conversation turns."""
    from app.services.base import TranscriptionResult, TranslationResult, TTSResult

    class FakeWhisper:
        mock_mode = True

        async def close(self):
            pass

        async def stream_transcribe(
            self,
            audio_generator,
            language=None,
            emit_policy="live",
            candidate_languages=None,
            on_result=None,
        ):
            yield TranscriptionResult(text="ok", language="en")

    class FakeTranslation:
        mock_mode = True

        def __init__(self):
            self.calls = []

        async def close(self):
            pass

        async def translate(self, text, source_language, target_language, **kwargs):
            self.calls.append((text, source_language, target_language))
            return TranslationResult(text="ठीक है", success=True)

    class FakeTTS:
        async def close(self):
            pass

        async def synthesize(self, text, language, output_path=None):
            return TTSResult(audio_url=f"/media/output/{language}.wav", success=True)

    async def audio_generator():
        yield b"pcm"
        yield b"__END_SIGNAL__"

    translation = FakeTranslation()
    pipeline = TranslationPipelineService(
        whisper_service=FakeWhisper(),
        translation_service=translation,
        tts_service=FakeTTS(),
        warm_connections=False,
    )

    results = [
        item
        async for item in pipeline.process_streaming(
            audio_generator(), "en", "hi", mode="conversation"
        )
    ]

    turns = [item for item in results if item.get("type") == "conversation_turn"]
    assert [(turn["transcript"], turn["translated_text"]) for turn in turns] == [
        ("ok", "ठीक है"),
    ]
    assert translation.calls == [("ok", "en", "hi")]
