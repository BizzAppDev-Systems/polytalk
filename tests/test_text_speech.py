"""Tests for selected-text translation and transient speech audio."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.base import TranslationResult, TTSResult
from app.services.pipeline_service import TranslationPipelineService


def build_pipeline(tmp_path: Path) -> TranslationPipelineService:
    """Build a pipeline with mocked providers and an isolated media directory."""
    translation = AsyncMock()
    tts = AsyncMock()
    pipeline = TranslationPipelineService(
        translation_service=translation,
        tts_service=tts,
        warm_connections=False,
    )
    pipeline.media_dir = tmp_path
    return pipeline


@pytest.mark.asyncio
async def test_translate_text_to_speech_returns_and_deletes_audio(tmp_path):
    """Successful audio is loaded into memory and its artifact is removed."""
    pipeline = build_pipeline(tmp_path)
    audio_path = tmp_path / "sentence.wav"
    audio_path.write_bytes(b"wave-audio")
    pipeline.translation.translate.return_value = TranslationResult(
        text="Bonjour", success=True
    )
    pipeline.tts.synthesize.return_value = TTSResult(
        audio_path=audio_path,
        audio_url="/media/output/sentence.wav",
        duration=1.5,
        success=True,
    )

    result = await pipeline.translate_text_to_speech("Hello", "auto", "fr")

    assert result.success is True
    assert result.audio == b"wave-audio"
    assert result.media_type == "audio/wav"
    assert result.duration == 1.5
    assert not audio_path.exists()
    pipeline.translation.translate.assert_awaited_once_with(
        "Hello", "auto", "fr", custom_translation_routing=False
    )
    pipeline.tts.synthesize.assert_awaited_once_with("Bonjour", "fr", None)


@pytest.mark.asyncio
async def test_translate_text_to_speech_stops_after_translation_failure(tmp_path):
    """TTS is not invoked when translation fails."""
    pipeline = build_pipeline(tmp_path)
    pipeline.translation.translate.return_value = TranslationResult(
        text="", success=False, error="provider unavailable"
    )

    result = await pipeline.translate_text_to_speech("Hello", "auto", "fr")

    assert result.success is False
    pipeline.tts.synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_translate_text_to_speech_rejects_artifact_outside_media(tmp_path):
    """A compromised provider result cannot make the pipeline read arbitrary files."""
    pipeline = build_pipeline(tmp_path / "media")
    pipeline.media_dir.mkdir()
    outside_path = tmp_path / "outside.wav"
    outside_path.write_bytes(b"not-audio")
    pipeline.translation.translate.return_value = TranslationResult(
        text="Bonjour", success=True
    )
    pipeline.tts.synthesize.return_value = TTSResult(
        audio_path=outside_path,
        success=True,
    )

    result = await pipeline.translate_text_to_speech("Hello", "auto", "fr")

    assert result.success is False
    assert outside_path.exists()


@pytest.mark.asyncio
async def test_translate_text_to_speech_deletes_empty_artifact(tmp_path):
    """Empty provider audio fails safely and is still removed."""
    pipeline = build_pipeline(tmp_path)
    audio_path = tmp_path / "empty.mp3"
    audio_path.touch()
    pipeline.translation.translate.return_value = TranslationResult(
        text="Bonjour", success=True
    )
    pipeline.tts.synthesize.return_value = TTSResult(
        audio_path=audio_path,
        success=True,
    )

    result = await pipeline.translate_text_to_speech("Hello", "auto", "fr")

    assert result.success is False
    assert not audio_path.exists()
