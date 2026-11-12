"""Installed voice selection must remain request-local and language-bound."""

from unittest.mock import AsyncMock

import pytest

from app.services.base import TTSResult
from app.services.tts_service import TTSService


@pytest.fixture
def service():
    instance = TTSService.__new__(TTSService)
    instance.enabled = True
    instance._get_provider_for_language = lambda _language: "piper"
    instance._fetch_voices = AsyncMock(
        return_value={
            "de_DE-thorsten-medium.onnx": {},
            "en_US-lessac-medium.onnx": {},
            "de_DE-eva_k-x_low": {},
        }
    )
    instance._piper_synthesize = AsyncMock(return_value=TTSResult(success=True))
    return instance


@pytest.mark.asyncio
async def test_only_installed_language_voices_are_offered(service):
    voices = await service.available_text_voices("de")
    assert {voice["id"] for voice in voices} == {
        "de_DE-thorsten-medium",
        "de_DE-eva_k-x_low",
    }


@pytest.mark.asyncio
async def test_selected_voice_is_passed_without_mutating_defaults(service):
    result = await service.synthesize_selected_voice(
        "Hallo", "de", "de_DE-thorsten-medium"
    )
    assert result.success
    service._piper_synthesize.assert_awaited_once_with(
        "Hallo", "de", voice_override="de_DE-thorsten-medium"
    )
    assert "voice" not in service.__dict__


@pytest.mark.asyncio
async def test_foreign_or_uninstalled_voice_cannot_generate_audio(service):
    for voice in ["en_US-lessac-medium", "../../secret", "unknown"]:
        result = await service.synthesize_selected_voice("Hallo", "de", voice)
        assert not result.success
    service._piper_synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_providers_keep_configured_default(service):
    service._get_provider_for_language = lambda _language: "supertonic"
    assert await service.available_text_voices("de") == []
    service._fetch_voices.assert_not_awaited()
