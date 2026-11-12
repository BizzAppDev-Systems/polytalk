"""WebSocket routing regressions for shared VAD profiles."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.audio_vad import VadMode


def test_websocket_rejects_unknown_audio_input_type() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        with client.websocket_connect(
            "/api/ws/translate?source_language=en&target_language=de&input_type=system"
        ) as websocket:
            assert websocket.receive_json() == {
                "type": "error",
                "error": "input_type must be microphone or share",
            }


def test_active_share_vad_is_announced_and_propagated_to_pipeline() -> None:
    pipeline = MagicMock()
    pipeline.warm_connections = AsyncMock()
    pipeline.vad.effective_mode.return_value = VadMode.ACTIVE
    captured_kwargs = {}

    async def process_streaming(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"type": "complete"}

    pipeline.process_streaming = process_streaming

    with (
        patch("app.routers.api.get_pipeline_service", return_value=pipeline),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        with client.websocket_connect(
            "/api/ws/translate?source_language=en&target_language=de&input_type=share"
        ) as websocket:
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message.get("type") == "complete":
                    break

    assert {
        "type": "audio_segmentation",
        "mode": "active",
        "input_type": "share",
        "continuous_audio": True,
    } in messages
    assert captured_kwargs["input_type"] == "share"


def test_legacy_numeric_pipeline_mode_defaults_to_translation_tts() -> None:
    """Legacy numeric values are not part of the named API contract."""
    pipeline = MagicMock()
    pipeline.warm_connections = AsyncMock()
    captured_kwargs = {}

    async def process_streaming(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"type": "complete"}

    pipeline.process_streaming = process_streaming

    with (
        patch("app.routers.api.get_pipeline_service", return_value=pipeline),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        with client.websocket_connect(
            "/api/ws/translate?source_language=en&target_language=de&input_type=share&pipeline_mode=-1"
        ) as websocket:
            while websocket.receive_json().get("type") != "complete":
                pass

    assert captured_kwargs["translation_enabled"] is True
    assert captured_kwargs["tts_enabled"] is True


def test_unknown_pipeline_mode_defaults_to_translation_tts() -> None:
    """Unknown values fall back to the full named mode."""
    pipeline = MagicMock()
    pipeline.warm_connections = AsyncMock()
    captured_kwargs = {}

    async def process_streaming(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"type": "complete"}

    pipeline.process_streaming = process_streaming

    with (
        patch("app.routers.api.get_pipeline_service", return_value=pipeline),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        with client.websocket_connect(
            "/api/ws/translate?source_language=en&target_language=de&input_type=share&pipeline_mode=99"
        ) as websocket:
            while websocket.receive_json().get("type") != "complete":
                pass

    assert captured_kwargs["translation_enabled"] is True
    assert captured_kwargs["tts_enabled"] is True


def test_pipeline_mode_valid_values_pass_through() -> None:
    """Named pipeline modes map to their declared capabilities."""
    for mode, expect_translation, expect_tts in [
        ("transcript", False, False),
        ("translation", True, False),
        ("translation_tts", True, True),
    ]:
        pipeline = MagicMock()
        pipeline.warm_connections = AsyncMock()
        captured_kwargs = {}

        async def process_streaming(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield {"type": "complete"}

        pipeline.process_streaming = process_streaming

        with (
            patch("app.routers.api.get_pipeline_service", return_value=pipeline),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            with client.websocket_connect(
                f"/api/ws/translate?source_language=en&target_language=de&input_type=share&pipeline_mode={mode}"
            ) as websocket:
                while websocket.receive_json().get("type") != "complete":
                    pass

        assert captured_kwargs["translation_enabled"] is expect_translation
        assert captured_kwargs["tts_enabled"] is expect_tts


def test_pipeline_mode_applies_in_microphone_mode() -> None:
    """Named result modes apply to microphone input as well as shared audio."""
    pipeline = MagicMock()
    pipeline.warm_connections = AsyncMock()
    captured_kwargs = {}

    async def process_streaming(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"type": "complete"}

    pipeline.process_streaming = process_streaming

    with (
        patch("app.routers.api.get_pipeline_service", return_value=pipeline),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        with client.websocket_connect(
            "/api/ws/translate?source_language=en&target_language=de&input_type=microphone&pipeline_mode=transcript"
        ) as websocket:
            while websocket.receive_json().get("type") != "complete":
                pass

    assert captured_kwargs["translation_enabled"] is False
    assert captured_kwargs["tts_enabled"] is False
    assert captured_kwargs["pipeline_mode_queue"] is not None


def test_pipeline_mode_invalid_string_defaults_to_translation_tts() -> None:
    """Invalid pipeline_mode strings default to translation_tts."""
    pipeline = MagicMock()
    pipeline.warm_connections = AsyncMock()
    captured_kwargs = {}

    async def process_streaming(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"type": "complete"}

    pipeline.process_streaming = process_streaming

    with (
        patch("app.routers.api.get_pipeline_service", return_value=pipeline),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        with client.websocket_connect(
            "/api/ws/translate?source_language=en&target_language=de&input_type=share&pipeline_mode=abc"
        ) as websocket:
            while websocket.receive_json().get("type") != "complete":
                pass

    assert captured_kwargs["translation_enabled"] is True
    assert captured_kwargs["tts_enabled"] is True
