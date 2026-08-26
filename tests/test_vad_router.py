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
