"""Cross-service regressions for the CE-to-STT VAD protocol."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import websockets
from fastapi.testclient import TestClient

from app.services.audio_vad import END_SIGNAL, VAD_UTTERANCE_BOUNDARY
from app.services.whisper_service import WhisperService


def load_stt_main_module(env: dict[str, str]):
    """Load faster-whisper worker with its heavyweight dependency stubbed."""
    module_name = "polytalk_stt_main_vad_protocol_tests_" + "_".join(
        f"{key}_{value}" for key, value in sorted(env.items())
    )
    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = object
    sys.modules.setdefault("faster_whisper", faster_whisper_stub)

    module_path = Path(__file__).parent.parent / "stt" / "app" / "main.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec and spec.loader
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


def test_faster_whisper_upstream_vad_bypasses_job_rms_rejection(monkeypatch):
    worker = load_stt_main_module({"STT_SILENCE_RMS_THRESHOLD": "0.5"})
    captured = []

    def fake_transcribe(model, audio, language, task, candidate_languages):
        captured.append(audio)
        return "quiet word", True, "en"

    monkeypatch.setattr(worker, "_transcribe_audio", fake_transcribe)
    job = worker.TranscribeJob(
        sequence=0,
        audio_bytes=(1).to_bytes(2, "little", signed=True) * 512,
        language="en",
        candidate_languages=(),
        task="transcribe",
        queued_at=0.0,
        queue_depth_at_enqueue=0,
        upstream_vad=True,
        finalization_reason="vad_pause",
        force_emit=True,
    )

    result = worker._process_transcribe_job(object(), job)

    assert captured
    assert result.skipped_silence is False
    assert result.transcript == "quiet word"
    assert worker._result_metrics(result)["finalization_reason"] == "vad_pause"


def test_faster_whisper_accepts_quiet_upstream_audio_and_vad_flush():
    worker = load_stt_main_module(
        {
            "STT_SAMPLE_RATE": "10",
            "STT_SAMPLE_WIDTH_BYTES": "2",
            "STT_STREAM_CHUNK_SECONDS": "10",
            "STT_EMIT_MIN_CHARS": "1",
            "STT_TRANSCRIBE_WORKERS": "1",
            "STT_SILENCE_RMS_THRESHOLD": "0.5",
        }
    )
    captured_jobs = []

    def fake_process(model, job):
        captured_jobs.append(job)
        return worker.TranscribeResult(
            sequence=job.sequence,
            transcript="quiet word",
            has_speech=True,
            detected_language="en",
            force_emit=job.force_emit,
            finalization_reason=job.finalization_reason,
        )

    worker._get_model = lambda: object()
    worker._process_transcribe_job = fake_process
    quiet_audio = (1).to_bytes(2, "little", signed=True) * 4

    with TestClient(worker.app) as client:
        with client.websocket_connect("/v1/stream/transcriptions") as websocket:
            websocket.send_text(json.dumps({"input_segmentation": "upstream_vad"}))
            ack = websocket.receive_json()
            websocket.send_bytes(quiet_audio)
            websocket.send_text(json.dumps({"type": "flush", "reason": "vad_pause"}))
            result = websocket.receive_json()
            websocket.send_text(json.dumps({"type": "end"}))

    assert ack == {
        "type": "input_segmentation_ack",
        "input_segmentation": "upstream_vad",
    }
    assert len(captured_jobs) == 1
    assert captured_jobs[0].audio_bytes == quiet_audio
    assert captured_jobs[0].upstream_vad is True
    assert result["metrics"]["force_emit"] is True
    assert result["metrics"]["finalization_reason"] == "vad_pause"


@pytest.mark.asyncio
async def test_specialized_stt_accepts_quiet_upstream_audio_and_vad_flush(
    monkeypatch,
):
    monkeypatch.setenv("STT_PROVIDER", "parakeet")
    monkeypatch.setenv("STT_PROVIDER_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    monkeypatch.setenv("STT_SILENCE_RMS_THRESHOLD", "0.5")
    worker = importlib.import_module("stt_providers.app.main")
    worker = importlib.reload(worker)
    captured_audio = []

    def fake_transcribe(audio, language):
        captured_audio.append(audio)
        return "quiet word", language or "en"

    monkeypatch.setattr(worker, "_transcribe", fake_transcribe)
    quiet_audio = (1).to_bytes(2, "little", signed=True) * 512

    class FakeWebSocket:
        def __init__(self):
            self.messages = [
                {"text": json.dumps({"input_segmentation": "upstream_vad"})},
                {"bytes": quiet_audio},
                {"text": json.dumps({"type": "flush", "reason": "vad_pause"})},
                {"type": "websocket.disconnect"},
            ]
            self.sent = []

        async def accept(self):
            pass

        async def receive(self):
            return self.messages.pop(0)

        async def send_json(self, payload):
            self.sent.append(payload)

        async def close(self):
            pass

    websocket = FakeWebSocket()
    await worker.stream_transcriptions(websocket)

    assert websocket.sent[0] == {
        "type": "input_segmentation_ack",
        "input_segmentation": "upstream_vad",
    }
    assert captured_audio == [quiet_audio]
    assert websocket.sent[1]["metrics"]["force_emit"] is True
    assert websocket.sent[1]["metrics"]["finalization_reason"] == "vad_pause"


@pytest.mark.asyncio
async def test_ce_stt_client_sends_segmentation_and_vad_flush_reason():
    class FakeConfig:
        whisper = {
            "enabled": True,
            "mock_mode": False,
            "base_url": "http://stt:8000",
            "ws_endpoint": "/v1/stream/transcriptions",
            "max_reconnect_attempts": 1,
        }

    service = WhisperService(FakeConfig())
    sent: list[bytes | str] = []
    end_sent = asyncio.Event()

    async def audio_source():
        yield b"speech"
        yield VAD_UTTERANCE_BOUNDARY
        yield END_SIGNAL

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
                audio_source(),
                language="en",
                input_segmentation="upstream_vad",
            )
        ]

    controls = [json.loads(item) for item in sent if isinstance(item, str)]
    assert controls[0] == {
        "language": "en",
        "emit_policy": "live",
        "input_segmentation": "upstream_vad",
    }
    assert {"type": "flush", "reason": "vad_pause"} in controls
    assert controls[-1] == {"type": "end"}
    assert results == []
