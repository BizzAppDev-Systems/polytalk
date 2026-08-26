"""Focused tests for specialized STT and TTS provider routing."""

import asyncio
import importlib
import json
import re
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import websockets
import yaml

from app.services.base import TranscriptionResult, TTSResult
from app.services.tts_service import TTSService
from app.services.whisper_service import WhisperService, _ReplayableAudioStream


class FakeConfig:
    def __init__(self, whisper):
        self.whisper = whisper


def stt_config():
    return {
        "enabled": True,
        "mock_mode": False,
        "base_url": "http://whisper:8000",
        "ws_endpoint": "/v1/stream/transcriptions",
        "routing_enabled": True,
        "fallback_provider": "whisper",
        "providers": {
            "whisper": {"enabled": True, "base_url": "http://whisper:8000"},
            "indicconformer": {
                "enabled": True,
                "base_url": "http://indic:8000",
            },
            "parakeet": {"enabled": True, "base_url": "http://parakeet:8000"},
            "sensevoice": {"enabled": True, "base_url": "http://sense:8000"},
        },
        "routing": [
            {"priority": 10, "provider": "indicconformer", "languages": ["hi", "gu"]},
            {"priority": 20, "provider": "sensevoice", "languages": ["zh", "ja"]},
            {"priority": 30, "provider": "parakeet", "languages": ["de", "fr"]},
        ],
    }


def test_stt_explicit_language_routing_and_unknown_fallback():
    service = WhisperService(FakeConfig(stt_config()))

    assert service._get_provider_for_language("hi-IN") == "indicconformer"
    assert service._get_provider_for_language("de-DE") == "parakeet"
    assert service._get_provider_for_language("zh-CN") == "sensevoice"
    assert service._get_provider_for_language("sw") == "whisper"
    assert service._get_provider_for_language(None) == "whisper"


def test_production_routing_sends_english_to_parakeet():
    config = yaml.safe_load(
        Path("config/config.yaml.example").read_text(encoding="utf-8")
    )
    parakeet_rule = next(
        rule for rule in config["whisper"]["routing"] if rule["provider"] == "parakeet"
    )

    assert "en" in parakeet_rule["languages"]


def test_stt_unresolved_replay_config_uses_safe_default():
    config = stt_config()
    config["fallback_replay_max_seconds"] = "${STT_FALLBACK_REPLAY_MAX_SECONDS}"
    config["sample_rate"] = "${STT_SAMPLE_RATE}"
    service = WhisperService(FakeConfig(config))

    assert service.fallback_replay_max_bytes == 30 * 16000 * 2


def test_stt_disabled_specialized_provider_uses_fallback():
    config = stt_config()
    config["providers"]["indicconformer"]["enabled"] = False
    service = WhisperService(FakeConfig(config))

    assert service._get_provider_for_language("hi") == "whisper"


@pytest.mark.asyncio
async def test_stt_replays_buffer_to_fallback_before_any_visible_result():
    service = WhisperService(FakeConfig(stt_config()))
    calls = []
    fallback_metadata = []

    async def audio():
        yield b"one"
        yield b"two"
        yield b"__END_SIGNAL__"

    async def stream(provider, audio_generator, *_args, **_kwargs):
        chunks = []
        if provider == "indicconformer":
            chunks.append(await anext(audio_generator))
            calls.append((provider, chunks))
            yield TranscriptionResult(text="", success=False, error="worker down")
            return
        async for chunk in audio_generator:
            chunks.append(chunk)
        calls.append((provider, chunks))
        fallback_metadata.append(_kwargs)
        yield TranscriptionResult(text="नमस्ते", language="hi")

    service._stream_with_provider = stream
    results = [
        result async for result in service.stream_transcribe(audio(), language="hi")
    ]

    assert [result.text for result in results] == ["नमस्ते"]
    assert calls == [
        ("indicconformer", [b"one"]),
        ("whisper", [b"one", b"two", b"__END_SIGNAL__"]),
    ]
    assert fallback_metadata == [
        {
            "fallback_reason": "worker down",
            "fallback_provider": "indicconformer",
        }
    ]


@pytest.mark.asyncio
async def test_stt_sends_initial_pacing_and_receives_final_result_before_close():
    """Initial pacing is sent before audio and end flushing preserves final text."""
    service = WhisperService(FakeConfig(stt_config()))
    end_received = asyncio.Event()

    async def audio():
        yield b"voice"
        yield b"__END_SIGNAL__"

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.response_sent = False
            self.closed = False

        async def send(self, payload):
            self.sent.append(payload)
            if payload == json.dumps({"type": "end"}):
                end_received.set()

        async def recv(self):
            await end_received.wait()
            if not self.response_sent:
                self.response_sent = True
                return json.dumps(
                    {
                        "text": "final transcript",
                        "is_final": True,
                        "language": "en",
                    }
                )
            raise websockets.exceptions.ConnectionClosedOK(None, None)

        async def close(self):
            self.closed = True

    websocket = FakeWebSocket()
    with patch(
        "app.services.whisper_service.websockets.connect",
        new=AsyncMock(return_value=websocket),
    ):
        results = [
            result
            async for result in service._real_stream_transcribe(
                audio(),
                language="en",
                emit_interval_seconds=1.5,
                provider="whisper",
                provider_config=service._get_provider_config("whisper"),
            )
        ]

    initial_control = json.loads(websocket.sent[0])
    assert initial_control["language"] == "en"
    assert initial_control["emit_interval_seconds"] == 1.5
    assert [result.text for result in results] == ["final transcript"]
    assert websocket.closed is True


@pytest.mark.asyncio
async def test_stt_fallback_replay_buffer_is_bounded():
    async def audio():
        for chunk in (b"one", b"two", b"three"):
            yield chunk

    replayable = _ReplayableAudioStream(audio(), max_buffer_bytes=8)
    assert [chunk async for chunk in replayable.primary()] == [
        b"one",
        b"two",
        b"three",
    ]
    assert replayable.buffered_bytes <= 8
    assert [chunk async for chunk in replayable.replay()] == [b"two", b"three"]


@pytest.mark.asyncio
async def test_stt_releases_replay_audio_after_visible_result():
    async def audio():
        yield b"sensitive audio"

    replayable = _ReplayableAudioStream(audio(), max_buffer_bytes=32000)
    primary = replayable.primary()
    assert await anext(primary) == b"sensitive audio"
    replayable.disable_replay()

    assert replayable.buffered_bytes == 0
    assert list(replayable.buffer) == []


@pytest.mark.asyncio
async def test_indic_parler_response_is_saved_with_voice_controls(tmp_path):
    audio = b"RIFF-indic-parler-wav"
    config = {
        "enabled": True,
        "mock_mode": False,
        "provider": "piper",
        "fallback_provider": "piper",
        "base_url": "http://piper:5000",
        "language_providers": {"mr": "indic_parler", "gu": "indic_parler"},
        "providers": {
            "indic_parler": {
                "base_url": "http://indic-parler:7790",
                "gender": "female",
                "voice": "auto",
                "pace": "moderate",
                "paces": {"gu": "slightly fast"},
                "speed": 1.0,
                "speeds": {"gu": 1.15},
                "timeout_seconds": 180,
            }
        },
    }
    with patch("app.services.tts_service.get_config") as get_config:
        get_config.return_value.tts = config
        get_config.return_value.app = {}
        get_config.return_value.media_output_dir = tmp_path
        service = TTSService()
        response = httpx.Response(
            200,
            content=audio,
            request=httpx.Request("POST", "http://indic-parler:7790/v1/tts"),
        )
        service._http_client.post = AsyncMock(return_value=response)

        output_path = tmp_path / "marathi.wav"
        result = await service.synthesize("तुम्ही कसे आहात?", "mr-IN", output_path)

        assert result.success is True
        assert result.audio_path == output_path
        assert output_path.read_bytes() == audio
        request = service._http_client.post.await_args
        assert request.args[0] == "http://indic-parler:7790/v1/tts"
        assert request.kwargs["timeout"] == 180.0
        assert request.kwargs["json"] == {
            "text": "तुम्ही कसे आहात?",
            "lang": "mr",
            "gender": "female",
            "voice": "auto",
            "pace": "moderate",
            "speed": 1.0,
        }

        gujarati_path = tmp_path / "gujarati.wav"
        gujarati = await service.synthesize("તમે કેમ છો?", "gu-IN", gujarati_path)
        assert gujarati.success is True
        gujarati_payload = service._http_client.post.await_args.kwargs["json"]
        assert gujarati_payload["pace"] == "slightly fast"
        assert gujarati_payload["speed"] == 1.15
        await service.close()


def test_current_indic_targets_route_to_indic_parler():
    config = yaml.safe_load(
        Path("config/config.yaml.example").read_text(encoding="utf-8")
    )
    routes = config["tts"]["language_providers"]
    for language in {
        "as",
        "bn",
        "brx",
        "gu",
        "hi",
        "kn",
        "ml",
        "mni",
        "mr",
        "or",
        "pa",
        "ta",
        "te",
        "ur",
    }:
        assert routes[language] == "indic_parler"
    assert "raj" not in routes


@pytest.mark.asyncio
async def test_tts_specialized_failure_calls_configured_fallback(tmp_path):
    config = {
        "enabled": True,
        "mock_mode": False,
        "provider": "piper",
        "fallback_provider": "piper",
        "base_url": "http://piper:5000",
        "language_providers": {"hi": "indic_parler"},
    }
    with patch("app.services.tts_service.get_config") as get_config:
        get_config.return_value.tts = config
        get_config.return_value.app = {}
        get_config.return_value.media_output_dir = tmp_path
        service = TTSService()
        service._synthesize_with_provider = AsyncMock(
            side_effect=[
                TTSResult(success=False, error="indic parler unavailable"),
                TTSResult(audio_url="/media/output/fallback.wav", success=True),
            ]
        )

        result = await service.synthesize("नमस्ते", "hi")

        assert result.success is True
        assert [
            call.args[0] for call in service._synthesize_with_provider.await_args_list
        ] == ["indic_parler", "piper"]
        await service.close()


INDIC_UI_LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


def test_indic_languages_are_enabled_in_both_ui_selectors():
    template = Path("app/templates/index.html").read_text(encoding="utf-8")
    for selector_id in ("source-lang", "target-lang"):
        match = re.search(
            rf'<select id="{selector_id}".*?</select>',
            template,
            flags=re.DOTALL,
        )
        assert match
        selector = match.group(0)
        for code, name in INDIC_UI_LANGUAGES.items():
            option = re.search(
                rf'<option value="{re.escape(code)}"([^>]*)>{re.escape(name)}</option>',
                selector,
            )
            assert option, f"{name} missing from {selector_id}"
            assert "disabled" not in option.group(1)


@pytest.mark.asyncio
async def test_specialized_stt_emits_only_confirmed_transcript_after_pause(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "indicconformer")
    monkeypatch.setenv(
        "STT_PROVIDER_MODEL", "ai4bharat/indic-conformer-600m-multilingual"
    )
    monkeypatch.setenv("STT_EMIT_PARTIALS", "false")
    worker = importlib.import_module("stt_providers.app.main")
    worker = importlib.reload(worker)
    transcripts = iter(("હેલો તમે કેમ છો", "પ્રોજેક્ટ પૂરો થઈ ગયો"))
    monkeypatch.setattr(
        worker,
        "_transcribe",
        lambda _audio, language: (next(transcripts), language),
    )

    voice = (1000).to_bytes(2, "little", signed=True) * 1600
    silence = b"\x00\x00" * 16000

    class FakeWebSocket:
        def __init__(self):
            self.messages = [
                {"bytes": voice},
                {"bytes": silence},
                {"bytes": voice},
                {"bytes": silence},
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

    assert len(websocket.sent) == 2
    assert websocket.sent[0]["text"] == "હેલો તમે કેમ છો"
    assert websocket.sent[1]["text"] == "હેલો તમે કેમ છો પ્રોજેક્ટ પૂરો થઈ ગયો"
    assert all(result["is_final"] is False for result in websocket.sent)
    assert all(result["metrics"]["force_emit"] is True for result in websocket.sent)


@pytest.mark.asyncio
async def test_specialized_stt_applies_live_session_emit_interval(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "parakeet")
    monkeypatch.setenv("STT_PROVIDER_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    monkeypatch.setenv("STT_MAX_UTTERANCE_SECONDS", "30")
    worker = importlib.import_module("stt_providers.app.main")
    worker = importlib.reload(worker)
    monkeypatch.setattr(
        worker,
        "_transcribe",
        lambda _audio, language: ("Bună, ce mai faci?", language),
    )
    continuous_voice = (1000).to_bytes(2, "little", signed=True) * 16000

    class FakeWebSocket:
        def __init__(self):
            self.messages = [
                {
                    "text": json.dumps(
                        {"type": "emit_interval_config", "emit_interval_seconds": 1}
                    )
                },
                {"bytes": continuous_voice},
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
        "type": "emit_interval_ack",
        "emit_interval_seconds": 1.0,
    }
    result = websocket.sent[1]
    assert result["text"] == "Bună, ce mai faci?"
    assert result["is_final"] is False
    assert result["metrics"]["finalization_reason"] == "emit_interval"


@pytest.mark.asyncio
async def test_specialized_stt_explicit_flush_commits_pending_audio(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "parakeet")
    monkeypatch.setenv("STT_PROVIDER_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    worker = importlib.import_module("stt_providers.app.main")
    worker = importlib.reload(worker)
    captured_audio = []

    def fake_transcribe(audio, language):
        captured_audio.append(audio)
        return "Would you like to join us?", language

    monkeypatch.setattr(worker, "_transcribe", fake_transcribe)
    voice = (1000).to_bytes(2, "little", signed=True) * 1600

    class FakeWebSocket:
        def __init__(self):
            self.messages = [
                {"bytes": voice},
                {"text": json.dumps({"type": "flush"})},
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

    assert captured_audio == [voice]
    assert websocket.sent[0]["text"] == "Would you like to join us?"
    assert websocket.sent[0]["is_final"] is False
    assert websocket.sent[0]["metrics"]["force_emit"] is True
    assert websocket.sent[0]["metrics"]["finalization_reason"] == "client_flush"


@pytest.mark.asyncio
async def test_specialized_stt_serializes_shared_model_inference(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "parakeet")
    monkeypatch.setenv("STT_PROVIDER_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    worker = importlib.import_module("stt_providers.app.main")
    worker = importlib.reload(worker)
    active_calls = 0
    max_active_calls = 0
    counter_lock = threading.Lock()

    def fake_transcribe(audio, language):
        nonlocal active_calls, max_active_calls
        with counter_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        time.sleep(0.05)
        with counter_lock:
            active_calls -= 1
        return "serialized", language

    monkeypatch.setattr(worker, "_transcribe", fake_transcribe)

    results = await asyncio.gather(
        *(
            asyncio.to_thread(worker._transcribe_serialized, b"audio", "en")
            for _ in range(4)
        )
    )

    assert results == [
        ("serialized", "en"),
        ("serialized", "en"),
        ("serialized", "en"),
        ("serialized", "en"),
    ]
    assert max_active_calls == 1


def test_parakeet_uses_non_lhotse_inference_dataloader(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "parakeet")
    monkeypatch.setenv("STT_PROVIDER_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    worker = importlib.import_module("stt_providers.app.main")
    worker = importlib.reload(worker)

    class FakeSamples:
        def astype(self, _dtype):
            return self

        def __truediv__(self, _value):
            return self

    numpy_stub = types.SimpleNamespace(
        frombuffer=lambda _audio, dtype: FakeSamples(),
        int16=object(),
        float32=object(),
    )
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    class FakeModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, **kwargs):
            self.calls.append((audio, kwargs))
            return ["clean transcript"]

    model = FakeModel()
    monkeypatch.setattr(worker, "_load_model", lambda: model)

    text, detected_language = worker._transcribe(b"\x00\x00" * 160, "en")

    assert text == "clean transcript"
    assert detected_language is None
    assert len(model.calls) == 1
    audio_paths, kwargs = model.calls[0]
    assert len(audio_paths) == 1
    assert kwargs == {"batch_size": 1, "use_lhotse": False, "verbose": False}


def test_specialized_stt_max_utterance_is_independent_from_emit_interval():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    for service_name in {
        "indicconformer-stt",
        "parakeet-stt",
        "sensevoice-stt",
    }:
        environment = compose["services"][service_name]["environment"]
        assert (
            "STT_MAX_UTTERANCE_SECONDS=${STT_MAX_UTTERANCE_SECONDS:-30.0}"
        ) in environment


def test_indicconformer_configures_cuda_warmup():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["indicconformer-stt"]["environment"]

    assert "STT_WARMUP_RUNS=${INDIC_STT_WARMUP_RUNS:-2}" in environment
    assert "STT_WARMUP_SECONDS=${INDIC_STT_WARMUP_SECONDS:-6.0}" in environment


def test_indicconformer_installs_gpu_onnxruntime():
    dockerfile = Path("stt_providers/Dockerfile").read_text(encoding="utf-8")

    assert "onnxruntime-gpu==1.23.2" in dockerfile
    assert "huggingface_hub onnxruntime ;;" not in dockerfile


def test_sensevoice_metadata_is_removed_from_transcript(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "sensevoice")
    monkeypatch.setenv("STT_PROVIDER_MODEL", "iic/SenseVoiceSmall")
    worker = importlib.import_module("stt_providers.app.main")
    worker = importlib.reload(worker)

    raw = "<|zh|><|HAPPY|><|Speech|><|withitn|>你好，今天怎么样？😊"
    assert worker._clean_sensevoice_text(raw) == "你好，今天怎么样？"
    assert worker._clean_sensevoice_text("今天我很开心。") == "今天我很开心。"


def test_speech_providers_start_in_standard_compose_stack():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    speech_providers = {
        "indicconformer-stt",
        "parakeet-stt",
        "sensevoice-stt",
        "indic-parler-tts",
    }

    for service_name in speech_providers:
        assert "profiles" not in services[service_name]

    depends_on = set(services["polytalk"]["depends_on"])
    assert depends_on.issuperset({"stt", "tts", "supertonic-tts"})


def test_gpu_compose_enables_every_model_service():
    compose = yaml.safe_load(Path("docker-compose.gpu.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    for service_name in {
        "stt",
        "indicconformer-stt",
        "parakeet-stt",
        "sensevoice-stt",
        "indic-parler-tts",
    }:
        assert services[service_name]["gpus"] == "all"

    parler = services["indic-parler-tts"]
    assert "cu121" in parler["build"]["args"]["INDIC_PARLER_TORCH_INDEX_URL"]
    assert "INDIC_PARLER_DEVICE=cuda" in parler["environment"]
