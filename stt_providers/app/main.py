"""Provider-neutral WebSocket worker for specialized STT models."""

import asyncio
import json
import logging
import math
import os
import re
import struct
import threading
import tempfile
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

LOGGER = logging.getLogger("gunicorn.error")
PROVIDER = os.environ["STT_PROVIDER"]
MODEL_NAME = os.environ["STT_PROVIDER_MODEL"]
DEVICE = os.environ.get("STT_DEVICE", "cuda")
SAMPLE_RATE = int(os.environ.get("STT_SAMPLE_RATE", "16000"))
CHUNK_SECONDS = float(os.environ.get("STT_STREAM_CHUNK_SECONDS", "3.0"))
WARMUP_RUNS = max(0, int(os.environ.get("STT_WARMUP_RUNS", "2")))
WARMUP_SECONDS = max(0.1, float(os.environ.get("STT_WARMUP_SECONDS", "6.0")))
SAMPLE_WIDTH_BYTES = 2
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH_BYTES
PAUSE_FLUSH_SECONDS = float(os.environ.get("STT_PAUSE_FLUSH_SECONDS", "1.0"))
SILENCE_RMS_THRESHOLD = float(os.environ.get("STT_SILENCE_RMS_THRESHOLD", "0.003"))
SPEECH_PAD_SECONDS = float(os.environ.get("STT_SPEECH_PAD_SECONDS", "0.2"))
MAX_UTTERANCE_SECONDS = float(os.environ.get("STT_MAX_UTTERANCE_SECONDS", "30.0"))
PRELOAD_MODEL = os.environ.get("STT_PRELOAD_MODEL", "true").lower() == "true"
MODEL: Any = None
MODEL_INFERENCE_LOCK = threading.Lock()

SENSEVOICE_CONTROL_TOKEN_RE = re.compile(r"<\|[^<>]*\|>")
SENSEVOICE_METADATA_EMOJI_RE = re.compile(r"[😊😔😡😰🤢😮😐🎼👏😀😭🤧]+")

LANGUAGE_ALIASES = {
    "zh": "zh",
    "yue": "yue",
    "ja": "ja",
    "ko": "ko",
    "as": "as",
    "bn": "bn",
    "brx": "brx",
    "doi": "doi",
    "gu": "gu",
    "hi": "hi",
    "kn": "kn",
    "kok": "kok",
    "ks": "ks",
    "mai": "mai",
    "ml": "ml",
    "mni": "mni",
    "mr": "mr",
    "ne": "ne",
    "or": "or",
    "pa": "pa",
    "sa": "sa",
    "sat": "sat",
    "sd": "sd",
    "ta": "ta",
    "te": "te",
    "ur": "ur",
}


def _validate_indicconformer_cuda(model: Any) -> None:
    """Fail startup when CUDA was requested but ONNX silently fell back to CPU."""
    if not DEVICE.startswith("cuda"):
        return

    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(
            "IndicConformer requires ONNX Runtime CUDAExecutionProvider when "
            f"STT_DEVICE={DEVICE}; available providers: {available_providers}"
        )

    sessions = {
        name: session
        for name, session in getattr(model, "models", {}).items()
        if hasattr(session, "get_providers")
    }
    if not sessions:
        raise RuntimeError("IndicConformer exposed no ONNX sessions to validate")

    fallback_sessions = {
        name: session.get_providers()
        for name, session in sessions.items()
        if not session.get_providers()
        or session.get_providers()[0] != "CUDAExecutionProvider"
    }
    if fallback_sessions:
        raise RuntimeError(
            "IndicConformer ONNX sessions did not select CUDAExecutionProvider: "
            f"{fallback_sessions}"
        )

    LOGGER.info(
        "IndicConformer ONNX CUDA sessions validated: %s",
        ", ".join(sorted(sessions)),
    )


def _warmup_indicconformer(model: Any) -> None:
    """Initialize lazy CUDA kernels before the worker becomes healthy."""
    if not DEVICE.startswith("cuda") or WARMUP_RUNS == 0:
        return

    import torch

    sample_count = max(1, int(SAMPLE_RATE * WARMUP_SECONDS))
    audio = torch.zeros((1, sample_count), dtype=torch.float32, device=DEVICE)
    with torch.inference_mode():
        for run_number in range(1, WARMUP_RUNS + 1):
            started = time.perf_counter()
            model(audio, "hi", "ctc")
            torch.cuda.synchronize()
            LOGGER.info(
                "IndicConformer CUDA warm-up %d/%d completed in %.3f seconds",
                run_number,
                WARMUP_RUNS,
                time.perf_counter() - started,
            )


def _load_model() -> Any:
    global MODEL
    if MODEL is not None:
        return MODEL
    if PROVIDER == "indicconformer":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = model.to(DEVICE).eval()
        _validate_indicconformer_cuda(model)
        _warmup_indicconformer(model)
    elif PROVIDER == "parakeet":
        from nemo.collections.asr.models import ASRModel

        model = ASRModel.from_pretrained(model_name=MODEL_NAME)
        if DEVICE.startswith("cuda"):
            model = model.cuda()
        model.eval()
    elif PROVIDER == "sensevoice":
        from funasr import AutoModel

        model = AutoModel(model=MODEL_NAME, device=DEVICE, disable_update=True)
    else:
        raise ValueError(f"Unsupported STT provider: {PROVIDER}")
    MODEL = model
    return MODEL


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if PRELOAD_MODEL:
        await asyncio.to_thread(_load_model)
    yield


app = FastAPI(title=f"PolyTalk STT worker ({PROVIDER})", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": PROVIDER, "model": MODEL_NAME}


def _write_wav(audio: bytes, target: Path) -> None:
    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio)


def _audio_rms(audio: bytes) -> float:
    """Return normalized RMS for mono signed 16-bit PCM audio."""
    usable_bytes = len(audio) - (len(audio) % SAMPLE_WIDTH_BYTES)
    if usable_bytes <= 0:
        return 0.0
    sample_count = usable_bytes // SAMPLE_WIDTH_BYTES
    square_sum = sum(
        sample * sample for (sample,) in struct.iter_unpack("<h", audio[:usable_bytes])
    )
    return (square_sum / sample_count) ** 0.5 / 32768.0


def _extract_text(result: Any) -> tuple[str, Optional[str]]:
    if isinstance(result, str):
        return result.strip(), None
    if isinstance(result, dict):
        text = result.get("text") or result.get("pred_text") or ""
        return str(text).strip(), result.get("language")
    if hasattr(result, "text"):
        return str(result.text).strip(), getattr(result, "language", None)
    if isinstance(result, (list, tuple)) and result:
        return _extract_text(result[0])
    return str(result or "").strip(), None


def _clean_sensevoice_text(text: str) -> str:
    """Remove SenseVoice language, emotion, event, and ITN metadata."""
    cleaned = SENSEVOICE_CONTROL_TOKEN_RE.sub("", text)
    cleaned = SENSEVOICE_METADATA_EMOJI_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _transcribe(audio: bytes, language: Optional[str]) -> tuple[str, Optional[str]]:
    # Provider ML dependencies stay lazy so this shared module does not require
    # every provider stack merely to be imported. Python caches them after the
    # first transcription in the worker process.
    import numpy as np

    model = _load_model()
    language = (language or "").replace("-", "_").split("_")[0].lower() or None
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    if PROVIDER == "indicconformer":
        import torch

        tensor = torch.from_numpy(samples).unsqueeze(0).to(DEVICE)
        result = model(tensor, LANGUAGE_ALIASES.get(language, language), "ctc")
        return _extract_text(result)

    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        _write_wav(audio, Path(handle.name))
        if PROVIDER == "parakeet":
            result = model.transcribe(
                [handle.name],
                batch_size=1,
                use_lhotse=False,
                verbose=False,
            )
            return _extract_text(result)
        result = model.generate(
            input=handle.name,
            cache={},
            language=language or "auto",
            use_itn=True,
            batch_size_s=60,
        )
        text, detected_language = _extract_text(result)
        return _clean_sensevoice_text(text), detected_language


def _transcribe_serialized(
    audio: bytes, language: Optional[str]
) -> tuple[str, Optional[str]]:
    """Prevent concurrent mutation of a shared provider model instance."""
    with MODEL_INFERENCE_LOCK:
        return _transcribe(audio, language)


@app.websocket("/v1/stream/transcriptions")
async def stream_transcriptions(websocket: WebSocket) -> None:
    await websocket.accept()
    language: Optional[str] = None
    emit_policy = "live"
    input_segmentation = "transport_vad"
    audio = bytearray()
    next_emit_size = int(BYTES_PER_SECOND * CHUNK_SECONDS)
    last_text = ""
    confirmed_text = ""
    speech_started = False
    trailing_silence_bytes = 0
    pause_flush_bytes = int(BYTES_PER_SECOND * PAUSE_FLUSH_SECONDS)
    speech_pad_bytes = int(BYTES_PER_SECOND * SPEECH_PAD_SECONDS)
    max_utterance_bytes = int(BYTES_PER_SECOND * MAX_UTTERANCE_SECONDS)
    emit_interval_bytes = int(BYTES_PER_SECOND * CHUNK_SECONDS)
    emit_partials = os.environ.get("STT_EMIT_PARTIALS", "false").lower() == "true"

    def set_emit_interval(value: object) -> Optional[float]:
        nonlocal emit_interval_bytes
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(seconds):
            return None
        seconds = min(30.0, max(1.0, seconds))
        emit_interval_bytes = int(BYTES_PER_SECOND * seconds)
        return seconds

    async def emit(
        is_final: bool,
        utterance: Optional[bytes] = None,
        finalization_reason: Optional[str] = None,
        commit: bool = False,
    ) -> None:
        nonlocal confirmed_text, last_text
        audio_bytes = utterance if utterance is not None else bytes(audio)
        if not audio_bytes:
            return
        started = time.perf_counter()
        text, detected_language = await asyncio.to_thread(
            _transcribe_serialized, audio_bytes, language
        )
        inference_seconds = time.perf_counter() - started
        cumulative_text = " ".join(
            part for part in (confirmed_text, text) if part
        ).strip()
        if text and (cumulative_text != last_text or is_final or commit):
            if is_final or commit:
                confirmed_text = cumulative_text
            last_text = cumulative_text
            await websocket.send_json(
                {
                    "text": cumulative_text,
                    "language": detected_language or language,
                    "is_final": is_final,
                    "has_speech": True,
                    "metrics": {
                        "backend": PROVIDER,
                        "inference_seconds": inference_seconds,
                        "audio_duration_seconds": len(audio_bytes) / BYTES_PER_SECOND,
                        "force_emit": is_final or commit,
                        "finalization_reason": finalization_reason,
                    },
                }
            )

    def reset_utterance() -> None:
        nonlocal speech_started, trailing_silence_bytes, next_emit_size
        audio.clear()
        speech_started = False
        trailing_silence_bytes = 0
        next_emit_size = int(BYTES_PER_SECOND * CHUNK_SECONDS)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if control.get("language"):
                    language = str(control["language"])
                if control.get("emit_policy") in {"live", "pause"}:
                    emit_policy = control["emit_policy"]
                    await websocket.send_json(
                        {"type": "emit_policy_ack", "emit_policy": emit_policy}
                    )
                if "candidate_languages" in control:
                    await websocket.send_json(
                        {
                            "type": "candidate_languages_ack",
                            "candidate_languages": control["candidate_languages"],
                        }
                    )
                if control.get("input_segmentation") == "upstream_vad":
                    input_segmentation = "upstream_vad"
                    await websocket.send_json(
                        {
                            "type": "input_segmentation_ack",
                            "input_segmentation": input_segmentation,
                        }
                    )
                if "emit_interval_seconds" in control:
                    seconds = set_emit_interval(control["emit_interval_seconds"])
                    if seconds is not None:
                        await websocket.send_json(
                            {
                                "type": "emit_interval_ack",
                                "emit_interval_seconds": seconds,
                            }
                        )
                if control.get("type") == "flush":
                    if speech_started:
                        trim_bytes = max(0, trailing_silence_bytes - speech_pad_bytes)
                        utterance = (
                            bytes(audio[:-trim_bytes]) if trim_bytes else bytes(audio)
                        )
                        await emit(
                            is_final=False,
                            utterance=utterance,
                            finalization_reason=(
                                control.get("reason")
                                if control.get("reason")
                                in {"client_flush", "vad_pause"}
                                else "client_flush"
                            ),
                            commit=True,
                        )
                        reset_utterance()
                    continue
                if control.get("type") == "end":
                    if speech_started:
                        await emit(
                            is_final=True,
                            finalization_reason="end",
                            commit=True,
                        )
                    break
                continue

            chunk = message.get("bytes")
            if not chunk:
                continue

            has_voice = (
                input_segmentation == "upstream_vad"
                or _audio_rms(chunk) >= SILENCE_RMS_THRESHOLD
            )
            if not speech_started and not has_voice:
                continue

            if has_voice:
                speech_started = True
                trailing_silence_bytes = 0
            else:
                trailing_silence_bytes += len(chunk)

            audio.extend(chunk)

            should_finalize = (
                speech_started
                and pause_flush_bytes > 0
                and trailing_silence_bytes >= pause_flush_bytes
            )
            reached_max_utterance = (
                speech_started
                and max_utterance_bytes > 0
                and len(audio) >= max_utterance_bytes
            )
            reached_emit_interval = (
                speech_started
                and emit_policy == "live"
                and emit_interval_bytes > 0
                and len(audio) >= emit_interval_bytes
            )
            if should_finalize or reached_max_utterance or reached_emit_interval:
                trim_bytes = max(0, trailing_silence_bytes - speech_pad_bytes)
                utterance = bytes(audio[:-trim_bytes]) if trim_bytes else bytes(audio)
                if should_finalize:
                    reason = "pause"
                elif reached_max_utterance:
                    reason = "max_duration"
                else:
                    reason = "emit_interval"
                await emit(
                    is_final=False,
                    utterance=utterance,
                    finalization_reason=reason,
                    commit=True,
                )
                reset_utterance()
            elif (
                emit_partials and emit_policy == "live" and len(audio) >= next_emit_size
            ):
                await emit(is_final=False)
                next_emit_size = len(audio) + int(BYTES_PER_SECOND * CHUNK_SECONDS)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json(
                {"error": str(exc), "provider": PROVIDER, "language": language}
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
