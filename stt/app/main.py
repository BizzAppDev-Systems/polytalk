# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import io
import json
import logging
import math
import os
import sys
import time
import unicodedata
import wave
from array import array
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel


def _resolve_log_level(default: int = logging.INFO) -> int:
    """Resolve LOG_LEVEL from the environment, falling back to default."""
    level_name = os.environ.get("LOG_LEVEL", "").strip().upper()
    if not level_name:
        return default

    level = logging.getLevelName(level_name)
    if isinstance(level, int):
        return level

    return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    if PRELOAD_MODEL:
        await asyncio.to_thread(_get_model)
    yield


app = FastAPI(title="STT Service (faster-whisper)", lifespan=lifespan)
LOG_LEVEL = _resolve_log_level()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("polytalk_stt")
logger.setLevel(LOG_LEVEL)

STORAGE_DIR = os.environ.get("STORAGE_DIR", "/tmp/stt")

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))

SAMPLE_RATE = int(os.environ.get("STT_SAMPLE_RATE", "16000"))
CHANNELS = int(os.environ.get("STT_CHANNELS", "1"))
SAMPLE_WIDTH_BYTES = int(os.environ.get("STT_SAMPLE_WIDTH_BYTES", "2"))
STREAM_CHUNK_SECONDS = float(os.environ.get("STT_STREAM_CHUNK_SECONDS", "1.2"))
CHUNK_OVERLAP_SECONDS = float(os.environ.get("STT_CHUNK_OVERLAP_SECONDS", "0.25"))
TRANSCRIBE_WORKERS = int(os.environ.get("STT_TRANSCRIBE_WORKERS", "2"))
TRANSCRIBE_QUEUE_SIZE = int(os.environ.get("STT_TRANSCRIBE_QUEUE_SIZE", "8"))
EMIT_MIN_CHARS = int(os.environ.get("STT_EMIT_MIN_CHARS", "40"))
EMIT_INTERVAL_SECONDS = float(os.environ.get("STT_EMIT_INTERVAL_SECONDS", "2.0"))
PAUSE_FLUSH_SECONDS = float(os.environ.get("STT_PAUSE_FLUSH_SECONDS", "1.0"))
IDLE_FLUSH_SECONDS = float(os.environ.get("STT_IDLE_FLUSH_SECONDS", "3.0"))
LEADING_SILENCE_PREROLL_SECONDS = float(
    os.environ.get("STT_LEADING_SILENCE_PREROLL_SECONDS", "0.2")
)
PRELOAD_MODEL = os.environ.get("STT_PRELOAD_MODEL", "true").lower() == "true"
SILENCE_RMS_THRESHOLD = float(os.environ.get("STT_SILENCE_RMS_THRESHOLD", "0.003"))
NO_SPEECH_PROB_THRESHOLD = float(os.environ.get("STT_NO_SPEECH_PROB_THRESHOLD", "0.50"))
LOG_PROB_THRESHOLD = float(os.environ.get("STT_LOG_PROB_THRESHOLD", "-1.0"))
VAD_FILTER = os.environ.get("STT_VAD_FILTER", "true").lower() == "true"
VAD_MIN_SILENCE_MS = int(os.environ.get("STT_VAD_MIN_SILENCE_MS", "500"))
VAD_SPEECH_PAD_MS = int(os.environ.get("STT_VAD_SPEECH_PAD_MS", "200"))
WORD_TIMESTAMPS = os.environ.get("STT_WORD_TIMESTAMPS", "true").lower() == "true"
CONDITION_ON_PREVIOUS_TEXT = (
    os.environ.get("STT_CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"
)
TEMPERATURE = float(os.environ.get("STT_TEMPERATURE", "0.0"))
INITIAL_PROMPT = os.environ.get("STT_INITIAL_PROMPT") or None
CANDIDATE_LANGUAGE_CONFIDENCE = float(
    os.environ.get("STT_CANDIDATE_LANGUAGE_CONFIDENCE", "0.70")
)
MAX_CANDIDATE_DECODES = max(1, int(os.environ.get("STT_MAX_CANDIDATE_DECODES", "3")))
CANDIDATE_SCRIPT_SCORE_WEIGHT = float(
    os.environ.get("STT_CANDIDATE_SCRIPT_SCORE_WEIGHT", "1.50")
)
MAX_STREAM_BYTES = MAX_UPLOAD_MB * 1024 * 1024
WORD_OVERLAP_STRIP_CHARS = " \t\r\n.,!?;:…\"'()[]{}"
MAX_ADJACENT_PHRASE_REPEATS = 2
# Keep this floor above natural short repeats so env values like 0 or 1 cannot
# make cross-delta trimming delete ordinary repeated words.
MAX_CROSS_DELTA_WORD_REPEATS = max(
    3, int(os.environ.get("STT_MAX_CROSS_DELTA_WORD_REPEATS", "6"))
)

WHISPER_MODEL_NAME = os.environ.get("STT_MODEL") or os.environ.get(
    "WHISPER_MODEL", "small"
)
DEVICE = os.environ.get("STT_DEVICE") or os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE") or os.environ.get(
    "COMPUTE_TYPE", "int8"
)
MODEL_WORKERS = int(os.environ.get("STT_MODEL_WORKERS", str(TRANSCRIBE_WORKERS)))
MAX_PENDING_RESULTS = max(TRANSCRIBE_QUEUE_SIZE * max(1, TRANSCRIBE_WORKERS) * 4, 32)

_model: Optional[WhisperModel] = None


@dataclass
class TranscribeJob:
    """A PCM audio window queued for transcription."""

    sequence: int
    audio_bytes: bytes
    language: Optional[str]
    candidate_languages: tuple[str, ...]
    task: str
    queued_at: float
    queue_depth_at_enqueue: int
    backpressure_seconds: float = 0.0
    force_emit: bool = False
    upstream_vad: bool = False
    finalization_reason: Optional[str] = None


@dataclass
class TranscribeResult:
    """Ordered transcription result from a worker."""

    sequence: int
    transcript: str = ""
    has_speech: bool = False
    detected_language: Optional[str] = None
    skipped_silence: bool = False
    error: Optional[str] = None
    audio_duration_seconds: float = 0.0
    audio_rms: float = 0.0
    queued_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    queue_depth_at_enqueue: int = 0
    backpressure_seconds: float = 0.0
    force_emit: bool = False
    finalization_reason: Optional[str] = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            num_workers=MODEL_WORKERS,
        )
    return _model


def _normalize_language_code(language: Optional[str]) -> Optional[str]:
    """Normalize a UI/provider language code to a base STT language code."""
    if not language:
        return None
    return str(language).replace("-", "_").split("_")[0].lower() or None


def _normalize_candidate_languages(value: object) -> tuple[str, ...]:
    """Normalize candidate language control-message values."""
    if not isinstance(value, list):
        return ()

    normalized: list[str] = []
    for item in value:
        code = _normalize_language_code(item if isinstance(item, str) else None)
        if code and code not in normalized:
            normalized.append(code)
    return tuple(normalized)


LANGUAGE_SCRIPT_NAMES = {
    "ar": ("ARABIC",),
    "as": ("BENGALI",),
    "bn": ("BENGALI",),
    "brx": ("DEVANAGARI",),
    "bg": ("CYRILLIC",),
    "el": ("GREEK",),
    "gu": ("GUJARATI",),
    "hi": ("DEVANAGARI",),
    "ja": ("HIRAGANA", "KATAKANA", "CJK"),
    "kn": ("KANNADA",),
    "ko": ("HANGUL",),
    "ml": ("MALAYALAM",),
    "mni": ("MEETEI MAYEK", "BENGALI"),
    "mr": ("DEVANAGARI",),
    "or": ("ORIYA",),
    "pa": ("GURMUKHI",),
    "ru": ("CYRILLIC",),
    "ta": ("TAMIL",),
    "te": ("TELUGU",),
    "uk": ("CYRILLIC",),
    "ur": ("ARABIC",),
    "zh": ("CJK", "BOPOMOFO"),
}
LATIN_SCRIPT_LANGUAGES = {
    "cs",
    "da",
    "de",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
    "tr",
    "vi",
}


def _language_script_match_ratio(text: str, language: str) -> Optional[float]:
    """Return the share of letters matching a language's expected script."""
    script_names = LANGUAGE_SCRIPT_NAMES.get(language)
    if script_names is None and language in LATIN_SCRIPT_LANGUAGES:
        script_names = ("LATIN",)
    if script_names is None:
        return None

    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    if not letters:
        return None
    matches = sum(
        any(script_name in unicodedata.name(char, "") for script_name in script_names)
        for char in letters
    )
    return matches / len(letters)


def _collect_transcription_segments(segments) -> tuple[str, bool, float]:
    """Collect accepted Whisper segments and their text-weighted log probability."""
    text_parts: list[str] = []
    weighted_logprob = 0.0
    total_weight = 0
    for segment in segments:
        no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
        avg_logprob = getattr(segment, "avg_logprob", 0.0)
        if (
            no_speech_prob >= NO_SPEECH_PROB_THRESHOLD
            or avg_logprob <= LOG_PROB_THRESHOLD
        ):
            continue
        text = segment.text
        if text.strip():
            text_parts.append(text)
            weight = max(1, len(text.strip()))
            weighted_logprob += avg_logprob * weight
            total_weight += weight
    transcript = "".join(text_parts)
    mean_logprob = weighted_logprob / total_weight if total_weight else float("-inf")
    return transcript, bool(text_parts), mean_logprob


def _calculate_rms(audio_bytes: bytes) -> float:
    """Return normalized RMS for signed 16-bit PCM audio."""
    if len(audio_bytes) < SAMPLE_WIDTH_BYTES:
        return 0.0

    sample_bytes = audio_bytes[: len(audio_bytes) - (len(audio_bytes) % 2)]
    samples = array("h")
    samples.frombytes(sample_bytes)
    if sys.byteorder == "big":
        samples.byteswap()

    if not samples:
        return 0.0

    square_sum = sum(sample * sample for sample in samples)
    return math.sqrt(square_sum / len(samples)) / 32768.0


def _transcribe_audio(
    model: WhisperModel,
    wav_buffer: io.BytesIO,
    language: Optional[str],
    task: str,
    candidate_languages: tuple[str, ...] = (),
):
    """Run blocking Whisper transcription and collect accepted segments.

    When no explicit language is supplied, conversation mode can provide a
    bounded candidate set. Confident in-set detections use the fast path;
    ambiguous or out-of-set detections decode both selected languages and
    choose using transcription likelihood and script consistency.
    """
    vad_parameters = None
    if VAD_FILTER:
        vad_parameters = {
            "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
        }

    def run_transcription(language_hint):
        return model.transcribe(
            wav_buffer,
            language=language_hint,
            task=task,
            vad_filter=VAD_FILTER,
            vad_parameters=vad_parameters,
            word_timestamps=WORD_TIMESTAMPS,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            temperature=TEMPERATURE,
            no_speech_threshold=NO_SPEECH_PROB_THRESHOLD,
            log_prob_threshold=LOG_PROB_THRESHOLD,
            initial_prompt=INITIAL_PROMPT,
        )

    normalized_candidates = tuple(
        dict.fromkeys(
            normalized
            for candidate in candidate_languages
            if (normalized := _normalize_language_code(candidate))
        )
    )
    if not language and normalized_candidates:
        detection_segments, detection_info = run_transcription(None)
        probabilities = {
            _normalize_language_code(code): float(probability)
            for code, probability in (
                getattr(detection_info, "all_language_probs", None) or []
            )
            if _normalize_language_code(code)
        }
        selected_language = max(
            normalized_candidates,
            key=lambda candidate: probabilities.get(candidate, float("-inf")),
        )
        detected_language = _normalize_language_code(
            getattr(detection_info, "language", None)
        )
        detected_confidence = probabilities.get(
            detected_language,
            float(getattr(detection_info, "language_probability", 0.0) or 0.0),
        )
        if (
            detected_language in normalized_candidates
            and detected_confidence >= CANDIDATE_LANGUAGE_CONFIDENCE
        ):
            transcript, has_speech, _ = _collect_transcription_segments(
                detection_segments
            )
            logger.info(
                "Conversation language fast-path selected %s (confidence=%.3f)",
                detected_language,
                detected_confidence,
            )
            return transcript, has_speech, detected_language

        best_result = ("", False, selected_language)
        best_score = float("-inf")
        candidate_scores: dict[str, float] = {}
        if len(normalized_candidates) <= MAX_CANDIDATE_DECODES:
            decode_candidates = normalized_candidates
        else:
            # Preserve the caller's selected-language order. Auto-detection is
            # intentionally not used to discard candidates because it is least
            # reliable on the ambiguous windows handled by this branch.
            decode_candidates = normalized_candidates[:MAX_CANDIDATE_DECODES]
        for candidate in decode_candidates:
            wav_buffer.seek(0)
            candidate_segments, _candidate_info = run_transcription(candidate)
            transcript, has_speech, mean_logprob = _collect_transcription_segments(
                candidate_segments
            )
            score = mean_logprob
            script_ratio = _language_script_match_ratio(transcript, candidate)
            if script_ratio is not None:
                score += (script_ratio - 0.5) * CANDIDATE_SCRIPT_SCORE_WEIGHT
            candidate_scores[candidate] = score
            if has_speech and score > best_score:
                best_score = score
                best_result = (transcript, True, candidate)

        logger.info(
            "Conversation language candidate decode selected %s "
            "(detected=%s, confidence=%.3f, candidates=%s, scores=%s)",
            best_result[2],
            detected_language,
            detected_confidence,
            decode_candidates,
            {key: round(value, 3) for key, value in candidate_scores.items()},
        )
        return best_result
    else:
        segments, info = run_transcription(language)

    transcript, has_speech, _ = _collect_transcription_segments(segments)
    return transcript, has_speech, getattr(info, "language", None)


def _trim_pause_flush_audio(
    audio_bytes: bytes, trailing_silence_bytes: int, bytes_per_second: int
) -> bytes:
    """Trim most trailing silence before transcribing a pause-flushed window."""
    if trailing_silence_bytes <= 0 or bytes_per_second <= 0:
        return audio_bytes

    speech_pad_bytes = int((VAD_SPEECH_PAD_MS / 1000.0) * bytes_per_second)
    speech_pad_bytes -= speech_pad_bytes % SAMPLE_WIDTH_BYTES
    trim_bytes = max(0, trailing_silence_bytes - speech_pad_bytes)
    trim_bytes -= trim_bytes % SAMPLE_WIDTH_BYTES

    if trim_bytes <= 0 or trim_bytes >= len(audio_bytes):
        return audio_bytes
    return audio_bytes[:-trim_bytes]


def _append_bounded_audio_preroll(
    chunks: list[bytes], audio_data: bytes, max_bytes: int, current_total_bytes: int
) -> int:
    """Keep only the most recent audio bytes for pre-speech context."""
    if max_bytes <= 0:
        chunks.clear()
        return 0

    chunks.append(audio_data)
    current_total_bytes += len(audio_data)
    while chunks and current_total_bytes > max_bytes:
        overflow_bytes = current_total_bytes - max_bytes
        first_chunk = chunks[0]
        if len(first_chunk) <= overflow_bytes:
            current_total_bytes -= len(first_chunk)
            chunks.pop(0)
            continue

        chunks[0] = first_chunk[overflow_bytes:]
        current_total_bytes -= overflow_bytes

    return current_total_bytes


def _build_wav_buffer(audio_bytes: bytes) -> io.BytesIO:
    """Wrap raw PCM audio bytes in an in-memory WAV container."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_bytes)
    wav_buffer.seek(0)
    return wav_buffer


def _process_transcribe_job(
    model: WhisperModel, job: TranscribeJob
) -> TranscribeResult:
    """Process one queued transcription job."""
    started_at = time.time()
    audio_duration_seconds = (
        len(job.audio_bytes) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH_BYTES)
        if SAMPLE_RATE and CHANNELS and SAMPLE_WIDTH_BYTES
        else 0.0
    )
    audio_rms = _calculate_rms(job.audio_bytes)
    if not job.upstream_vad and audio_rms < SILENCE_RMS_THRESHOLD:
        completed_at = time.time()
        return TranscribeResult(
            sequence=job.sequence,
            skipped_silence=True,
            audio_duration_seconds=audio_duration_seconds,
            audio_rms=audio_rms,
            queued_at=job.queued_at,
            started_at=started_at,
            completed_at=completed_at,
            queue_depth_at_enqueue=job.queue_depth_at_enqueue,
            backpressure_seconds=job.backpressure_seconds,
            force_emit=job.force_emit,
            finalization_reason=job.finalization_reason,
        )

    transcript, has_speech, detected_language = _transcribe_audio(
        model,
        _build_wav_buffer(job.audio_bytes),
        job.language,
        job.task,
        job.candidate_languages,
    )
    completed_at = time.time()
    return TranscribeResult(
        sequence=job.sequence,
        transcript=transcript,
        has_speech=has_speech,
        detected_language=detected_language,
        audio_duration_seconds=audio_duration_seconds,
        audio_rms=audio_rms,
        queued_at=job.queued_at,
        started_at=started_at,
        completed_at=completed_at,
        queue_depth_at_enqueue=job.queue_depth_at_enqueue,
        backpressure_seconds=job.backpressure_seconds,
        force_emit=job.force_emit,
        finalization_reason=job.finalization_reason,
    )


def _result_metrics(
    result: TranscribeResult, emitted_at: Optional[float] = None
) -> dict:
    """Build a compact metrics payload for logs and WebSocket responses."""
    emitted_at = emitted_at or time.time()
    inference_seconds = max(0.0, result.completed_at - result.started_at)
    queue_wait_seconds = max(0.0, result.started_at - result.queued_at)
    total_job_seconds = max(0.0, result.completed_at - result.queued_at)
    emit_delay_seconds = max(0.0, emitted_at - result.completed_at)
    return {
        "sequence": result.sequence,
        "audio_duration_seconds": round(result.audio_duration_seconds, 3),
        "audio_rms": round(result.audio_rms, 5),
        "queue_wait_seconds": round(queue_wait_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "total_job_seconds": round(total_job_seconds, 3),
        "emit_delay_seconds": round(emit_delay_seconds, 3),
        "queue_depth_at_enqueue": result.queue_depth_at_enqueue,
        "backpressure_seconds": round(result.backpressure_seconds, 3),
        "skipped_silence": result.skipped_silence,
        "has_speech": result.has_speech,
        "force_emit": result.force_emit,
        "finalization_reason": result.finalization_reason,
    }


def _should_drop_no_speech_new_text(result: TranscribeResult, new_text: str) -> bool:
    """Return whether new transcript text should be suppressed as no-speech."""
    return bool(new_text.strip()) and not result.has_speech


def _normalize_overlap_word(word: str) -> str:
    """Normalize one word for overlap comparison only."""
    return word.strip(WORD_OVERLAP_STRIP_CHARS).casefold()


def _overlap_words_match(left: list[str], right: list[str]) -> bool:
    """Return whether two word windows match after punctuation cleanup."""
    if len(left) != len(right):
        return False
    return all(
        _normalize_overlap_word(left_word) == _normalize_overlap_word(right_word)
        for left_word, right_word in zip(left, right)
    )


def _get_new_transcript_text(existing_text: str, update_text: str) -> str:
    """Return only new words from an overlapped chunk transcript."""
    existing = " ".join(existing_text.strip().split())
    update = " ".join(update_text.strip().split())

    if not update:
        return ""
    if not existing:
        return update
    if existing.endswith(update):
        return ""
    if update.startswith(existing):
        return update[len(existing) :].strip()

    existing_words = existing.split()
    update_words = update.split()
    max_overlap = min(len(existing_words), len(update_words), 16)

    for overlap_size in range(max_overlap, 0, -1):
        if _overlap_words_match(
            existing_words[-overlap_size:],
            update_words[:overlap_size],
        ):
            return " ".join(update_words[overlap_size:]).strip()

    return update


def _collapse_repeated_word_sequences(text: str) -> str:
    """Collapse pathological adjacent repeated word sequences in ASR deltas."""
    words = text.split()
    if len(words) < 4:
        return text

    max_ngram = min(8, len(words) // (MAX_ADJACENT_PHRASE_REPEATS + 1))
    ngram_size = max_ngram
    while ngram_size >= 1:
        collapsed: list[str] = []
        index = 0
        changed = False

        while index < len(words):
            window = words[index : index + ngram_size]
            if len(window) < ngram_size:
                collapsed.extend(words[index:])
                break

            repeats = 1
            next_index = index + ngram_size
            while next_index + ngram_size <= len(words) and _overlap_words_match(
                window,
                words[next_index : next_index + ngram_size],
            ):
                repeats += 1
                next_index += ngram_size

            kept_repeats = min(repeats, MAX_ADJACENT_PHRASE_REPEATS)
            for _ in range(kept_repeats):
                collapsed.extend(window)

            if repeats > MAX_ADJACENT_PHRASE_REPEATS:
                changed = True
            index = next_index

        if changed:
            words = collapsed

        ngram_size -= 1

    return " ".join(words)


def _trim_repeated_leading_words(existing_text: str, new_text: str) -> str:
    """Limit long same-word loops that continue across transcript deltas."""
    existing_words = existing_text.split()
    new_words = new_text.split()
    if not existing_words or not new_words:
        return new_text

    repeated_word = _normalize_overlap_word(existing_words[-1])
    if not repeated_word:
        return new_text

    existing_repeats = 0
    for word in reversed(existing_words):
        if _normalize_overlap_word(word) != repeated_word:
            break
        existing_repeats += 1

    if existing_repeats < MAX_CROSS_DELTA_WORD_REPEATS:
        return new_text

    first_content_index = 0
    while first_content_index < len(new_words) and not _normalize_overlap_word(
        new_words[first_content_index]
    ):
        first_content_index += 1

    if first_content_index == len(new_words):
        return ""

    if _normalize_overlap_word(new_words[first_content_index]) != repeated_word:
        if first_content_index > 0:
            return " ".join(new_words[first_content_index:])
        return new_text

    first_non_repeat_index = first_content_index
    while (
        first_non_repeat_index < len(new_words)
        and _normalize_overlap_word(new_words[first_non_repeat_index]) == repeated_word
    ):
        first_non_repeat_index += 1

    return " ".join(new_words[first_non_repeat_index:])


async def _safe_close_websocket(websocket: WebSocket) -> None:
    """Close a WebSocket without raising if the peer already closed it."""
    try:
        await websocket.close()
    except RuntimeError as e:
        if "websocket.close" not in str(e):
            raise
    except Exception:
        pass


@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}


@app.websocket("/v1/stream/transcriptions")
async def stream_transcription(
    websocket: WebSocket,
    language: Optional[str] = None,
    task: str = "transcribe",
):
    await websocket.accept()

    if task not in {"transcribe", "translate"}:
        await websocket.send_json({"error": "task must be transcribe or translate"})
        await _safe_close_websocket(websocket)
        return

    model = _get_model()
    job_queue: asyncio.Queue[TranscribeJob | None] = asyncio.Queue(
        maxsize=TRANSCRIBE_QUEUE_SIZE
    )
    result_queue: asyncio.Queue[TranscribeResult] = asyncio.Queue()
    stop_event = asyncio.Event()

    current_language = language
    current_candidate_languages: tuple[str, ...] = ()
    current_task = task
    current_emit_policy = "live"
    # Legacy clients remain on transport/RMS gating. WebSocket ordering ensures
    # an upstream_vad control is applied before any later binary audio message.
    current_input_segmentation = "transport_vad"
    current_emit_interval_seconds = EMIT_INTERVAL_SECONDS
    audio_chunks = []
    total_size = 0
    next_sequence = 0
    bytes_per_second = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH_BYTES
    min_chunk_bytes = int(STREAM_CHUNK_SECONDS * bytes_per_second)
    overlap_bytes = int(CHUNK_OVERLAP_SECONDS * bytes_per_second)
    overlap_bytes -= overlap_bytes % SAMPLE_WIDTH_BYTES

    async def receive_audio() -> None:
        """Receive WebSocket audio and enqueue PCM windows."""
        nonlocal \
            current_language, \
            current_candidate_languages, \
            current_task, \
            current_emit_policy, \
            current_input_segmentation, \
            current_emit_interval_seconds, \
            min_chunk_bytes, \
            total_size, \
            next_sequence
        current_window_has_voice = False
        trailing_silence_bytes = 0
        leading_silence_preroll_chunks: list[bytes] = []
        leading_silence_preroll_total_bytes = 0
        leading_silence_preroll_bytes = int(
            LEADING_SILENCE_PREROLL_SECONDS * bytes_per_second
        )
        leading_silence_preroll_bytes -= (
            leading_silence_preroll_bytes % SAMPLE_WIDTH_BYTES
        )
        pause_flush_bytes = int(PAUSE_FLUSH_SECONDS * bytes_per_second)
        pause_flush_bytes -= pause_flush_bytes % SAMPLE_WIDTH_BYTES

        async def enqueue_audio_window(
            audio_bytes: bytes,
            force_emit: bool = False,
            finalization_reason: Optional[str] = None,
        ) -> bool:
            nonlocal current_window_has_voice, next_sequence, trailing_silence_bytes
            enqueue_started_at = time.time()
            queue_depth_before = job_queue.qsize()
            while job_queue.full() and not stop_event.is_set():
                logger.warning(
                    "[STT_BACKPRESSURE] queue full before enqueue "
                    f"seq={next_sequence} depth={queue_depth_before} "
                    f"max={TRANSCRIBE_QUEUE_SIZE}"
                )
                await asyncio.sleep(0.01)
                queue_depth_before = job_queue.qsize()
            if stop_event.is_set():
                return False
            queued_at = time.time()
            job_audio_bytes = (
                _trim_pause_flush_audio(
                    audio_bytes, trailing_silence_bytes, bytes_per_second
                )
                if force_emit
                else audio_bytes
            )
            job = TranscribeJob(
                sequence=next_sequence,
                audio_bytes=job_audio_bytes,
                language=current_language,
                candidate_languages=current_candidate_languages,
                task=current_task,
                queued_at=queued_at,
                queue_depth_at_enqueue=queue_depth_before,
                backpressure_seconds=queued_at - enqueue_started_at,
                force_emit=force_emit,
                upstream_vad=current_input_segmentation == "upstream_vad",
                finalization_reason=finalization_reason,
            )
            if stop_event.is_set():
                return False
            job_queue.put_nowait(job)
            logger.debug(
                "[STT_QUEUE] enqueued "
                f"seq={next_sequence} depth_before={queue_depth_before} "
                f"depth_after={job_queue.qsize()} "
                f"backpressure={job.backpressure_seconds:.3f}s "
                f"audio_seconds={len(job_audio_bytes) / bytes_per_second:.2f} "
                f"input_audio_seconds={len(audio_bytes) / bytes_per_second:.2f} "
                f"force_emit={force_emit}"
            )
            next_sequence += 1

            if force_emit:
                audio_chunks.clear()
                current_window_has_voice = False
                trailing_silence_bytes = 0
            elif overlap_bytes > 0 and len(audio_bytes) > overlap_bytes:
                overlap_audio = audio_bytes[-overlap_bytes:]
                audio_chunks[:] = [overlap_audio]
                current_window_has_voice = (
                    current_input_segmentation == "upstream_vad"
                    or _calculate_rms(overlap_audio) >= SILENCE_RMS_THRESHOLD
                )
                trailing_silence_bytes = (
                    0 if current_window_has_voice else len(overlap_audio)
                )
            else:
                audio_chunks.clear()
                current_window_has_voice = False
                trailing_silence_bytes = 0
            return True

        try:
            while not stop_event.is_set():
                try:
                    if (
                        current_emit_policy == "pause"
                        and current_window_has_voice
                        and current_input_segmentation != "upstream_vad"
                        and IDLE_FLUSH_SECONDS > 0
                    ):
                        data = await asyncio.wait_for(
                            websocket.receive(), timeout=IDLE_FLUSH_SECONDS
                        )
                    else:
                        data = await websocket.receive()
                except asyncio.TimeoutError:
                    audio_bytes = b"".join(audio_chunks)
                    if audio_bytes and not await enqueue_audio_window(
                        audio_bytes,
                        force_emit=True,
                        finalization_reason="idle_pause",
                    ):
                        break
                    continue

                if data.get("type") == "websocket.disconnect":
                    stop_event.set()
                    break

                if "text" in data:
                    try:
                        message = json.loads(data["text"])
                    except json.JSONDecodeError:
                        continue

                    if message.get("type") == "end":
                        audio_bytes = b"".join(audio_chunks)
                        if audio_bytes:
                            await enqueue_audio_window(
                                audio_bytes,
                                force_emit=True,
                                finalization_reason="end",
                            )
                        stop_event.set()
                        break
                    if message.get("type") == "flush":
                        audio_bytes = b"".join(audio_chunks)
                        reason = message.get("reason")
                        if reason not in {"client_flush", "vad_pause"}:
                            reason = "client_flush"
                        if audio_bytes:
                            await enqueue_audio_window(
                                audio_bytes,
                                force_emit=True,
                                finalization_reason=reason,
                            )
                        continue
                    if message.get("language"):
                        current_language = message["language"]
                    if message.get("task") in {"transcribe", "translate"}:
                        current_task = message["task"]
                    if "candidate_languages" in message:
                        current_candidate_languages = _normalize_candidate_languages(
                            message.get("candidate_languages")
                        )
                        await websocket.send_json(
                            {
                                "type": "candidate_languages_ack",
                                "candidate_languages": list(
                                    current_candidate_languages
                                ),
                            }
                        )
                    if message.get("emit_policy") in {"live", "pause"}:
                        current_emit_policy = message["emit_policy"]
                        await websocket.send_json(
                            {
                                "type": "emit_policy_ack",
                                "emit_policy": current_emit_policy,
                            }
                        )
                    if message.get("input_segmentation") == "upstream_vad":
                        current_input_segmentation = "upstream_vad"
                        await websocket.send_json(
                            {
                                "type": "input_segmentation_ack",
                                "input_segmentation": current_input_segmentation,
                            }
                        )
                    if "emit_interval_seconds" in message:
                        try:
                            seconds = float(message["emit_interval_seconds"])
                        except (TypeError, ValueError):
                            seconds = None
                        if seconds is not None and math.isfinite(seconds):
                            current_emit_interval_seconds = min(30.0, max(1.0, seconds))
                            min_chunk_bytes = int(
                                current_emit_interval_seconds * bytes_per_second
                            )
                            min_chunk_bytes -= min_chunk_bytes % SAMPLE_WIDTH_BYTES
                            await websocket.send_json(
                                {
                                    "type": "emit_interval_ack",
                                    "emit_interval_seconds": current_emit_interval_seconds,
                                }
                            )
                    continue

                if "bytes" not in data:
                    continue

                audio_data = data["bytes"]
                if len(audio_data) == 0:
                    continue

                if total_size + len(audio_data) > MAX_STREAM_BYTES:
                    await result_queue.put(
                        TranscribeResult(
                            sequence=next_sequence,
                            error=f"Audio too large (max {MAX_STREAM_BYTES} bytes)",
                        )
                    )
                    stop_event.set()
                    break

                # This is a total received-byte guard, not the current
                # audio_chunks buffer size. Startup silence used later as
                # preroll is already counted here when received.
                total_size += len(audio_data)

                audio_rms = _calculate_rms(audio_data)
                has_voice = (
                    current_input_segmentation == "upstream_vad"
                    or audio_rms >= SILENCE_RMS_THRESHOLD
                )

                if not current_window_has_voice and not has_voice:
                    leading_silence_preroll_total_bytes = _append_bounded_audio_preroll(
                        leading_silence_preroll_chunks,
                        audio_data,
                        leading_silence_preroll_bytes,
                        leading_silence_preroll_total_bytes,
                    )
                    # The active speech window stays empty during startup silence;
                    # recent silence is kept separately as bounded preroll.
                    audio_chunks.clear()
                    continue

                if not current_window_has_voice and has_voice:
                    if leading_silence_preroll_chunks:
                        audio_chunks.extend(leading_silence_preroll_chunks)
                        logger.debug(
                            "[STT_LEADING_SILENCE] prepended preroll "
                            f"audio_seconds={leading_silence_preroll_total_bytes / bytes_per_second:.2f}"
                        )
                        leading_silence_preroll_chunks.clear()
                        leading_silence_preroll_total_bytes = 0
                    current_window_has_voice = True
                    trailing_silence_bytes = 0
                elif current_window_has_voice and not has_voice:
                    trailing_silence_bytes += len(audio_data)

                audio_chunks.append(audio_data)

                audio_bytes = b"".join(audio_chunks)
                should_flush_for_pause = (
                    current_window_has_voice
                    and pause_flush_bytes > 0
                    and trailing_silence_bytes >= pause_flush_bytes
                )
                should_flush_for_size = (
                    current_emit_policy == "live"
                    and len(audio_bytes) >= min_chunk_bytes
                )
                if should_flush_for_size or should_flush_for_pause:
                    if not await enqueue_audio_window(
                        audio_bytes,
                        force_emit=should_flush_for_pause,
                        finalization_reason=(
                            "pause" if should_flush_for_pause else None
                        ),
                    ):
                        break
        except WebSocketDisconnect:
            stop_event.set()
        except Exception as e:
            await result_queue.put(
                TranscribeResult(sequence=next_sequence, error=str(e))
            )
            stop_event.set()
        finally:
            for _ in range(max(1, TRANSCRIBE_WORKERS)):
                await job_queue.put(None)

    async def transcribe_worker() -> None:
        """Transcribe queued audio windows without blocking WebSocket receive."""
        while True:
            job = await job_queue.get()
            if job is None:
                break

            try:
                result = await asyncio.to_thread(_process_transcribe_job, model, job)
            except ValueError as e:
                if "empty sequence" in str(e).lower():
                    result = TranscribeResult(
                        sequence=job.sequence, skipped_silence=True
                    )
                else:
                    result = TranscribeResult(sequence=job.sequence, error=str(e))
            except Exception as e:
                result = TranscribeResult(sequence=job.sequence, error=str(e))

            if not result.queued_at:
                now = time.time()
                result.queued_at = job.queued_at
                result.started_at = result.started_at or now
                result.completed_at = result.completed_at or now
                result.audio_duration_seconds = (
                    len(job.audio_bytes) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH_BYTES)
                    if SAMPLE_RATE and CHANNELS and SAMPLE_WIDTH_BYTES
                    else 0.0
                )
                result.queue_depth_at_enqueue = job.queue_depth_at_enqueue
                result.backpressure_seconds = job.backpressure_seconds
                result.force_emit = job.force_emit
                result.finalization_reason = job.finalization_reason

            metrics = _result_metrics(result)
            logger.debug(
                "[STT_METRIC] "
                f"seq={metrics['sequence']} "
                f"audio={metrics['audio_duration_seconds']:.3f}s "
                f"queue_wait={metrics['queue_wait_seconds']:.3f}s "
                f"infer={metrics['inference_seconds']:.3f}s "
                f"total={metrics['total_job_seconds']:.3f}s "
                f"rms={metrics['audio_rms']:.5f} "
                f"depth={metrics['queue_depth_at_enqueue']} "
                f"backpressure={metrics['backpressure_seconds']:.3f}s "
                f"silence={metrics['skipped_silence']} "
                f"text_len={len(result.transcript)}"
            )
            await result_queue.put(result)

    receiver_task = asyncio.create_task(receive_audio())
    worker_tasks = [
        asyncio.create_task(transcribe_worker())
        for _ in range(max(1, TRANSCRIBE_WORKERS))
    ]

    full_transcript = ""
    pending_transcript = ""
    last_emit_time = time.time()
    next_emit_sequence = 0
    # Parallel workers can finish out of order; hold results until sequence order
    # is restored so the frontend and translation pipeline see monotonic text.
    pending_results: dict[int, TranscribeResult] = {}

    try:
        while True:
            if (
                receiver_task.done()
                and all(worker.done() for worker in worker_tasks)
                and result_queue.empty()
                and next_emit_sequence >= next_sequence
            ):
                break

            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if result.sequence < next_emit_sequence:
                logger.warning(
                    "[STT_ORDER] dropping stale result "
                    f"seq={result.sequence} next={next_emit_sequence}"
                )
                continue

            pending_results[result.sequence] = result
            if len(pending_results) > MAX_PENDING_RESULTS:
                oldest_sequence = min(pending_results)
                logger.warning(
                    "[STT_ORDER] advancing past missing result "
                    f"next={next_emit_sequence} oldest_pending={oldest_sequence} "
                    f"pending={len(pending_results)}"
                )
                next_emit_sequence = oldest_sequence

            while next_emit_sequence in pending_results:
                ordered_result = pending_results.pop(next_emit_sequence)
                next_emit_sequence += 1

                if ordered_result.error:
                    await websocket.send_json({"error": ordered_result.error})
                    stop_event.set()
                    continue

                if ordered_result.transcript.strip():
                    new_text = _get_new_transcript_text(
                        full_transcript, ordered_result.transcript
                    )
                    if not new_text:
                        if (
                            ordered_result.force_emit
                            and pending_transcript
                            and full_transcript
                        ):
                            emitted_at = time.time()
                            metrics = _result_metrics(ordered_result, emitted_at)
                            await websocket.send_json(
                                {
                                    "text": full_transcript,
                                    "is_final": False,
                                    "language": ordered_result.detected_language,
                                    "has_speech": ordered_result.has_speech,
                                    "metrics": metrics,
                                }
                            )
                            pending_transcript = ""
                            last_emit_time = emitted_at
                        continue

                    if _should_drop_no_speech_new_text(ordered_result, new_text):
                        logger.debug(
                            "[STT_EMIT] dropping no-speech transcript delta "
                            f"seq={ordered_result.sequence} "
                            f"new_text_len={len(new_text)}"
                        )
                        continue

                    collapsed_text = _collapse_repeated_word_sequences(new_text)
                    if collapsed_text != new_text:
                        logger.debug(
                            "[STT_EMIT] collapsed repeated transcript delta "
                            f"seq={ordered_result.sequence} "
                            f"from_len={len(new_text)} "
                            f"to_len={len(collapsed_text)}"
                        )
                        new_text = collapsed_text
                        if not new_text:
                            continue

                    trimmed_text = _trim_repeated_leading_words(
                        full_transcript, new_text
                    )
                    if trimmed_text != new_text:
                        logger.debug(
                            "[STT_EMIT] trimmed cross-delta repeated words "
                            f"seq={ordered_result.sequence} "
                            f"from_len={len(new_text)} "
                            f"to_len={len(trimmed_text)}"
                        )
                        new_text = trimmed_text
                        if not new_text:
                            continue

                    full_transcript = f"{full_transcript} {new_text}".strip()
                    pending_transcript = f"{pending_transcript} {new_text}".strip()

                    now = time.time()
                    should_emit = ordered_result.force_emit or (
                        current_emit_policy == "live"
                        and (
                            len(pending_transcript) >= EMIT_MIN_CHARS
                            or now - last_emit_time >= current_emit_interval_seconds
                        )
                    )

                    if should_emit:
                        emitted_at = time.time()
                        metrics = _result_metrics(ordered_result, emitted_at)
                        await websocket.send_json(
                            {
                                "text": full_transcript,
                                "is_final": False,
                                "language": ordered_result.detected_language,
                                "has_speech": ordered_result.has_speech,
                                "metrics": metrics,
                            }
                        )
                        logger.debug(
                            "[STT_EMIT] "
                            f"seq={metrics['sequence']} "
                            f"text_len={len(full_transcript)} "
                            f"pending_len={len(pending_transcript)} "
                            f"emit_delay={metrics['emit_delay_seconds']:.3f}s "
                            f"force_emit={ordered_result.force_emit}"
                        )
                        pending_transcript = ""
                        last_emit_time = emitted_at
                elif (
                    (current_emit_policy == "live" or ordered_result.force_emit)
                    and pending_transcript
                    and full_transcript
                ):
                    emitted_at = time.time()
                    metrics = _result_metrics(ordered_result, emitted_at)
                    await websocket.send_json(
                        {
                            "text": full_transcript,
                            "is_final": False,
                            "language": ordered_result.detected_language,
                            "has_speech": False,
                            "metrics": metrics,
                        }
                    )
                    pending_transcript = ""
                    last_emit_time = emitted_at
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        stop_event.set()
        receiver_task.cancel()
        for worker in worker_tasks:
            worker.cancel()

        await asyncio.gather(receiver_task, *worker_tasks, return_exceptions=True)
        await _safe_close_websocket(websocket)
