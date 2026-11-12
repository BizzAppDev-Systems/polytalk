# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Whisper transcription service.

Supports both real Whisper API and mock mode for testing.
"""

import asyncio
import io
import json
import math
import struct
import unicodedata
import wave
from collections import Counter, deque
from typing import AsyncGenerator, Callable, Optional

import websockets

from ..config import Config, get_config
from ..utils.config import parse_bool_config, parse_float_config, parse_int_config
from ..utils.logger import get_logger
from .audio_vad import (
    CLIENT_UTTERANCE_BOUNDARY,
    END_SIGNAL,
    VAD_UTTERANCE_BOUNDARY,
)
from .base import BaseTranscriptionService, TranscriptionResult

logger = get_logger(__name__)

_MIN_CONVERSATION_CONFIDENCE_MARGIN = 0.01

_LANGUAGE_SCRIPT_NAMES = {
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
_LATIN_SCRIPT_LANGUAGES = {
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
    script_names = _LANGUAGE_SCRIPT_NAMES.get(language)
    if script_names is None and language in _LATIN_SCRIPT_LANGUAGES:
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


def _pcm_rms(audio: bytes) -> float:
    """Return normalized RMS for mono signed 16-bit PCM."""
    usable_bytes = len(audio) - (len(audio) % 2)
    if usable_bytes <= 0:
        return 0.0
    sample_count = usable_bytes // 2
    square_sum = sum(
        sample * sample for (sample,) in struct.iter_unpack("<h", audio[:usable_bytes])
    )
    return math.sqrt(square_sum / sample_count) / 32768.0


class _ReplayableAudioStream:
    """Retain a bounded tail so a pre-result provider failure can be replayed.

    A provider may stop consuming primary() before the source is exhausted.
    In that case replay() yields the retained prefix and then resumes the
    unread remainder of the same source generator for the fallback provider.
    """

    def __init__(self, source: AsyncGenerator[bytes, None], max_buffer_bytes: int):
        self.source = source
        self.buffer: deque[bytes] = deque()
        self.max_buffer_bytes = max(0, max_buffer_bytes)
        self.buffered_bytes = 0
        self.exhausted = False
        self.replay_enabled = True
        self.overflow_logged = False

    def _remember(self, chunk: bytes) -> None:
        if not self.replay_enabled or self.max_buffer_bytes <= 0:
            return
        if len(chunk) >= self.max_buffer_bytes:
            self.buffer.clear()
            self.buffer.append(chunk[-self.max_buffer_bytes :])
            self.buffered_bytes = self.max_buffer_bytes
            self._log_overflow()
            return
        self.buffer.append(chunk)
        self.buffered_bytes += len(chunk)
        dropped = False
        while self.buffered_bytes > self.max_buffer_bytes and self.buffer:
            self.buffered_bytes -= len(self.buffer.popleft())
            dropped = True
        if dropped:
            self._log_overflow()

    def _log_overflow(self) -> None:
        if not self.overflow_logged:
            logger.warning(
                "STT fallback replay buffer reached its configured limit; "
                "oldest audio will not be retained"
            )
            self.overflow_logged = True

    def disable_replay(self) -> None:
        """Release retained audio once fallback is no longer eligible."""
        self.replay_enabled = False
        self.buffer.clear()
        self.buffered_bytes = 0

    async def primary(self) -> AsyncGenerator[bytes, None]:
        async for chunk in self.source:
            self._remember(chunk)
            yield chunk
        self.exhausted = True

    async def replay(self) -> AsyncGenerator[bytes, None]:
        for chunk in self.buffer:
            yield chunk
        if not self.exhausted:
            async for chunk in self.source:
                yield chunk
            self.exhausted = True


class WhisperService(BaseTranscriptionService):
    """
    Whisper transcription service using faster-whisper.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize Whisper service with configuration."""
        self.config = (config or get_config()).whisper
        self.enabled = self.config.get("enabled", True)
        self.mock_mode = self.config.get("mock_mode", True)
        self.base_url = self.config.get("base_url", "http://stt:8000")
        self.ws_endpoint = self.config.get("ws_endpoint", "/v1/stream/transcriptions")
        self.api_key = self.config.get("api_key", "")
        self.model = self.config.get("model", "whisper-1")
        self.timeout = self.config.get("timeout_seconds", 60)
        self.max_reconnect_attempts = self.config.get("max_reconnect_attempts", 3)
        self.reconnect_delay = self.config.get("reconnect_delay_seconds", 2)
        self.ping_interval = self.config.get("ping_interval_seconds", None)
        self.ping_timeout = self.config.get("ping_timeout_seconds", None)
        self.providers = self.config.get("providers", {})
        self.routing = self.config.get("routing", [])
        self.fallback_provider = self.config.get("fallback_provider", "whisper")
        self.routing_enabled = parse_bool_config(
            self.config.get("routing_enabled", False), False
        )
        replay_seconds = max(
            0.0,
            parse_float_config(self.config.get("fallback_replay_max_seconds"), 30.0),
        )
        sample_rate = max(1, parse_int_config(self.config.get("sample_rate"), 16000))
        self.fallback_replay_max_bytes = int(replay_seconds * sample_rate * 2)

        conversation = self.config.get("conversation", {})
        self.conversation_detector_provider = str(
            conversation.get("detector_provider") or self.fallback_provider
        )
        self.conversation_parallel_timeout_seconds = max(
            0.1,
            parse_float_config(
                conversation.get("parallel_timeout_seconds"),
                15.0,
            ),
        )
        self.conversation_confidence_margin = min(
            1.0,
            max(
                _MIN_CONVERSATION_CONFIDENCE_MARGIN,
                parse_float_config(conversation.get("confidence_margin"), 0.10),
            ),
        )
        self.conversation_silence_rms_threshold = max(
            0.0,
            parse_float_config(conversation.get("silence_rms_threshold"), 0.003),
        )
        self.conversation_pause_seconds = max(
            0.0,
            parse_float_config(conversation.get("pause_seconds"), 2.0),
        )
        self.conversation_max_utterance_seconds = max(
            self.conversation_pause_seconds,
            parse_float_config(
                conversation.get("max_utterance_seconds"),
                30.0,
            ),
        )
        self.conversation_sample_rate = sample_rate
        fallback_languages = conversation.get("fallback_languages", [])
        if not isinstance(fallback_languages, list):
            fallback_languages = []
        self.conversation_fallback_languages = {
            normalized
            for item in fallback_languages
            if (normalized := self._normalize_language_hint(str(item)))
        }

    async def stream_transcribe(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
        emit_policy: str = "live",
        candidate_languages: Optional[list[str]] = None,
        emit_interval_seconds: Optional[float] = None,
        emit_interval_config_queue: Optional[asyncio.Queue] = None,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
        input_segmentation: Optional[str] = None,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Stream audio chunks for real-time transcription.

        Args:
            audio_generator: Async generator yielding audio chunks
            language: Optional source language code hint
            emit_policy: STT emission policy, either live or pause
            candidate_languages: Optional language candidates for detection
            on_result: Optional callback for each transcription result

        Yields:
            TranscriptionResult with incremental transcription updates
        """
        if not self.enabled:
            logger.warning("Whisper service is disabled")
            result = TranscriptionResult(
                text="", success=False, error="Whisper service is disabled"
            )
            yield result
            if on_result:
                on_result(result)
            return

        if self.mock_mode:
            logger.info("Using mock streaming transcription")
            async for result in self._mock_stream_transcribe(audio_generator, language):
                yield result
                if on_result:
                    on_result(result)
            return

        provider = self._get_provider_for_language(language)
        replayable = _ReplayableAudioStream(
            audio_generator, self.fallback_replay_max_bytes
        )
        emitted_result = False
        pending_error: Optional[TranscriptionResult] = None

        try:
            async for result in self._stream_with_provider(
                provider,
                replayable.primary(),
                language,
                emit_policy,
                candidate_languages,
                emit_interval_seconds,
                emit_interval_config_queue,
                **(
                    {"input_segmentation": input_segmentation}
                    if input_segmentation
                    else {}
                ),
            ):
                if not result.success:
                    pending_error = result
                    continue
                emitted_result = True
                replayable.disable_replay()
                yield result
                if on_result:
                    on_result(result)
        except Exception as exc:
            pending_error = TranscriptionResult(text="", success=False, error=str(exc))

        if pending_error and provider != self.fallback_provider and not emitted_result:
            logger.warning(
                "STT backend failed before emitting text; falling back: "
                f"selected={provider} fallback={self.fallback_provider} "
                f"language={language} error={pending_error.error}"
            )
            try:
                async for result in self._stream_with_provider(
                    self.fallback_provider,
                    replayable.replay(),
                    language,
                    emit_policy,
                    candidate_languages,
                    emit_interval_seconds,
                    emit_interval_config_queue,
                    fallback_reason=pending_error.error,
                    fallback_provider=provider,
                    **(
                        {"input_segmentation": input_segmentation}
                        if input_segmentation
                        else {}
                    ),
                ):
                    yield result
                    if on_result:
                        on_result(result)
                return
            except Exception as exc:
                pending_error = TranscriptionResult(
                    text="", success=False, error=f"STT fallback failed: {exc}"
                )

        if pending_error:
            logger.error(f"Streaming transcription failed: {pending_error.error}")
            yield pending_error
            if on_result:
                on_result(pending_error)

    def _get_routed_provider_for_language(
        self, language: Optional[str]
    ) -> Optional[str]:
        """Return an explicitly configured and enabled provider for a language."""
        if not self.routing_enabled or not language:
            return None

        language_code = self._normalize_language_hint(language)
        for rule in sorted(
            (item for item in self.routing if isinstance(item, dict)),
            key=lambda item: int(item.get("priority", 100)),
        ):
            provider = str(rule.get("provider", "")).strip()
            languages = {
                self._normalize_language_hint(str(item))
                for item in rule.get("languages", [])
            }
            provider_config = self.providers.get(provider, {})
            if (
                provider
                and language_code in languages
                and parse_bool_config(provider_config.get("enabled", True), True)
            ):
                return provider
        return None

    def _get_provider_for_language(self, language: Optional[str]) -> str:
        """Resolve an explicit source language to a configured STT backend."""
        return self._get_routed_provider_for_language(language) or str(
            self.fallback_provider
        )

    def _get_conversation_provider_for_language(
        self, language: Optional[str]
    ) -> Optional[str]:
        """Resolve a conversation language without an unconditional fallback."""
        provider = self._get_routed_provider_for_language(language)
        if provider:
            return provider

        language_code = self._normalize_language_hint(language)
        fallback = str(self.fallback_provider)
        fallback_config = self.providers.get(fallback, {})
        if language_code in self.conversation_fallback_languages and parse_bool_config(
            fallback_config.get("enabled", True), True
        ):
            return fallback
        return None

    def _provider_detects_language(self, provider: str) -> bool:
        """Return whether a provider independently detects spoken language."""
        provider_config = self.providers.get(provider, {})
        return parse_bool_config(
            provider_config.get("language_detection", False),
            False,
        )

    def _provider_uses_language_hint(self, provider: str) -> bool:
        """Return whether changing the language hint changes provider decoding."""
        provider_config = self.providers.get(provider, {})
        return parse_bool_config(
            provider_config.get("language_hint_supported", True),
            True,
        )

    def _conversation_candidate_specs(
        self,
        source_language: str,
        target_language: str,
    ) -> list[dict]:
        """Build the minimal candidate set for one selected language pair."""
        detector_config = self.providers.get(self.conversation_detector_provider, {})
        if not detector_config or not parse_bool_config(
            detector_config.get("enabled", True), True
        ):
            raise ValueError(
                "Conversation language detector is not configured and enabled: "
                f"{self.conversation_detector_provider}"
            )
        if not self._provider_detects_language(self.conversation_detector_provider):
            raise ValueError(
                "Conversation language detector must provide independent language "
                f"detection: {self.conversation_detector_provider}"
            )

        languages = list(
            dict.fromkeys(
                normalized
                for language in (source_language, target_language)
                if (normalized := self._normalize_language_hint(language))
            )
        )
        routes = [
            (language, self._get_conversation_provider_for_language(language))
            for language in languages
        ]
        missing = [language for language, provider in routes if not provider]
        if missing:
            raise ValueError(
                "No conversation STT provider or permitted fallback configured for: "
                + ", ".join(missing)
            )

        providers = {provider for _, provider in routes}
        if len(providers) == 1:
            provider = next(iter(providers))
            if self._provider_detects_language(
                provider
            ) or not self._provider_uses_language_hint(provider):
                return [
                    {
                        "provider": provider,
                        "candidate_language": None,
                        "candidate_languages": languages,
                    }
                ]

        return [
            {
                "provider": provider,
                "candidate_language": language,
                "candidate_languages": [],
            }
            for language, provider in routes
        ]

    async def _conversation_utterances(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        *,
        input_segmentation: Optional[str] = None,
    ) -> AsyncGenerator[tuple[bytes, ...], None]:
        """Split normalized PCM into bounded, replayable utterances.

        The WebSocket audio contract is mono signed int16 PCM at the configured
        sample rate. Raw PCM carries no sample-rate metadata, so callers must
        resample before yielding chunks to this method.
        """
        chunks: list[bytes] = []
        total_bytes = 0
        upstream_segmented = input_segmentation == "upstream_vad"
        speech_started = False
        trailing_silence_bytes = 0
        bytes_per_second = self.conversation_sample_rate * 2
        pause_bytes = int(bytes_per_second * self.conversation_pause_seconds)
        max_bytes = int(bytes_per_second * self.conversation_max_utterance_seconds)

        async for chunk in audio_generator:
            if chunk == END_SIGNAL:
                if chunks:
                    yield tuple(chunks)
                return
            if chunk in {CLIENT_UTTERANCE_BOUNDARY, VAD_UTTERANCE_BOUNDARY}:
                if chunks:
                    yield tuple(chunks)
                chunks = []
                total_bytes = 0
                speech_started = False
                trailing_silence_bytes = 0
                continue
            if not chunk:
                continue

            has_voice = upstream_segmented or (
                _pcm_rms(chunk) >= self.conversation_silence_rms_threshold
            )
            if not speech_started and not has_voice:
                continue

            speech_started = speech_started or has_voice
            trailing_silence_bytes = (
                0 if has_voice else trailing_silence_bytes + len(chunk)
            )
            chunks.append(chunk)
            total_bytes += len(chunk)

            if speech_started and (
                (
                    not upstream_segmented
                    and pause_bytes > 0
                    and trailing_silence_bytes >= pause_bytes
                )
                or (max_bytes > 0 and total_bytes >= max_bytes)
            ):
                yield tuple(chunks)
                chunks = []
                total_bytes = 0
                speech_started = False
                trailing_silence_bytes = 0

        if chunks:
            yield tuple(chunks)

    @staticmethod
    async def _replay_utterance(
        chunks: tuple[bytes, ...],
    ) -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk
        yield END_SIGNAL

    async def _collect_conversation_candidate(
        self,
        spec: dict,
        chunks: tuple[bytes, ...],
    ) -> dict:
        """Transcribe one utterance with one provider/language hypothesis."""
        provider = str(spec["provider"])
        candidate_language = spec.get("candidate_language")
        last_success: Optional[TranscriptionResult] = None
        last_error: Optional[str] = None

        async for result in self._stream_with_provider(
            provider,
            self._replay_utterance(chunks),
            candidate_language,
            "pause",
            spec.get("candidate_languages") or None,
            input_segmentation="upstream_vad",
        ):
            if result.success and result.text.strip():
                last_success = result
            elif not result.success:
                last_error = result.error

        if last_success is None:
            return {
                **spec,
                "result": None,
                "error": last_error or "STT candidate returned no transcript",
                "outcome": "provider_error",
            }

        metrics = dict(last_success.metrics or {})
        independently_detected = self._normalize_language_hint(
            metrics.get("detected_language")
        )
        normalized_candidate = self._normalize_language_hint(candidate_language)
        confidence = metrics.get("normalized_confidence")
        if confidence is None:
            confidence = metrics.get("provider_confidence")
        try:
            normalized_confidence = float(confidence)
        except (TypeError, ValueError):
            normalized_confidence = None

        script_language = (
            independently_detected
            or normalized_candidate
            or self._normalize_language_hint(last_success.language)
        )
        script_score = (
            _language_script_match_ratio(last_success.text, script_language)
            if script_language
            else None
        )
        # A script match describes the returned text, not which spoken
        # language produced it. Keep it for diagnostics, but never treat it as
        # model confidence when choosing a conversation direction.

        return {
            **spec,
            "candidate_language": normalized_candidate,
            "detected_language": independently_detected,
            "normalized_confidence": normalized_confidence,
            "script_score": script_score,
            "result": last_success,
            "error": None,
            "outcome": "success",
        }

    def _select_conversation_candidate(
        self,
        candidates: list[dict],
        source_language: str,
        target_language: Optional[str] = None,
        detector: Optional[dict] = None,
    ) -> TranscriptionResult:
        """Choose one transcript using independent LID, then safe fallbacks."""
        valid = [
            candidate
            for candidate in candidates
            if candidate.get("result") is not None and candidate["result"].text.strip()
        ]
        candidate_outcomes = Counter(
            candidate.get("outcome")
            or ("success" if candidate.get("result") is not None else "provider_error")
            for candidate in candidates
        )
        pair_languages = {
            language
            for value in (source_language, target_language)
            if (language := self._normalize_language_hint(value))
        }
        detector_result = detector.get("result") if detector else None
        detector_language = detector.get("detected_language") if detector else None
        detector_valid = bool(
            detector_result is not None
            and detector_result.text.strip()
            and detector_language in pair_languages
        )

        if not valid and not detector_valid:
            errors = [
                str(candidate.get("error"))
                for candidate in candidates
                if candidate.get("error")
            ]
            logger.warning(
                "Conversation STT utterance failed: candidate_outcomes=%s candidates=%d",
                dict(candidate_outcomes),
                len(candidates),
            )
            return TranscriptionResult(
                text="",
                success=False,
                error="; ".join(errors) or "No conversation STT candidate succeeded",
                metrics={
                    "conversation_utterance": True,
                    "selection_reason": "no_valid_candidate",
                    "conversation_candidate_count": len(candidates),
                    "conversation_candidate_outcomes": dict(candidate_outcomes),
                },
            )

        selected: Optional[dict] = None
        selection_reason = "only_valid_candidate"

        if detector_valid:
            configured_provider = self._get_conversation_provider_for_language(
                detector_language
            )
            selected = next(
                (
                    candidate
                    for candidate in valid
                    if candidate.get("provider") == configured_provider
                    and (
                        candidate.get("candidate_language") in {None, detector_language}
                        or detector_language
                        in (candidate.get("candidate_languages") or [])
                    )
                ),
                None,
            )
            if selected is not None:
                selection_reason = "detector_language"
            else:
                # The detector is also a complete transcript. Use it if the
                # configured provider for the detected language failed.
                selected = detector
                selection_reason = "detector_transcript_fallback"

        detected = [
            candidate.get("detected_language")
            for candidate in valid
            if candidate.get("detected_language")
        ]
        if selected is None and len(detected) == len(valid) and len(set(detected)) == 1:
            agreed_language = detected[0]
            configured_provider = self._get_conversation_provider_for_language(
                agreed_language
            )
            selected = next(
                (
                    candidate
                    for candidate in valid
                    if candidate["provider"] == configured_provider
                ),
                None,
            )
            if selected is not None:
                selection_reason = "detected_language_agreement"

        if selected is None and len(valid) > 1:
            scored = [
                candidate
                for candidate in valid
                if candidate.get("normalized_confidence") is not None
            ]
            if len(scored) > 1:
                scored.sort(
                    key=lambda candidate: candidate["normalized_confidence"],
                    reverse=True,
                )
                confidence_gap = (
                    scored[0]["normalized_confidence"]
                    - scored[1]["normalized_confidence"]
                )
                if confidence_gap >= self.conversation_confidence_margin:
                    selected = scored[0]
                    selection_reason = "confidence_margin"

        source_code = self._normalize_language_hint(source_language)
        if selected is None:
            selected = next(
                (
                    candidate
                    for candidate in valid
                    if candidate.get("candidate_language") == source_code
                ),
                valid[0],
            )
            selection_reason = "configured_default"

        result = selected["result"]
        metrics = dict(result.metrics or {})
        metrics.update(
            {
                "backend": selected["provider"],
                "candidate_language": selected.get("candidate_language"),
                "detected_language": selected.get("detected_language"),
                "normalized_confidence": selected.get("normalized_confidence"),
                "script_score": selected.get("script_score"),
                "selection_reason": selection_reason,
                "conversation_candidate_count": len(candidates),
                "conversation_candidate_outcomes": dict(candidate_outcomes),
                "conversation_detector_backend": (
                    detector.get("provider") if detector else None
                ),
                "conversation_detector_language": detector_language,
                "conversation_detector_outcome": (
                    detector.get("outcome") if detector else "not_run"
                ),
                "conversation_utterance": True,
            }
        )
        result.metrics = metrics
        result.language = (
            (detector_language if detector_valid else None)
            or selected.get("detected_language")
            or selected.get("candidate_language")
            or result.language
            or source_code
        )
        result.is_partial = False
        logger.info(
            "Conversation STT utterance selected: backend=%s "
            "selection_reason=%s language=%s confidence=%s detector=%s "
            "detector_outcome=%s candidates=%d candidate_outcomes=%s",
            selected["provider"],
            selection_reason,
            result.language,
            selected.get("normalized_confidence"),
            detector_language,
            detector.get("outcome") if detector else "not_run",
            len(candidates),
            dict(candidate_outcomes),
        )
        return result

    async def _transcribe_conversation_utterance(
        self,
        chunks: tuple[bytes, ...],
        source_language: str,
        target_language: str,
        specs: list[dict],
    ) -> TranscriptionResult:
        """Run independent LID and configured transcriptions concurrently."""
        detector_spec = {
            "provider": self.conversation_detector_provider,
            "candidate_language": None,
            # Use one automatic faster-whisper pass for both LID and the
            # fallback transcript. Supplying the selected pair here can make
            # the STT service perform two additional forced-language decodes
            # whenever auto-detection is below its confidence threshold.
            "candidate_languages": [],
        }
        all_specs = [detector_spec, *specs]
        tasks = [
            asyncio.create_task(self._collect_conversation_candidate(spec, chunks))
            for spec in all_specs
        ]
        done, pending = await asyncio.wait(
            tasks,
            timeout=self.conversation_parallel_timeout_seconds,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        collected: list[dict] = []
        for index, task in enumerate(tasks):
            if task in done:
                try:
                    collected.append(task.result())
                except Exception as exc:
                    collected.append(
                        {
                            **all_specs[index],
                            "result": None,
                            "error": str(exc),
                            "outcome": "provider_error",
                        }
                    )
            else:
                collected.append(
                    {
                        **all_specs[index],
                        "result": None,
                        "error": "STT candidate timed out",
                        "outcome": "timeout",
                    }
                )

        detector = collected[0]
        candidates = collected[1:]
        return self._select_conversation_candidate(
            candidates,
            source_language,
            target_language,
            detector,
        )

    async def stream_transcribe_conversation(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        source_language: str,
        target_language: str,
        *,
        input_segmentation: Optional[str] = None,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Transcribe conversation turns with pair-aware parallel routing."""
        if self.mock_mode or not self.routing_enabled:
            async for result in self.stream_transcribe(
                audio_generator,
                None,
                emit_policy="pause",
                candidate_languages=[source_language, target_language],
                input_segmentation=input_segmentation,
            ):
                yield result
                if on_result:
                    on_result(result)
            return

        try:
            specs = self._conversation_candidate_specs(
                source_language,
                target_language,
            )
        except ValueError as exc:
            result = TranscriptionResult(text="", success=False, error=str(exc))
            yield result
            if on_result:
                on_result(result)
            return

        logger.info(
            "Conversation STT detector=%s candidates=%s",
            self.conversation_detector_provider,
            [
                {
                    "provider": spec["provider"],
                    "language": spec.get("candidate_language") or "auto",
                }
                for spec in specs
            ],
        )
        async for chunks in self._conversation_utterances(
            audio_generator,
            input_segmentation=input_segmentation,
        ):
            result = await self._transcribe_conversation_utterance(
                chunks,
                source_language,
                target_language,
                specs,
            )
            yield result
            if on_result:
                on_result(result)

    def _get_provider_config(self, provider: str) -> dict:
        """Return provider settings merged with legacy Whisper defaults."""
        return {**self.config, **self.providers.get(provider, {})}

    async def _stream_with_provider(
        self,
        provider: str,
        audio_generator: AsyncGenerator[bytes, None],
        language: Optional[str],
        emit_policy: str,
        candidate_languages: Optional[list[str]],
        emit_interval_seconds: Optional[float] = None,
        emit_interval_config_queue: Optional[asyncio.Queue] = None,
        fallback_reason: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        input_segmentation: Optional[str] = None,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Stream through one provider and annotate normalized metrics."""
        logger.info(
            "STT route selected: "
            f"backend={provider} requested_language={language or 'auto'} "
            f"fallback_reason={fallback_reason or 'none'}"
        )
        async for result in self._real_stream_transcribe(
            audio_generator,
            language,
            emit_policy=emit_policy,
            candidate_languages=candidate_languages,
            emit_interval_seconds=emit_interval_seconds,
            emit_interval_config_queue=emit_interval_config_queue,
            provider=provider,
            provider_config=self._get_provider_config(provider),
            **(
                {"input_segmentation": input_segmentation} if input_segmentation else {}
            ),
        ):
            metrics = dict(result.metrics or {})
            metrics["backend"] = provider
            if fallback_reason:
                metrics["fallback_reason"] = fallback_reason
                if fallback_provider:
                    metrics["fallback_provider"] = fallback_provider
                if result.success:
                    logger.info(
                        "STT fallback provider succeeded: "
                        f"from={fallback_provider or 'unknown'} to={provider}"
                    )
            result.metrics = metrics
            yield result

    def _normalize_language_hint(self, language: Optional[str]) -> Optional[str]:
        """
        Normalize UI language codes to ASR language hints.

        Faster Whisper expects base language codes such as "es" or "nl", while
        the UI can use regional codes such as "es_MX" for translation/TTS.
        """
        if not language:
            return None
        return language.replace("-", "_").split("_")[0].lower()

    async def _real_stream_transcribe(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
        emit_policy: str = "live",
        candidate_languages: Optional[list[str]] = None,
        emit_interval_seconds: Optional[float] = None,
        emit_interval_config_queue: Optional[asyncio.Queue] = None,
        provider: str = "whisper",
        provider_config: Optional[dict] = None,
        input_segmentation: Optional[str] = None,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Stream audio chunks using WebSocket for real-time transcription.

        Args:
            audio_generator: Async generator yielding audio chunks
            language: Optional source language code hint
            emit_policy: STT emission policy, either live or pause
            candidate_languages: Optional language candidates for detection

        Yields:
            TranscriptionResult with incremental transcription updates
        """
        active_config = provider_config or self.config
        base_url = active_config.get("base_url", self.base_url)
        ws_endpoint = active_config.get("ws_endpoint", self.ws_endpoint)
        api_key = active_config.get("api_key", self.api_key)

        # Convert https:// to wss:// and http:// to ws://
        ws_base = base_url.replace("https://", "wss://", 1).replace(
            "http://", "ws://", 1
        )
        ws_url = ws_base.rstrip("/") + ws_endpoint

        # Add Authorization header if API key is configured
        additional_headers = {}
        if api_key:
            additional_headers["Authorization"] = f"Bearer {api_key}"
            logger.info(f"Using API key authentication for WebSocket: {ws_url}")

        reconnect_attempts = 0
        full_text = ""
        language_hint = self._normalize_language_hint(language)

        while reconnect_attempts < self.max_reconnect_attempts:
            ws = None
            try:
                logger.info(f"Connecting to WebSocket: {ws_url}")
                ws = await websockets.connect(
                    ws_url,
                    additional_headers=additional_headers,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                )
                logger.info(f"STT WebSocket connection established: backend={provider}")

                control_message = {}
                if language_hint:
                    control_message["language"] = language_hint
                if emit_policy:
                    control_message["emit_policy"] = emit_policy
                if candidate_languages:
                    control_message["candidate_languages"] = [
                        self._normalize_language_hint(item)
                        for item in candidate_languages
                        if item
                    ]
                if emit_interval_seconds is not None:
                    control_message["emit_interval_seconds"] = emit_interval_seconds
                if input_segmentation == "upstream_vad":
                    control_message["input_segmentation"] = input_segmentation
                if control_message:
                    await ws.send(json.dumps(control_message))

                # Use asyncio Queue to communicate between tasks
                result_queue = asyncio.Queue()
                send_done = asyncio.Event()
                recv_done = asyncio.Event()
                send_task = None
                recv_task = None

                async def send_chunks():
                    """Send audio chunks and explicitly flush the final window."""
                    nonlocal send_done
                    end_sent = False

                    async def send_interval_update() -> None:
                        if emit_interval_config_queue is None:
                            return
                        latest_seconds = None
                        while True:
                            try:
                                latest_seconds = emit_interval_config_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        if latest_seconds is not None:
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "emit_interval_config",
                                        "emit_interval_seconds": latest_seconds,
                                    }
                                )
                            )

                    try:
                        async for audio_chunk in audio_generator:
                            await send_interval_update()
                            if audio_chunk == b"__END_SIGNAL__":
                                logger.info(
                                    "Received end signal from audio generator, flushing STT"
                                )
                                await ws.send(json.dumps({"type": "end"}))
                                end_sent = True
                                break

                            if audio_chunk in {
                                CLIENT_UTTERANCE_BOUNDARY,
                                VAD_UTTERANCE_BOUNDARY,
                            }:
                                reason = (
                                    "vad_pause"
                                    if audio_chunk == VAD_UTTERANCE_BOUNDARY
                                    else "client_flush"
                                )
                                logger.info(
                                    "Received utterance boundary, requesting STT flush: reason=%s",
                                    reason,
                                )
                                await ws.send(
                                    json.dumps({"type": "flush", "reason": reason})
                                )
                                continue

                            try:
                                await ws.send(audio_chunk)
                            except Exception as e:
                                logger.error(f"Error sending chunk: {e}")
                                break
                    except asyncio.CancelledError:
                        logger.info("send_chunks task cancelled")
                    except GeneratorExit:
                        logger.info("send_chunks GeneratorExit received")
                    except Exception as e:
                        logger.error(f"Send task error: {e}")
                    finally:
                        if not end_sent:
                            try:
                                await ws.send(json.dumps({"type": "end"}))
                            except Exception as e:
                                logger.debug(f"Could not send STT end control: {e}")
                        # Keep the socket open here. The receive task collects
                        # the final transcript and observes provider teardown.
                        send_done.set()

                async def receive_responses():
                    """Receive transcription responses from ASR WebSocket."""
                    nonlocal recv_done
                    try:
                        while True:
                            response = await ws.recv()
                            result_data = json.loads(response)

                            if result_data.get("type") == "emit_policy_ack":
                                logger.debug(
                                    "STT emit policy acknowledged: "
                                    f"{result_data.get('emit_policy')}"
                                )
                                continue
                            if result_data.get("type") == "candidate_languages_ack":
                                logger.debug(
                                    "STT candidate languages acknowledged: "
                                    f"{result_data.get('candidate_languages')}"
                                )
                                continue
                            if result_data.get("type") == "emit_interval_ack":
                                logger.debug(
                                    "STT emit interval acknowledged: "
                                    f"{result_data.get('emit_interval_seconds')}"
                                )
                                continue

                            text = result_data.get("text", "")
                            is_final = result_data.get("is_final", False)
                            error = result_data.get("error")
                            metrics = dict(result_data.get("metrics") or {})
                            reported_language = result_data.get("language")
                            independently_detected = result_data.get(
                                "detected_language"
                            )
                            if independently_detected is None and not language_hint:
                                independently_detected = reported_language
                            metrics["detected_language"] = independently_detected

                            if error:
                                await result_queue.put(
                                    {"type": "error", "error": error}
                                )
                                continue

                            # Send the text as-is (incremental update from ASR)
                            if text:
                                await result_queue.put(
                                    {
                                        "type": "result",
                                        "text": text,
                                        "is_partial": not is_final,
                                        "language": reported_language,
                                        "metrics": metrics,
                                    }
                                )
                    except asyncio.CancelledError:
                        logger.info("receive_responses task cancelled")
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("ASR WebSocket closed")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse response: {e}")
                    except Exception as e:
                        logger.error(f"Error receiving response: {e}")
                    finally:
                        recv_done.set()

                # Run both tasks concurrently
                send_task = asyncio.create_task(send_chunks())
                recv_task = asyncio.create_task(receive_responses())

                # Collect results from queue
                try:
                    while not (send_done.is_set() and recv_done.is_set()):
                        try:
                            item = await asyncio.wait_for(
                                result_queue.get(), timeout=1.0
                            )
                            if item is None:
                                continue

                            if item["type"] == "error":
                                yield TranscriptionResult(
                                    text="", success=False, error=item["error"]
                                )
                            elif item["type"] == "result":
                                yield TranscriptionResult(
                                    text=item["text"],
                                    language=item.get("language") or language,
                                    success=True,
                                    is_partial=item["is_partial"],
                                    metrics=item.get("metrics"),
                                )
                        except asyncio.TimeoutError:
                            continue

                    logger.info("Audio streaming complete")
                except GeneratorExit:
                    logger.info("GeneratorExit received in stream_transcribe")
                    raise
                finally:
                    # Cleanup: cancel tasks and close WebSocket
                    logger.info("Cleaning up Whisper WebSocket...")

                    if send_task and not send_task.done():
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logger.debug(f"Send task error on cancel: {e}")
                    if recv_task and not recv_task.done():
                        recv_task.cancel()
                        try:
                            await recv_task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logger.debug(f"Recv task error on cancel: {e}")
                    if ws:
                        try:
                            await ws.close()
                        except Exception as e:
                            logger.debug(f"Error closing WebSocket: {e}")
                    return

            except websockets.exceptions.ConnectionClosed as e:
                reconnect_attempts += 1
                logger.warning(
                    f"WebSocket connection closed (attempt {reconnect_attempts}/{self.max_reconnect_attempts}): {e}"
                )
                if reconnect_attempts < self.max_reconnect_attempts:
                    logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    logger.error("Max reconnect attempts reached")
                    yield TranscriptionResult(
                        text=full_text,
                        language=language,
                        success=False,
                        error="Connection lost after max reconnect attempts",
                    )
                    raise

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                yield TranscriptionResult(
                    text=full_text,
                    language=language,
                    success=False,
                    error=str(e),
                )
                raise

    def _estimate_audio_duration(self, audio_bytes: bytes) -> float:
        """
        Estimate audio duration from raw bytes.

        Args:
            audio_bytes: Raw audio data

        Returns:
            Estimated duration in seconds
        """
        try:
            wav_file = io.BytesIO(audio_bytes)
            with wave.open(wav_file, "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                if rate > 0:
                    return frames / rate
        except Exception as e:
            logger.debug(f"Could not estimate duration: {e}")

        return 5.0

    async def _mock_stream_transcribe(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Mock streaming transcription for testing.

        Args:
            audio_generator: Async generator yielding audio chunks (ignored in mock mode)
            language: Optional source language code hint
            emit_policy: STT emission policy, either live or pause
            candidate_languages: Optional language candidates for detection

        Yields:
            Mock TranscriptionResult with incremental updates
        """
        mock_texts = {
            "en": [
                "Hello",
                ", how are you",
                " doing today",
                "? I would like to test",
                " the speech to text functionality",
                ".",
            ],
            "gu": [
                "હેલો",
                ", તમે કેમ છો",
                "? હું સ્પીચ ટુ ટેક્સ્ટ",
                " ફંક્શનલિટી ટેસ્ટ કરવા માંગું છું",
                ".",
            ],
            "hi": [
                "नमस्ते",
                ", आप कैसे हैं",
                "? मैं स्पीच टु टेक्स्ट",
                " फंक्शनलिटी टेस्ट करना चाहता हूं",
                ".",
            ],
        }

        detected_lang = self._normalize_language_hint(language) or "en"
        text_chunks = mock_texts.get(detected_lang, mock_texts["en"])

        full_text = ""
        for i, chunk in enumerate(text_chunks):
            full_text += chunk
            is_final = i == len(text_chunks) - 1

            yield TranscriptionResult(
                text=full_text,
                language=detected_lang,
                success=True,
                is_partial=not is_final,
            )

            if not is_final:
                await asyncio.sleep(0.5)

        logger.info(
            "Mock streaming transcription complete: language=%s chars=%d",
            detected_lang,
            len(full_text),
        )

    async def close(self) -> None:
        """
        Close the service and cleanup resources.
        """
        logger.info("WhisperService closed")
