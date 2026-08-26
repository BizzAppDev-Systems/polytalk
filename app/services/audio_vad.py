# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared voice activity detection for streaming PCM audio.

The stream state machine is deliberately independent from the inference
backend. This keeps byte-order, preroll, endpoint, and failure-open behavior
testable without loading an ONNX model.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional, Protocol

from ..config import Config, get_config
from ..utils.config import parse_float_config, parse_int_config
from ..utils.logger import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BYTES = 2
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH_BYTES
FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE

END_SIGNAL = b"__END_SIGNAL__"
CLIENT_UTTERANCE_BOUNDARY = b"__END_OF_UTTERANCE__"
VAD_UTTERANCE_BOUNDARY = b"__VAD_END_OF_UTTERANCE__"
VAD_RESET = b"__VAD_RESET__"


class VadMode(str, Enum):
    """Supported rollout modes for one VAD profile."""

    OFF = "off"
    SHADOW = "shadow"
    BOUNDARY = "boundary"
    ACTIVE = "active"

    @classmethod
    def parse(cls, value: object) -> "VadMode":
        """Return a safe rollout mode, defaulting unknown values to off."""
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.OFF


class SpeechProbabilityStream(Protocol):
    """Per-connection probability predictor used by the VAD state machine."""

    def predict(self, pcm_frame: bytes) -> float:
        """Return the speech probability for one 512-sample PCM16 frame."""

    def reset(self) -> None:
        """Reset recurrent model state for a new utterance/session."""


@dataclass(frozen=True)
class VadProfile:
    """Timing and threshold settings for one capture profile."""

    mode: VadMode = VadMode.OFF
    positive_threshold: float = 0.5
    negative_threshold: float = 0.35
    pre_roll_ms: int = 640
    start_confirm_ms: int = 96
    post_roll_ms: int = 320
    end_hangover_ms: int = 960
    max_utterance_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.negative_threshold < self.positive_threshold <= 1.0:
            raise ValueError(
                "VAD thresholds must satisfy 0 <= negative < positive <= 1"
            )

    @staticmethod
    def _frames_for_ms(milliseconds: int, *, minimum: int = 0) -> int:
        return max(minimum, math.ceil(max(0, milliseconds) / FRAME_MS))

    @property
    def pre_roll_frames(self) -> int:
        return self._frames_for_ms(self.pre_roll_ms, minimum=1)

    @property
    def start_confirm_frames(self) -> int:
        return self._frames_for_ms(self.start_confirm_ms, minimum=1)

    @property
    def post_roll_frames(self) -> int:
        return self._frames_for_ms(self.post_roll_ms)

    @property
    def end_hangover_frames(self) -> int:
        return self._frames_for_ms(self.end_hangover_ms, minimum=1)

    @property
    def max_utterance_frames(self) -> int:
        if self.max_utterance_seconds <= 0:
            return 0
        return max(
            1,
            math.ceil(self.max_utterance_seconds * 1000 / FRAME_MS),
        )


class _VadState(str, Enum):
    IDLE = "idle"
    POSSIBLE_SPEECH = "possible_speech"
    SPEECH = "speech"
    POSSIBLE_END = "possible_end"


class VadStreamSegmenter:
    """Segment one stream without dropping bytes that might contain speech."""

    def __init__(
        self,
        predictor: SpeechProbabilityStream,
        profile: VadProfile,
    ) -> None:
        self.predictor = predictor
        self.profile = profile
        self.state = _VadState.IDLE
        self.pre_roll: deque[bytes] = deque(maxlen=profile.pre_roll_frames)
        self.possible_end: list[bytes] = []
        self.start_positive_frames = 0
        self.utterance_frames = 0
        self.failed = False

    @property
    def speech_active(self) -> bool:
        """Return whether speech has already been confirmed."""
        return self.state in {_VadState.SPEECH, _VadState.POSSIBLE_END}

    def reset(self) -> None:
        """Reset all stream and recurrent model state."""
        self.state = _VadState.IDLE
        self.pre_roll.clear()
        self.possible_end.clear()
        self.start_positive_frames = 0
        self.utterance_frames = 0
        self.predictor.reset()

    def _fail_open(self, frame: bytes) -> list[bytes]:
        """Release every buffered in-flight byte when inference fails."""
        self.failed = True
        if self.state in {_VadState.IDLE, _VadState.POSSIBLE_SPEECH}:
            # process_frame appends the current frame to pre_roll before
            # inference, so the buffered copy is already complete.
            output = list(self.pre_roll)
        elif self.state == _VadState.POSSIBLE_END:
            output = [*self.possible_end, frame]
        else:
            output = [frame]
        self.pre_roll.clear()
        self.possible_end.clear()
        logger.exception("VAD inference failed; switching this stream to pass-through")
        return output

    def _finish_utterance(self) -> list[bytes]:
        output = self.possible_end[: self.profile.post_roll_frames]
        output.append(VAD_UTTERANCE_BOUNDARY)
        self.reset()
        return output

    def process_frame(self, frame: bytes) -> list[bytes]:
        """Process exactly one model frame and return zero or more stream items."""
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"VAD frames must be exactly {FRAME_BYTES} bytes")
        if self.failed:
            return [frame]

        if self.state in {_VadState.IDLE, _VadState.POSSIBLE_SPEECH}:
            self.pre_roll.append(frame)

        try:
            probability = float(self.predictor.predict(frame))
        except Exception:
            return self._fail_open(frame)

        probability = min(1.0, max(0.0, probability))

        if self.state in {_VadState.IDLE, _VadState.POSSIBLE_SPEECH}:
            if probability >= self.profile.positive_threshold:
                self.state = _VadState.POSSIBLE_SPEECH
                self.start_positive_frames += 1
                if self.start_positive_frames >= self.profile.start_confirm_frames:
                    output = list(self.pre_roll)
                    self.utterance_frames = len(output)
                    self.pre_roll.clear()
                    self.start_positive_frames = 0
                    self.state = _VadState.SPEECH
                    if (
                        self.profile.max_utterance_frames
                        and self.utterance_frames >= self.profile.max_utterance_frames
                    ):
                        output.append(VAD_UTTERANCE_BOUNDARY)
                        self.reset()
                    return output
            elif probability <= self.profile.negative_threshold:
                self.state = _VadState.IDLE
                self.start_positive_frames = 0
            return []

        if self.state == _VadState.SPEECH:
            if probability <= self.profile.negative_threshold:
                self.state = _VadState.POSSIBLE_END
                self.possible_end = [frame]
                if len(self.possible_end) >= self.profile.end_hangover_frames:
                    return self._finish_utterance()
                return []
            self.utterance_frames += 1
            output = [frame]
            if (
                self.profile.max_utterance_frames
                and self.utterance_frames >= self.profile.max_utterance_frames
            ):
                output.append(VAD_UTTERANCE_BOUNDARY)
                self.reset()
            return output

        # POSSIBLE_END: keep the whole uncertain interval buffered. If speech
        # resumes, replay it all; if the pause is confirmed, retain only the
        # configured post-roll and suppress the remaining confirmed silence.
        if probability >= self.profile.positive_threshold:
            output = [*self.possible_end, frame]
            self.utterance_frames += len(output)
            self.possible_end.clear()
            self.state = _VadState.SPEECH
            if (
                self.profile.max_utterance_frames
                and self.utterance_frames >= self.profile.max_utterance_frames
            ):
                output.append(VAD_UTTERANCE_BOUNDARY)
                self.reset()
            return output

        self.possible_end.append(frame)
        if len(self.possible_end) >= self.profile.end_hangover_frames:
            return self._finish_utterance()
        return []

    def finish(self, partial_frame: bytes = b"") -> list[bytes]:
        """Flush safe in-flight speech bytes when the source stream ends."""
        if self.failed:
            return [partial_frame] if partial_frame else []

        if self.state == _VadState.SPEECH:
            output = [partial_frame] if partial_frame else []
            output.append(VAD_UTTERANCE_BOUNDARY)
            self.reset()
            return output

        if self.state == _VadState.POSSIBLE_END:
            output = self.possible_end[: self.profile.post_roll_frames]
            if partial_frame and len(output) < self.profile.post_roll_frames:
                output.append(partial_frame)
            output.append(VAD_UTTERANCE_BOUNDARY)
            self.reset()
            return output

        # A stream can end inside the 96 ms start-confirmation window. Preserve
        # that candidate so a short real word is not silently lost.
        if self.state == _VadState.POSSIBLE_SPEECH:
            output = list(self.pre_roll)
            if partial_frame:
                output.append(partial_frame)
            output.append(VAD_UTTERANCE_BOUNDARY)
            self.reset()
            return output

        self.reset()
        return []


class SileroOnnxStream:
    """Per-connection recurrent state for the Silero VAD ONNX model."""

    def __init__(self, session: object) -> None:
        import numpy as np

        self._np = np
        self.session = session
        self.reset()

    def reset(self) -> None:
        self.state = self._np.zeros((2, 1, 128), dtype=self._np.float32)
        self.context = self._np.zeros((1, 64), dtype=self._np.float32)

    def predict(self, pcm_frame: bytes) -> float:
        audio = self._np.frombuffer(pcm_frame, dtype="<i2").astype(self._np.float32)
        audio = (audio / 32768.0).reshape(1, -1)
        model_input = self._np.concatenate((self.context, audio), axis=1)
        probability, self.state = self.session.run(
            None,
            {
                "input": model_input,
                "state": self.state,
                "sr": self._np.array(SAMPLE_RATE, dtype=self._np.int64),
            },
        )
        self.context = model_input[:, -64:]
        return float(probability.reshape(-1)[0])


class SileroOnnxRuntime:
    """Shared thread-limited ONNX session with isolated per-stream state."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def create_stream(self) -> SileroOnnxStream:
        """Create independent recurrent state for a client connection."""
        return SileroOnnxStream(self.session)


class AudioVadService:
    """Configure and apply shared VAD to CE audio streams."""

    PROFILE_CONVERSATION = "conversation"
    PROFILE_LIVE_MICROPHONE = "live_microphone"
    PROFILE_SHARE = "share"

    def __init__(
        self,
        config: Optional[Config] = None,
        predictor_factory: Optional[Callable[[], SpeechProbabilityStream]] = None,
    ) -> None:
        vad_config = (config or get_config()).vad
        self.config = vad_config
        self._profiles = self._build_profiles(vad_config)
        self._predictor_factory = predictor_factory
        self._runtime: Optional[SileroOnnxRuntime] = None
        self._load_attempted = predictor_factory is not None

    @staticmethod
    def _build_profiles(config: dict) -> dict[str, VadProfile]:
        profiles = config.get("profiles", {})

        def profile(name: str, default_hangover_ms: int) -> VadProfile:
            values = profiles.get(name, {})
            positive = parse_float_config(
                values.get("positive_threshold", config.get("positive_threshold", 0.5)),
                0.5,
            )
            negative = parse_float_config(
                values.get(
                    "negative_threshold", config.get("negative_threshold", 0.35)
                ),
                0.35,
            )
            if not 0.0 <= negative < positive <= 1.0:
                logger.warning("Invalid VAD thresholds for %s; using defaults", name)
                positive, negative = 0.5, 0.35
            return VadProfile(
                mode=VadMode.parse(values.get("mode", config.get("mode", "off"))),
                positive_threshold=positive,
                negative_threshold=negative,
                pre_roll_ms=parse_int_config(
                    values.get("pre_roll_ms", config.get("pre_roll_ms")), 640
                ),
                start_confirm_ms=parse_int_config(
                    values.get("start_confirm_ms", config.get("start_confirm_ms")),
                    96,
                ),
                post_roll_ms=parse_int_config(
                    values.get("post_roll_ms", config.get("post_roll_ms")), 320
                ),
                end_hangover_ms=parse_int_config(
                    values.get("end_hangover_ms"), default_hangover_ms
                ),
                max_utterance_seconds=parse_float_config(
                    values.get(
                        "max_utterance_seconds",
                        config.get("max_utterance_seconds"),
                    ),
                    30.0,
                ),
            )

        return {
            AudioVadService.PROFILE_CONVERSATION: profile("conversation", 960),
            AudioVadService.PROFILE_LIVE_MICROPHONE: profile("live_microphone", 1600),
            AudioVadService.PROFILE_SHARE: profile("share", 1920),
        }

    def _profile_name(self, mode: str, input_type: str) -> str:
        """Map conversation first, then shared audio, else live microphone."""
        if mode == "conversation":
            return self.PROFILE_CONVERSATION
        if input_type == "share":
            return self.PROFILE_SHARE
        return self.PROFILE_LIVE_MICROPHONE

    def profile_for(self, mode: str, input_type: str) -> VadProfile:
        """Resolve settings for a conversation, microphone, or share stream."""
        return self._profiles[self._profile_name(mode, input_type)]

    def _ensure_predictor_factory(self) -> bool:
        if self._predictor_factory is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        model_path = Path(
            str(
                self.config.get(
                    "model_path",
                    Path(__file__).resolve().parents[1] / "assets" / "silero_vad.onnx",
                )
            )
        )
        try:
            self._runtime = SileroOnnxRuntime(model_path)
        except Exception:
            logger.exception(
                "Unable to initialize shared VAD; configured streams will fail open"
            )
            return False
        self._predictor_factory = self._runtime.create_stream
        logger.info("Shared VAD model initialized")
        return True

    def effective_mode(self, mode: str, input_type: str) -> VadMode:
        """Return the active rollout mode after checking model availability."""
        configured_mode = self.profile_for(mode, input_type).mode
        if configured_mode == VadMode.OFF:
            return configured_mode
        if not self._ensure_predictor_factory():
            return VadMode.OFF
        return configured_mode

    async def process_stream(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        *,
        mode: str,
        input_type: str,
    ) -> AsyncGenerator[bytes, None]:
        """Apply the configured VAD rollout behavior to one PCM stream."""
        profile = self.profile_for(mode, input_type)
        effective_mode = self.effective_mode(mode, input_type)
        if effective_mode == VadMode.OFF or self._predictor_factory is None:
            async for chunk in audio_generator:
                if chunk != VAD_RESET:
                    yield chunk
            return

        segmenter = VadStreamSegmenter(self._predictor_factory(), profile)
        pending = bytearray()

        async for chunk in audio_generator:
            if chunk == VAD_RESET:
                if effective_mode in {VadMode.BOUNDARY, VadMode.ACTIVE}:
                    # BOUNDARY is byte-transparent: every source chunk, including
                    # pending partial-frame bytes, was already yielded. Re-emit
                    # only the control marker there; ACTIVE still needs the data.
                    for item in segmenter.finish(bytes(pending)):
                        if effective_mode == VadMode.ACTIVE:
                            yield item
                        elif item == VAD_UTTERANCE_BOUNDARY:
                            yield item
                else:
                    segmenter.reset()
                pending.clear()
                continue
            if chunk == END_SIGNAL:
                if effective_mode == VadMode.ACTIVE:
                    for item in segmenter.finish(bytes(pending)):
                        yield item
                pending.clear()
                yield chunk
                return
            if chunk == CLIENT_UTTERANCE_BOUNDARY:
                if effective_mode in {VadMode.OFF, VadMode.SHADOW}:
                    yield chunk
                continue

            if effective_mode in {VadMode.SHADOW, VadMode.BOUNDARY}:
                yield chunk

            pending.extend(chunk)
            while len(pending) >= FRAME_BYTES:
                frame = bytes(pending[:FRAME_BYTES])
                del pending[:FRAME_BYTES]
                output = segmenter.process_frame(frame)
                for item in output:
                    if effective_mode == VadMode.ACTIVE:
                        yield item
                    elif (
                        effective_mode == VadMode.BOUNDARY
                        and item == VAD_UTTERANCE_BOUNDARY
                    ):
                        yield item
                if segmenter.failed:
                    if effective_mode == VadMode.ACTIVE and pending:
                        yield bytes(pending)
                    pending.clear()
                    break

        if effective_mode == VadMode.ACTIVE:
            for item in segmenter.finish(bytes(pending)):
                yield item


__all__ = [
    "AudioVadService",
    "CLIENT_UTTERANCE_BOUNDARY",
    "END_SIGNAL",
    "FRAME_BYTES",
    "VAD_RESET",
    "VAD_UTTERANCE_BOUNDARY",
    "VadMode",
    "VadProfile",
    "VadStreamSegmenter",
]
