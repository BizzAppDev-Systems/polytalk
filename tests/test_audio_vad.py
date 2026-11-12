"""Regression tests for shared PCM voice activity detection."""

from __future__ import annotations

from collections import deque

import pytest

from app.services.audio_vad import (
    CLIENT_UTTERANCE_BOUNDARY,
    END_SIGNAL,
    FRAME_BYTES,
    VAD_UTTERANCE_BOUNDARY,
    AudioVadService,
    VadMode,
    VadProfile,
    VadStreamSegmenter,
)


class ScriptedPredictor:
    """Return deterministic probabilities and optionally fail on one call."""

    def __init__(self, probabilities: list[float], fail_at: int | None = None):
        self.probabilities = deque(probabilities)
        self.fail_at = fail_at
        self.calls = 0
        self.resets = 0

    def predict(self, pcm_frame: bytes) -> float:
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("inference failed")
        return self.probabilities.popleft() if self.probabilities else 0.0

    def reset(self) -> None:
        self.resets += 1


class VadConfig:
    def __init__(self, vad: dict):
        self.vad = vad


def frame(index: int) -> bytes:
    """Make one distinguishable, model-sized PCM frame."""
    return bytes([index % 251 + 1]) * FRAME_BYTES


def test_active_vad_preserves_preroll_speech_and_configured_postroll():
    predictor = ScriptedPredictor([0.0, 0.0, 0.8, 0.8, 0.8, 0.9, 0.1, 0.1, 0.1])
    profile = VadProfile(
        mode=VadMode.ACTIVE,
        pre_roll_ms=128,
        start_confirm_ms=96,
        post_roll_ms=64,
        end_hangover_ms=96,
        max_utterance_seconds=0,
    )
    segmenter = VadStreamSegmenter(predictor, profile)
    output: list[bytes] = []

    for index in range(9):
        output.extend(segmenter.process_frame(frame(index)))

    assert output == [
        frame(1),
        frame(2),
        frame(3),
        frame(4),
        frame(5),
        frame(6),
        frame(7),
        VAD_UTTERANCE_BOUNDARY,
    ]


def test_false_start_is_not_forwarded_as_speech():
    predictor = ScriptedPredictor([0.8, 0.1, 0.0, 0.0])
    profile = VadProfile(
        mode=VadMode.ACTIVE,
        pre_roll_ms=128,
        start_confirm_ms=96,
        end_hangover_ms=96,
    )
    segmenter = VadStreamSegmenter(predictor, profile)

    output = []
    for index in range(4):
        output.extend(segmenter.process_frame(frame(index)))

    assert output == []


def test_possible_end_audio_is_replayed_when_speech_resumes():
    predictor = ScriptedPredictor([0.8, 0.8, 0.8, 0.1, 0.1, 0.9])
    profile = VadProfile(
        mode=VadMode.ACTIVE,
        pre_roll_ms=96,
        start_confirm_ms=96,
        post_roll_ms=32,
        end_hangover_ms=160,
        max_utterance_seconds=0,
    )
    segmenter = VadStreamSegmenter(predictor, profile)
    output: list[bytes] = []

    for index in range(6):
        output.extend(segmenter.process_frame(frame(index)))

    assert output == [frame(index) for index in range(6)]
    assert segmenter.speech_active is True


def test_stream_end_preserves_short_unconfirmed_word_candidate():
    predictor = ScriptedPredictor([0.0, 0.8, 0.8])
    profile = VadProfile(
        mode=VadMode.ACTIVE,
        pre_roll_ms=96,
        start_confirm_ms=96,
    )
    segmenter = VadStreamSegmenter(predictor, profile)

    output: list[bytes] = []
    for index in range(3):
        output.extend(segmenter.process_frame(frame(index)))
    output.extend(segmenter.finish(b"partial"))

    assert output == [
        frame(0),
        frame(1),
        frame(2),
        b"partial",
        VAD_UTTERANCE_BOUNDARY,
    ]


@pytest.mark.asyncio
async def test_active_stream_handles_arbitrary_transport_chunk_sizes():
    probabilities = [0.8, 0.8, 0.8, 0.9, 0.1, 0.1]
    predictor = ScriptedPredictor(probabilities)
    service = AudioVadService(
        VadConfig(
            {
                "profiles": {
                    "live_microphone": {
                        "mode": "active",
                        "pre_roll_ms": 96,
                        "start_confirm_ms": 96,
                        "post_roll_ms": 32,
                        "end_hangover_ms": 64,
                        "max_utterance_seconds": 0,
                    }
                }
            }
        ),
        predictor_factory=lambda: predictor,
    )
    raw = b"".join(frame(index) for index in range(6))

    async def source():
        yield raw[:333]
        yield raw[333:1701]
        yield raw[1701:]
        yield END_SIGNAL

    output = [
        item
        async for item in service.process_stream(
            source(), mode="live", input_type="microphone"
        )
    ]

    assert b"".join(
        item for item in output if item not in {END_SIGNAL, VAD_UTTERANCE_BOUNDARY}
    ) == b"".join([frame(0), frame(1), frame(2), frame(3), frame(4)])
    assert output[-2:] == [VAD_UTTERANCE_BOUNDARY, END_SIGNAL]


@pytest.mark.asyncio
async def test_active_stream_fails_open_without_losing_inflight_bytes():
    predictor = ScriptedPredictor([0.0, 0.8], fail_at=3)
    service = AudioVadService(
        VadConfig(
            {
                "profiles": {
                    "live_microphone": {
                        "mode": "active",
                        "pre_roll_ms": 128,
                        "start_confirm_ms": 96,
                    }
                }
            }
        ),
        predictor_factory=lambda: predictor,
    )
    raw = b"".join(frame(index) for index in range(5)) + b"tail"

    async def source():
        yield raw[: FRAME_BYTES * 3 + 17]
        yield raw[FRAME_BYTES * 3 + 17 :]
        yield END_SIGNAL

    output = [
        item
        async for item in service.process_stream(
            source(), mode="live", input_type="microphone"
        )
    ]

    assert b"".join(item for item in output if item != END_SIGNAL) == raw
    assert output[-1] == END_SIGNAL


@pytest.mark.asyncio
async def test_boundary_mode_passes_audio_and_adds_only_server_boundary():
    predictor = ScriptedPredictor([0.8, 0.8, 0.8, 0.1, 0.1])
    service = AudioVadService(
        VadConfig(
            {
                "profiles": {
                    "share": {
                        "mode": "boundary",
                        "pre_roll_ms": 96,
                        "start_confirm_ms": 96,
                        "post_roll_ms": 32,
                        "end_hangover_ms": 64,
                    }
                }
            }
        ),
        predictor_factory=lambda: predictor,
    )
    chunks = [frame(index) for index in range(5)]

    async def source():
        for chunk in chunks[:3]:
            yield chunk
        yield CLIENT_UTTERANCE_BOUNDARY
        for chunk in chunks[3:]:
            yield chunk
        yield END_SIGNAL

    output = [
        item
        async for item in service.process_stream(
            source(), mode="live", input_type="share"
        )
    ]

    assert [item for item in output if item in chunks] == chunks
    assert CLIENT_UTTERANCE_BOUNDARY not in output
    assert output.count(VAD_UTTERANCE_BOUNDARY) == 1
    assert output[-1] == END_SIGNAL


@pytest.mark.asyncio
async def test_shadow_mode_is_byte_and_control_transparent():
    predictor = ScriptedPredictor([0.8, 0.8, 0.8, 0.1, 0.1])
    service = AudioVadService(
        VadConfig({"mode": "shadow"}),
        predictor_factory=lambda: predictor,
    )
    expected = [frame(0), CLIENT_UTTERANCE_BOUNDARY, frame(1), END_SIGNAL]

    async def source():
        for item in expected:
            yield item

    output = [
        item
        async for item in service.process_stream(
            source(), mode="conversation", input_type="microphone"
        )
    ]

    assert output == expected
