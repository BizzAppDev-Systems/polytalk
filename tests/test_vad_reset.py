"""No-word-loss regression for manual pause/reset during active speech."""

from collections import deque

import pytest

from app.services.audio_vad import (
    END_SIGNAL,
    FRAME_BYTES,
    VAD_RESET,
    VAD_UTTERANCE_BOUNDARY,
    AudioVadService,
)


class Predictor:
    def __init__(self):
        self.values = deque([0.9, 0.9, 0.9])

    def predict(self, frame: bytes) -> float:
        return self.values.popleft() if self.values else 0.0

    def reset(self) -> None:
        pass


class Config:
    def __init__(self, mode: str):
        self.vad = {
            "profiles": {
                "conversation": {
                    "mode": mode,
                    "pre_roll_ms": 96,
                    "start_confirm_ms": 96,
                }
            }
        }


@pytest.mark.parametrize("vad_mode", ["active", "boundary"])
@pytest.mark.asyncio
async def test_manual_pause_preserves_confirmed_speech_and_partial_frame(
    vad_mode: str,
) -> None:
    service = AudioVadService(Config(vad_mode), predictor_factory=Predictor)
    speech = b"".join(bytes([index + 1]) * FRAME_BYTES for index in range(3))
    partial = b"quiet-final-syllable"

    async def source():
        yield speech + partial
        yield VAD_RESET
        yield END_SIGNAL

    output = [
        item
        async for item in service.process_stream(
            source(), mode="conversation", input_type="microphone"
        )
    ]

    assert (
        b"".join(
            item for item in output if item not in {VAD_UTTERANCE_BOUNDARY, END_SIGNAL}
        )
        == speech + partial
    )
    assert output[-2:] == [VAD_UTTERANCE_BOUNDARY, END_SIGNAL]
