"""Regression tests for TTS provider boundary failures."""

import pytest

from indic_parler_tts.audio import normalize_generated_audio
from supertonic_tts.text_filter import remove_unsupported_characters


class FakeAudio:
    """Minimal NumPy-compatible array used by service-boundary tests."""

    def __init__(self, ndim: int, size: int, samples: list[float]):
        self.ndim = ndim
        self.size = size
        self.samples = samples

    def squeeze(self):
        return self

    def min(self):
        if any(sample != sample for sample in self.samples):
            return float("nan")
        return min(self.samples)

    def max(self):
        if any(sample != sample for sample in self.samples):
            return float("nan")
        return max(self.samples)


def test_supertonic_filter_removes_only_rejected_characters():
    """Mixed Gujarati glyphs are ignored without changing supported text."""
    unsupported = set("ટદષિૃ્")

    def validate(text: str) -> tuple[bool, list[str]]:
        rejected = sorted(set(text) & unsupported)
        return not rejected, rejected

    filtered, removed_count = remove_unsupported_characters(
        "Hello ટદષિૃ્ world", validate
    )

    assert filtered == "Hello world"
    assert removed_count == 6


def test_supertonic_filter_preserves_valid_text():
    """Valid text is passed through byte-for-byte."""
    text = "Привет, мир!"

    filtered, removed_count = remove_unsupported_characters(
        text, lambda _text: (True, [])
    )

    assert filtered == text
    assert removed_count == 0


@pytest.mark.parametrize(
    "audio",
    [
        FakeAudio(ndim=0, size=1, samples=[0.0]),
        FakeAudio(ndim=1, size=0, samples=[]),
        FakeAudio(ndim=1, size=1, samples=[0.0]),
    ],
)
def test_indic_parler_rejects_unusable_audio(audio):
    """Scalar, empty, and one-sample generations do not reach soundfile.write."""
    with pytest.raises(ValueError, match="no usable audio samples"):
        normalize_generated_audio(audio)


def test_indic_parler_accepts_mono_audio():
    """Valid mono model output remains unchanged."""
    generated = FakeAudio(ndim=1, size=3, samples=[0.1, -0.2, 0.3])

    audio = normalize_generated_audio(generated)

    assert audio is generated
    assert audio.ndim == 1


def test_indic_parler_rejects_multidimensional_audio():
    """Unexpected channels are rejected instead of being interleaved."""
    generated = FakeAudio(ndim=2, size=4, samples=[0.1, -0.2, 0.3, -0.4])

    with pytest.raises(ValueError, match="unsupported audio shape"):
        normalize_generated_audio(generated)


def test_indic_parler_rejects_non_finite_audio():
    """NaN model output is rejected before WAV serialization."""
    generated = FakeAudio(ndim=1, size=2, samples=[0.1, float("nan")])

    with pytest.raises(ValueError, match="non-finite"):
        normalize_generated_audio(generated)
