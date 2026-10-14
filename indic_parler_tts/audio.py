"""Audio-shape validation for generated Indic Parler waveforms."""

import math
from typing import Any


def normalize_generated_audio(audio: Any) -> Any:
    """Return one-dimensional mono audio or reject unusable model output."""
    normalized = audio.squeeze()
    if normalized.ndim == 0 or normalized.size < 2:
        raise ValueError("model generated no usable audio samples")
    if normalized.ndim != 1:
        raise ValueError("model generated unsupported audio shape")
    minimum = float(normalized.min())
    maximum = float(normalized.max())
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("model generated non-finite audio samples")
    return normalized
