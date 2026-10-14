"""Harden the third-party Supertonic server before its CLI starts."""

import logging

import numpy as np
from supertonic import TTS

from text_filter import remove_unsupported_characters


logger = logging.getLogger(__name__)
_original_synthesize = TTS.synthesize
# Compatibility contract for the Dockerfile-pinned Supertonic 1.3.1 release:
# validate_text() returns (is_valid, unsupported_characters), and synthesize()
# returns audio shaped (channels, samples) plus a one-dimensional duration array;
# TTS.sample_rate is the public rate attribute. Revalidate these contracts before
# changing the pin. The model/default rate fallbacks keep this error path safe.


def _synthesize_with_supported_text(self, text, *args, **kwargs):
    """Ignore unsupported input characters instead of failing the request."""
    filtered_text, removed_count = remove_unsupported_characters(
        text, self.model.text_processor.validate_text
    )
    if removed_count:
        logger.warning(
            "Removed %d unsupported character(s) from Supertonic input",
            removed_count,
        )
    if not filtered_text:
        duration = 0.1
        sample_rate = getattr(self, "sample_rate", None) or getattr(
            self.model, "sample_rate", 44100
        )
        samples = max(1, int(sample_rate * duration))
        return (
            np.zeros((1, samples), dtype=np.float32),
            np.array([duration], dtype=np.float32),
        )
    return _original_synthesize(self, filtered_text, *args, **kwargs)


TTS.synthesize = _synthesize_with_supported_text


del TTS
