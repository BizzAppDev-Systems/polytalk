# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stable names and capabilities for streaming result modes."""

PIPELINE_MODE_TRANSCRIPT = "transcript"
PIPELINE_MODE_TRANSLATION = "translation"
PIPELINE_MODE_TRANSLATION_TTS = "translation_tts"

PIPELINE_MODE_OPTIONS = frozenset(
    {
        PIPELINE_MODE_TRANSCRIPT,
        PIPELINE_MODE_TRANSLATION,
        PIPELINE_MODE_TRANSLATION_TTS,
    }
)
PIPELINE_MODE_DEFAULT = PIPELINE_MODE_TRANSLATION_TTS


def normalize_pipeline_mode(value: object) -> str:
    """Return a supported mode name, falling back to the full pipeline."""
    if isinstance(value, str) and value in PIPELINE_MODE_OPTIONS:
        return value
    return PIPELINE_MODE_DEFAULT


def pipeline_mode_capabilities(mode: str) -> tuple[bool, bool]:
    """Return translation and TTS enablement for a normalized mode."""
    return mode != PIPELINE_MODE_TRANSCRIPT, mode == PIPELINE_MODE_TRANSLATION_TTS
