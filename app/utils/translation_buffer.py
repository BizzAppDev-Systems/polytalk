# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validation helpers for per-session translation buffering."""

import math

TRANSLATION_BUFFER_SECONDS_MIN = 1.0
TRANSLATION_BUFFER_SECONDS_MAX = 10.0
TRANSLATION_BUFFER_SECONDS_STEP = 0.5


def clamp_translation_buffer_seconds(value: object) -> float | None:
    """Return a finite session buffer value clamped to the supported UI range."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return min(
        TRANSLATION_BUFFER_SECONDS_MAX,
        max(TRANSLATION_BUFFER_SECONDS_MIN, seconds),
    )
