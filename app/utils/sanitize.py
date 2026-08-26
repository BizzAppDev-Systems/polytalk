# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sanitization helpers for user-provided guidance."""

import re
import unicodedata


# Pre-computed at module load: iterates BMP codepoints (~65k) to build digit mapping.
# Covers all known digit scripts (Devanagari, Gujarati, Arabic-Indic, Thai, etc.).
# Only covers the Basic Multilingual Plane (BMP). If future Unicode scripts with
# digits in supplementary planes (> 0xFFFF) are needed, the range should be extended.
_DIGIT_TRANSLATE_TABLE = str.maketrans(
    {
        c: str(int(c))
        for c in "".join(chr(i) for i in range(0x0000, 0x10000))
        if unicodedata.category(c) == "Nd" and not c.isascii()
    }
)


def normalize_instruction(value: object, max_chars: int | None = None) -> str:
    """Normalize custom instruction text and optionally bound its length."""
    without_control_chars = re.sub(r"[\x00-\x1f\x7f]", " ", str(value or ""))
    normalized = " ".join(without_control_chars.split())
    if max_chars is None or max_chars <= 0:
        return normalized
    return normalized[:max_chars]


def normalize_tts_text(text: str) -> str:
    """Convert non-Latin digits to ASCII digits (0-9) for TTS compatibility.

    Many TTS engines (including Supertonic) only support ASCII digits. This
    function converts all Unicode digits to their ASCII equivalents using
    Python's native Unicode digit handling.

    This is safe because:
    - Only digit glyphs are changed, never letters
    - The numeric meaning is fully preserved
    - The result does not alter the meaning of the source text
    - Works for all Unicode scripts automatically (no hardcoded mappings)

    Args:
        text: The text to normalize.

    Returns:
        Text with all non-Latin digits replaced by ASCII digits.
    """
    return text.translate(_DIGIT_TRANSLATE_TABLE)
