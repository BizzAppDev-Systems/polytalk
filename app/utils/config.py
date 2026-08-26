# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuration parsing helpers."""

import logging
from functools import lru_cache

from ..config import get_config

logger = logging.getLogger(__name__)


def parse_bool_config(value: object, default: bool) -> bool:
    """Parse common boolean config values with a safe default."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


CUSTOM_INSTRUCTION_MAX_CHARS_DEFAULT = 250
_MISSING = object()


@lru_cache(maxsize=None)
def _cached_custom_instruction_max_chars(raw_value: object) -> int:
    return parse_int_config(
        raw_value,
        CUSTOM_INSTRUCTION_MAX_CHARS_DEFAULT,
        name="custom_instruction_max_chars",
        warn_on_invalid=True,
    )


def get_custom_instruction_max_chars(raw_value: object = _MISSING) -> int:
    """Return the cached custom instruction character limit."""
    if raw_value is _MISSING:
        raw_value = get_config().translation.get("custom_instruction_max_chars")

    return _cached_custom_instruction_max_chars(raw_value)


def clear_custom_instruction_max_chars_cache() -> None:
    """Clear cached custom instruction limit for tests and config reloads."""
    _cached_custom_instruction_max_chars.cache_clear()


def parse_float_config(value: object, default: float) -> float:
    """Parse a floating-point config value with a safe default."""
    if isinstance(value, bool):
        return default
    normalized = str(value or "").strip()
    if not normalized:
        return default
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return default


def parse_int_config(
    value: object,
    default: int,
    *,
    name: str | None = None,
    warn_on_invalid: bool = False,
) -> int:
    """Parse integer config values with a safe default."""
    if isinstance(value, bool):
        if warn_on_invalid:
            logger.warning(
                "Invalid integer config%s=%r; using default %s",
                f" {name}" if name else "",
                value,
                default,
            )
        return default

    normalized = str(value or "").strip()
    if not normalized:
        return default

    try:
        return int(normalized)
    except (TypeError, ValueError):
        if warn_on_invalid and not (
            normalized.startswith("${") and normalized.endswith("}")
        ):
            logger.warning(
                "Invalid integer config%s=%r; using default %s",
                f" {name}" if name else "",
                value,
                default,
            )
        return default
