"""Text filtering helpers for the Supertonic service."""

from collections.abc import Callable


TextValidator = Callable[[str], tuple[bool, list[str]]]


def remove_unsupported_characters(
    text: str, validate_text: TextValidator
) -> tuple[str, int]:
    """Remove characters rejected by Supertonic's loaded text processor."""
    is_valid, unsupported = validate_text(text)
    if is_valid:
        return text, 0

    unsupported_characters = set(unsupported)
    filtered = "".join(
        " " if character in unsupported_characters else character for character in text
    )
    return " ".join(filtered.split()), sum(
        character in unsupported_characters for character in text
    )
