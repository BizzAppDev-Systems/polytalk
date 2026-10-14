# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional

from ..version import __version__


class HealthResponse(BaseModel):
    """
    Health check response schema.

    Attributes:
        status: Service status
        version: Application version
    """

    status: str
    version: str = __version__


class ErrorResponse(BaseModel):
    """
    Error response schema.

    Attributes:
        success: Always False for errors
        error: Error message
        details: Optional detailed error information
    """

    success: bool = False
    error: str
    details: Optional[str] = None


class TextTranslationRequest(BaseModel):
    """Validated request for translating user-provided Unicode text."""

    text: str = Field(min_length=1, max_length=5000)
    source_language: str = Field(pattern=r"^[a-z]{2,3}(?:_[A-Z]{2})?$")
    target_language: str = Field(pattern=r"^[a-z]{2,3}(?:_[A-Z]{2})?$")
    custom_instruction: Optional[str] = Field(default=None, max_length=1000)


class TextTranslationResponse(BaseModel):
    """Successful text translation response."""

    translated_text: str
    source_language: str
    target_language: str


class TextSpeechRequest(BaseModel):
    """Validated request for generating speech from translated Unicode text."""

    text: str = Field(min_length=1, max_length=5000)
    language: str = Field(pattern=r"^[a-z]{2,3}(?:_[A-Z]{2})?$")
    voice: Optional[str] = Field(default=None, min_length=1, max_length=120)
