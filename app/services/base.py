# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Base service class and interfaces for PolyTalk services.

Defines abstract base classes for transcription, translation, and TTS services.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """
    Result of audio transcription.

    Attributes:
        text: Transcribed text
        language: Detected language code
        duration: Audio duration in seconds (optional)
        success: Whether transcription succeeded
        error: Error message if failed (optional)
        is_partial: Whether this is a partial result (for streaming)
    """

    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    is_partial: bool = False
    metrics: Optional[dict] = None


@dataclass
class TranslationResult:
    """
    Result of text translation.

    Attributes:
        text: Translated text
        source_language: Source language code
        target_language: Target language code
        success: Whether translation succeeded
        error: Error message if failed (optional)
        translation_quality: LLM-assessed confidence in the translation quality (0.0-1.0)
    """

    text: str
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    translation_quality: Optional[float] = None


@dataclass
class TTSResult:
    """
    Result of text-to-speech conversion.

    Attributes:
        audio_path: Path to generated audio file
        audio_url: URL to access audio file (optional)
        duration: Audio duration in seconds (optional)
        success: Whether TTS succeeded
        error: Error message if failed (optional)
    """

    audio_path: Optional[Path] = None
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


class BaseTranscriptionService(ABC):
    """Abstract base class for transcription services."""

    @abstractmethod
    def stream_transcribe(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
        emit_policy: str = "live",
        candidate_languages: Optional[list[str]] = None,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Stream transcription from raw PCM audio chunks.

        Args:
            audio_generator: Async generator yielding 16 kHz mono int16 PCM chunks
            language: Optional source language code hint
            emit_policy: STT emission policy, either live or pause
            candidate_languages: Optional language candidates for detection
            on_result: Optional callback for each transcription result

        Yields:
            Incremental transcription results.
        """
        raise NotImplementedError


class BaseTranslationService(ABC):
    """
    Abstract base class for translation services.

    Implementations should provide text translation between languages.
    """

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: Optional[list[dict[str, str]]] = None,
        visual_context: Optional[str] = None,
        custom_instruction: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text from source to target language.

        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            context: Optional prior source/target translations to use as
                read-only context
            visual_context: Optional shared tab/page visual summary to use as
                a read-only hint
            custom_instruction: Optional user-provided translation guidance

        Returns:
            TranslationResult with translated text
        """
        pass


class BaseTTSService(ABC):
    """
    Abstract base class for text-to-speech services.

    Implementations should convert text to audio.
    """

    @abstractmethod
    async def synthesize(
        self, text: str, language: str, output_path: Optional[Path] = None
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech
            language: Language code for speech
            output_path: Optional path to save audio file

        Returns:
            TTSResult with audio file path
        """
        pass
