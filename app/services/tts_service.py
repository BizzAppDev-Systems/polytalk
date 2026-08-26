# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Text-to-Speech service supporting multiple providers (OpenAI, Piper).

Supports both real API and mock mode for testing.

Voice Selection Fallback Chain:
1. Use configured default voice from config.yaml for the language
2. Match exact language code from Piper TTS API voices (e.g., 'en_GB-jenny_dioco-medium')
3. Match base language code from API voices (e.g., 'en' from 'en_GB' or 'en_US')
4. Fall back to default voice from config
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from .base import BaseTTSService, TTSResult
from ..config import get_config
from ..utils.logger import get_logger
from ..utils.sanitize import normalize_tts_text

logger = get_logger(__name__)


class TTSService(BaseTTSService):
    """
    Multi-provider TTS service.

    Converts text to speech using OpenAI TTS API, Piper TTS, or mock mode.
    Falls back to mock mode when configured or when services are unavailable.
    """

    def __init__(self) -> None:
        """Initialize TTS service with configuration."""
        self.config = get_config().tts
        self.app_config = get_config().app
        self.enabled = self.config.get("enabled", True)
        self.mock_mode = self.config.get("mock_mode", True)
        self.provider = self.config.get("provider", "piper")
        self.fallback_provider = self.config.get("fallback_provider", self.provider)
        self.base_url = self.config.get("base_url", "http://localhost:5000")
        self.providers = self.config.get("providers", {})
        self.language_providers = self.config.get("language_providers", {})
        self.voice = self.config.get("voice", "en_US-lessac-medium")
        self.timeout = self.config.get("timeout_seconds", 15)
        self.media_dir = get_config().media_output_dir
        self._voices_cache: Optional[dict] = None
        self._voices_cache_timestamp: Optional[float] = None
        self._voices_cache_ttl = 3600
        self._voices_lock = asyncio.Lock()
        self.default_voices = self.config.get("default_voices", {})
        self.length_scales = self.config.get("length_scales", {})

        # Singleton httpx.AsyncClient with connection pooling
        self._http_client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=50,
                keepalive_expiry=60.0,
            ),
        )

        logger.info(
            f"TTSService initialized: provider={self.provider}, enabled={self.enabled}, mock_mode={self.mock_mode}"
        )

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
        if not self.enabled:
            logger.warning("TTS service is disabled")
            return TTSResult(success=False, error="TTS service is disabled")

        if self.mock_mode:
            logger.info("Using mock TTS")
            return await self._mock_synthesize(text, language, output_path)

        provider = self._get_provider_for_language(language)
        try:
            result = await self._synthesize_with_provider(
                provider, text, language, output_path
            )
        except Exception as exc:
            logger.error(f"TTS synthesis failed: provider={provider} error={exc}")
            result = TTSResult(success=False, error=str(exc))

        if not result.success and provider != self.fallback_provider:
            logger.warning(
                "TTS provider failed; using fallback: "
                f"selected={provider} fallback={self.fallback_provider} "
                f"language={language} error={result.error}"
            )
            try:
                return await self._synthesize_with_provider(
                    self.fallback_provider, text, language, output_path
                )
            except Exception as exc:
                return TTSResult(
                    success=False,
                    error=(
                        f"{provider} failed ({result.error}); "
                        f"{self.fallback_provider} fallback failed ({exc})"
                    ),
                )
        return result

    async def _synthesize_with_provider(
        self,
        provider: str,
        text: str,
        language: str,
        output_path: Optional[Path],
    ) -> TTSResult:
        """Dispatch synthesis while keeping provider formats out of the pipeline."""
        if provider == "piper":
            return await self._piper_synthesize(text, language, output_path)
        if provider == "supertonic":
            return await self._supertonic_synthesize(text, language, output_path)
        if provider == "indic_parler":
            return await self._indic_parler_synthesize(text, language, output_path)
        return await self._openai_synthesize(text, language, output_path)

    async def _mock_synthesize(
        self, text: str, language: str, output_path: Optional[Path] = None
    ) -> TTSResult:
        """
        Generate mock TTS audio for testing.

        Creates a simple silence audio file to simulate TTS output.

        Args:
            text: Text to convert to speech
            language: Language code for speech
            output_path: Optional path to save audio file

        Returns:
            Mock TTSResult with generated audio path
        """
        try:
            import wave

            if output_path is None:
                unique_id = str(uuid.uuid4())[:8]
                output_path = self.media_dir / f"tts_{unique_id}.wav"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            sample_rate = 22050
            duration = min(max(len(text) / 10, 1.0), 10.0)
            num_samples = int(sample_rate * duration)

            with wave.open(str(output_path), "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b"\x00\x00" * num_samples)

            audio_url = f"/media/output/{output_path.name}"

            logger.info(f"Mock TTS generated: {output_path}")

            return TTSResult(
                audio_path=output_path,
                audio_url=audio_url,
                duration=duration,
                success=True,
            )
        except Exception as e:
            logger.error(f"Mock TTS failed: {e}")
            return TTSResult(success=False, error=f"Mock TTS failed: {e}")

    def _normalize_language(self, language: str) -> tuple[str, str]:
        """Return normalized exact and base language codes."""
        normalized_language = language.replace("-", "_")
        lang_base = normalized_language.split("_")[0].lower()
        return normalized_language, lang_base

    def _get_provider_for_language(self, language: str) -> str:
        """Resolve the TTS provider for a language, falling back to default."""
        normalized_language, lang_base = self._normalize_language(language)
        return self.language_providers.get(
            normalized_language,
            self.language_providers.get(lang_base, self.provider),
        )

    def _get_provider_config(self, provider: str) -> dict:
        """Return provider-specific config merged with top-level defaults."""
        provider_config = self.providers.get(provider, {})
        return {**self.config, **provider_config}

    def _get_provider_base_url(self, provider: str) -> str:
        """Return the base URL for a provider."""
        provider_config = self._get_provider_config(provider)
        return provider_config.get("base_url", self.base_url).rstrip("/")

    @staticmethod
    def _config_int(value: object, default: int) -> int:
        """Parse an integer config value with a safe fallback."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _config_float(value: object, default: float) -> float:
        """Parse a float config value with a safe fallback."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def _fetch_voices(self) -> dict:
        """
        Fetch available voices from Piper TTS API with thread-safe caching.

        Uses asyncio.Lock to prevent concurrent requests and implements
        TTL-based cache expiration (1 hour).

        Returns:
            Dictionary of available voices with metadata
        """
        if not self.enabled:
            logger.debug("TTS service disabled; skipping voice fetch")
            return {}

        current_time = time.time()

        if (
            self._voices_cache is not None
            and self._voices_cache_timestamp is not None
            and (current_time - self._voices_cache_timestamp) < self._voices_cache_ttl
        ):
            return self._voices_cache

        async with self._voices_lock:
            if (
                self._voices_cache is not None
                and self._voices_cache_timestamp is not None
                and (current_time - self._voices_cache_timestamp)
                < self._voices_cache_ttl
            ):
                return self._voices_cache

            try:
                response = await self._http_client.get(
                    f"{self._get_provider_base_url('piper')}/voices"
                )
                response.raise_for_status()
                self._voices_cache = response.json()
                self._voices_cache_timestamp = current_time
                logger.info(f"Fetched {len(self._voices_cache)} voices from Piper TTS")
                return self._voices_cache
            except Exception as e:
                logger.warning(f"Failed to fetch voices from Piper TTS: {e}")
                return {}

    async def _get_voice_for_language(self, language: str) -> str:
        """
        Get the appropriate voice for a given language code from Piper TTS.

        Args:
            language: Language code (e.g., 'en', 'en_US', 'hi_IN', 'en-US')

        Returns:
            Voice name for the language, falling back to default voice

        Fallback chain:
        1. Available configured default voice for exact language
        2. Available configured default voice for base language
        3. Exact language match from API voices
        4. Base language match from API voices
        5. Default voice from config
        """
        normalized_language, lang_base = self._normalize_language(language)
        voices = await self._fetch_voices()

        def voice_available(voice_name: str) -> bool:
            return any(
                voice_key.replace(".onnx", "") == voice_name
                for voice_key in voices.keys()
            )

        if normalized_language in self.default_voices:
            default_voice = self.default_voices[normalized_language]
            if not voices or voice_available(default_voice):
                logger.info(f"Using default voice for '{language}': {default_voice}")
                return default_voice
            logger.warning(
                f"Configured default voice for '{language}' is unavailable: "
                f"{default_voice}. Falling back to base language or discovered voices."
            )

        if lang_base in self.default_voices:
            default_voice = self.default_voices[lang_base]
            if not voices or voice_available(default_voice):
                logger.info(f"Using default voice for '{language}': {default_voice}")
                return default_voice
            logger.warning(
                f"Configured default voice for '{lang_base}' is unavailable: "
                f"{default_voice}. Falling back to discovered voices."
            )

        if not voices:
            logger.warning(
                f"No voices available from TTS API. "
                f"Ensure default voice '{self.voice}' is configured correctly."
            )
            return self.voice

        for voice_key in voices.keys():
            clean_key = voice_key.replace(".onnx", "")
            if clean_key.startswith(normalized_language):
                logger.info(f"Matched voice for '{language}': {clean_key}")
                return clean_key

        for voice_key in voices.keys():
            clean_key = voice_key.replace(".onnx", "")
            if clean_key.startswith(f"{lang_base}_"):
                logger.info(
                    f"Matched voice for '{language}' (base: '{lang_base}'): {clean_key}"
                )
                return clean_key

        logger.warning(
            f"No voice found for language '{language}'. "
            f"Using default: {self.voice}. "
            f"Available languages: {set(v.split('_')[0] for v in voices.keys())}"
        )
        return self.voice

    def _get_length_scale_for_language(self, language: str, voice: str) -> float:
        """
        Get Piper speaking speed for a language/voice.

        Piper length scale is inverse speed: lower values speak faster. The
        lookup supports voice-specific, exact language, base language, then
        global default settings.
        """
        normalized_language, lang_base = self._normalize_language(language)

        candidates = [
            voice,
            normalized_language,
            lang_base,
            "default",
        ]

        for key in candidates:
            if key in self.length_scales:
                return float(self.length_scales[key])

        return float(self.config.get("length_scale", 0.9))

    async def _piper_synthesize(
        self, text: str, language: str, output_path: Optional[Path] = None
    ) -> TTSResult:
        """
        Synthesize speech using Piper TTS API.

        Args:
            text: Text to convert to speech
            language: Language code for speech
            output_path: Optional path to save audio file

        Returns:
            TTSResult with audio file path
        """
        url = self._get_provider_base_url("piper")

        # Select voice dynamically from Piper TTS API
        voice = await self._get_voice_for_language(language)

        payload = {
            "text": text,
            "voice": voice,
            "length_scale": self._get_length_scale_for_language(language, voice),
        }

        try:
            response = await self._http_client.post(url, json=payload)
            response.raise_for_status()

            audio_content = response.content

            if output_path is None:
                unique_id = str(uuid.uuid4())[:8]
                output_path = self.media_dir / f"tts_{unique_id}.wav"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(audio_content)

            audio_url = f"/media/output/{output_path.name}"

            logger.info(f"Piper TTS generated: {output_path} (voice: {voice})")

            return TTSResult(
                audio_path=output_path,
                audio_url=audio_url,
                success=True,
            )
        except httpx.HTTPError as e:
            logger.error(f"Piper TTS HTTP error: {e}")
            return TTSResult(
                success=False,
                error=f"Piper TTS HTTP error: {e}",
            )

    def _get_supertonic_voice_for_language(self, language: str) -> str:
        """Return Supertonic voice style for a language."""
        provider_config = self._get_provider_config("supertonic")
        voices = provider_config.get("voices", {})
        normalized_language, lang_base = self._normalize_language(language)
        return voices.get(
            normalized_language,
            voices.get(lang_base, provider_config.get("voice", "M1")),
        )

    async def _supertonic_synthesize(
        self, text: str, language: str, output_path: Optional[Path] = None
    ) -> TTSResult:
        """Synthesize speech using a Supertonic TTS HTTP server."""
        provider_config = self._get_provider_config("supertonic")
        normalized_language, lang_base = self._normalize_language(language)
        voice = self._get_supertonic_voice_for_language(language)
        url = self._get_provider_base_url("supertonic") + "/v1/tts"
        text = normalize_tts_text(text)

        payload = {
            "text": text,
            "voice": voice,
            "lang": lang_base,
            "steps": self._config_int(provider_config.get("steps"), 8),
            "speed": self._config_float(provider_config.get("speed"), 1.05),
            "response_format": provider_config.get("response_format", "wav"),
        }

        try:
            async with self._http_client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                if output_path is None:
                    unique_id = str(uuid.uuid4())[:8]
                    output_path = (
                        self.media_dir / f"tts_{normalized_language}_{unique_id}.wav"
                    )

                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            f.write(chunk)

                audio_url = f"/media/output/{output_path.name}"
                duration_header = response.headers.get("X-Audio-Duration")
                duration = float(duration_header) if duration_header else None

                logger.info(
                    f"Supertonic TTS generated: {output_path} "
                    f"(lang: {lang_base}, voice: {voice})"
                )

                return TTSResult(
                    audio_path=output_path,
                    audio_url=audio_url,
                    duration=duration,
                    success=True,
                )
        except httpx.HTTPError as e:
            logger.error(f"Supertonic TTS HTTP error: {e}")
            return TTSResult(
                success=False,
                error=f"Supertonic TTS HTTP error: {e}",
            )

    async def _indic_parler_synthesize(
        self, text: str, language: str, output_path: Optional[Path] = None
    ) -> TTSResult:
        """Synthesize WAV audio through the local Indic Parler service."""
        provider_config = self._get_provider_config("indic_parler")
        normalized_language, lang_base = self._normalize_language(language)
        genders = provider_config.get("genders", {})
        voices = provider_config.get("voices", {})
        gender = genders.get(
            normalized_language,
            genders.get(lang_base, provider_config.get("gender", "female")),
        )
        voice = voices.get(
            normalized_language,
            voices.get(lang_base, provider_config.get("voice", "auto")),
        )
        paces = provider_config.get("paces", {})
        pace = paces.get(
            normalized_language,
            paces.get(lang_base, provider_config.get("pace", "moderate")),
        )
        speeds = provider_config.get("speeds", {})
        speed = self._config_float(
            speeds.get(
                normalized_language,
                speeds.get(lang_base, provider_config.get("speed", 1.0)),
            ),
            1.0,
        )
        payload = {
            "text": normalize_tts_text(text),
            "lang": lang_base,
            "gender": gender,
            "voice": voice,
            "pace": pace,
            "speed": speed,
        }
        description = provider_config.get("description")
        if description:
            payload["description"] = description

        url = self._get_provider_base_url("indic_parler") + "/v1/tts"
        timeout = self._config_float(provider_config.get("timeout_seconds"), 180.0)
        try:
            response = await self._http_client.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            audio_content = response.content
            if not audio_content:
                return TTSResult(
                    success=False,
                    error="Indic Parler TTS returned an empty response",
                )
        except httpx.HTTPError as exc:
            logger.error("Indic Parler TTS HTTP error: %s", exc)
            return TTSResult(
                success=False,
                error=f"Indic Parler TTS HTTP error: {exc}",
            )

        if output_path is None:
            unique_id = str(uuid.uuid4())[:8]
            output_path = self.media_dir / f"tts_{normalized_language}_{unique_id}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as output_file:
            output_file.write(audio_content)

        logger.info(
            "Indic Parler TTS generated: %s (lang: %s, gender: %s, voice: %s)",
            output_path,
            lang_base,
            gender,
            voice,
        )
        return TTSResult(
            audio_path=output_path,
            audio_url=f"/media/output/{output_path.name}",
            success=True,
        )

    async def _openai_synthesize(
        self, text: str, language: str, output_path: Optional[Path] = None
    ) -> TTSResult:
        """
        Synthesize speech using OpenAI TTS API.

        Args:
            text: Text to convert to speech
            language: Language code for speech
            output_path: Optional path to save audio file

        Returns:
            TTSResult with audio file path
        """
        url = self.base_url.rstrip("/") + "/v1/audio/speech"

        headers = {
            "Authorization": f"Bearer {self.config.get('api_key', '')}",
        }

        payload = {
            "model": "tts-1",
            "input": text,
            "voice": self.config.get("voice", "alloy"),
            "response_format": "mp3",
        }

        try:
            response = await self._http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            audio_content = response.content
        except httpx.HTTPError as e:
            logger.error(f"OpenAI TTS HTTP error: {e}")
            return TTSResult(
                success=False,
                error=f"OpenAI TTS HTTP error: {e}",
            )

        if output_path is None:
            unique_id = str(uuid.uuid4())[:8]
            output_path = self.media_dir / f"tts_{unique_id}.mp3"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(audio_content)

        audio_url = f"/media/output/{output_path.name}"

        logger.info(f"OpenAI TTS generated: {output_path}")

        return TTSResult(
            audio_path=output_path,
            audio_url=audio_url,
            success=True,
        )

    async def close(self) -> None:
        """Close the HTTP client connection pool."""
        if self._http_client:
            await self._http_client.aclose()
            logger.info("TTSService HTTP client closed")
