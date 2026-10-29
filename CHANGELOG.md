# Changelog

All notable changes to PolyTalk CE are documented in this file.

PolyTalk CE uses app versions such as `0.4.0` and Git tags such as `v0.4.0-ce`.

## Unreleased

- Added text translation, text-to-speech, and voice catalog HTTP endpoints (`/api/v1/text/translations`, `/api/v1/text/speech`, `/api/v1/text/voices`).
- Added Redis-backed translation cache with TTL-bounded growth and privacy-safe logging (no transcript content in cache logs).
- Added configurable translation pacing (`translation_buffer_seconds`) controls for the streaming pipeline.
- Added optional company glossary terms to the text-translation endpoint request.
- Renamed the local Redis service to `polytalk-cache-redis` and restricted its published port to localhost.

## 0.4.0 - 2026-07-06

- Added Community Edition UI localization infrastructure with locale catalogs and browser-side translation support.
- Added UI locale selection with support for English, German, Spanish, French, Italian, Dutch, Polish, Portuguese, Czech, Danish, Finnish, Romanian, and Swedish.
- Added custom AI translation instructions so users can provide bounded session guidance for live, conversation, and tab-audio translation.
- Added bidirectional conversation mode for pause-delimited two-way translation turns.
- Added multiple translation provider routing with optional named providers and priority-based routing rules.
- Improved stream conversation transcript delivery before translation completes.
- Expanded Supertonic TTS language routing for additional supported languages.
- Added `CUSTOM_INSTRUCTION_MAX_CHARS` and translation provider routing examples to environment/config examples.
- Added tests for UI localization completeness, custom instructions, conversation mode, provider routing, and updated frontend behavior.

- Added optional company glossary terms to the text-translation endpoint request.

## 0.3.0 - 2026-06-13

- Added Supertonic TTS support for Japanese and Korean, including Docker Compose service configuration and provider-specific examples.
- Added language-based TTS provider routing so selected languages can use Supertonic while other languages continue using the default Piper path.
- Added context-aware translation prompts using bounded prior source/target translation history.
- Added optional shared tab/page visual context summarization for tab-audio sessions, with screenshot summaries passed as translation hints.
- Improved streaming STT behavior for leading startup silence and pause-flushed speech windows.
- Added GitLab CI checks for pre-commit validation, tests, and coverage artifacts.
- Expanded README and configuration examples for provider compatibility, visual context, STT tuning, benchmarking, and self-hosted deployment.
- Added tests for Supertonic TTS, contextual translation, visual context routing, and STT silence handling.

## 0.1.0 - 2026-05-29

- Initial public Community Edition release baseline.
- Provides the FastAPI application, browser UI, mock mode, Docker Compose setup, faster-whisper STT service, and Piper TTS service.
- Supports configurable STT, translation, and TTS providers for self-hosted deployments.
