# Changelog

All notable changes to PolyTalk CE are documented in this file.

PolyTalk CE uses app versions such as `0.3.0` and Git tags such as `v0.3.0-ce`.

## Unreleased

- No unreleased changes yet.

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
