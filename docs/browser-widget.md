# Text speech voices for the Enterprise browser widget

The browser extension communicates with Enterprise, which proxies to core.
Core exposes GET /api/text/voices?language=de and accepts an optional voice
on POST /api/text/speech.

Voice discovery currently lists installed Piper voices matching the requested
base language. Other providers retain their configured default and return an
empty alternatives list. An explicit voice is validated against installed
language-matching choices before synthesis; selection never mutates shared
provider configuration or silently falls back to a different selected voice.
Omitting voice preserves the existing provider-routing/fallback behavior.

Translation and speech remain separate operations. Core reports generated audio
duration; Enterprise owns authentication, quotas and usage recording. This
change does not add usage accounting to the core.

Relevant tests are tests/test_text_voices.py, tests/test_text_speech.py,
tests/test_api_router.py, tests/test_tts_detailed.py, and
tests/test_tts_service_safety.py. Run with the declared
test-requirements.txt in a development/test container. The browser extension
repository contains a synthetic Chrome smoke test; it does not replace a local
Enterprise/account/provider integration check.
