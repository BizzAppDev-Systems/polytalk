# Provider Extension Guide

PolyTalk is intentionally service-oriented rather than tied to one hosted
provider. The current extension model is configuration-first:

- Use OpenAI-compatible STT, translation, or TTS endpoints when possible.
- Keep provider credentials in `.env`.
- Keep application behavior in `config/config.yaml`.
- Add or replace service classes only when a provider does not fit the existing
  HTTP/WebSocket contracts.

There is no runtime plugin loader yet. Custom providers are integrated by
implementing the base service interfaces in `app/services/base.py` and wiring
the implementation into the application service layer.

## Service Contracts

The main contracts are:

- `BaseTranscriptionService.stream_transcribe(...)`
- `BaseTranslationService.translate(...)`
- `BaseTTSService.synthesize(...)`

Implementations return these result objects:

- `TranscriptionResult`
- `TranslationResult`
- `TTSResult`

Keep provider-specific request payloads, authentication, retry behavior, and
response parsing inside the provider service class. The pipeline should receive
normalized result objects and should not need to know provider-specific details.

## STT Providers

The default `WhisperService` expects a WebSocket STT provider.

Configured keys:

```yaml
whisper:
  enabled: true
  mock_mode: false
  base_url: "${WHISPER_BASE_URL}"
  ws_endpoint: "${WHISPER_WS_ENDPOINT}"
  ping_interval_seconds: null
  ping_timeout_seconds: null
  max_reconnect_attempts: 3
  reconnect_delay_seconds: 2
```

Expected stream shape:

- The app sends raw `16 kHz`, mono, signed `int16` PCM bytes.
- The app may first send a JSON language hint: `{"language": "de"}`.
- The STT provider returns JSON messages with `text`, optional `language`,
  optional `is_final`, optional `metrics`, and optional `error`.

For a compatible custom STT service, deploy it separately and set:

```env
WHISPER_BASE_URL=https://stt.example.com
WHISPER_WS_ENDPOINT=/v1/stream/transcriptions
WHISPER_API_KEY=optional-token
```

If the provider uses a different wire format, create a new implementation of
`BaseTranscriptionService` that translates provider responses into
`TranscriptionResult`.

## Translation Providers

The default `TranslationService` targets OpenAI-compatible chat completions.

Configured keys:

```yaml
translation:
  enabled: true
  mock_mode: false
  base_url: "${TRANSLATION_BASE_URL}"
  endpoint: "/v1/chat/completions"
  api_key: "${TRANSLATION_API_KEY}"
  model: "${TRANSLATION_MODEL}"
  temperature: 0.0
  max_tokens: "${TRANSLATION_MAX_TOKENS}"
  system_prompt: "..."
```

Use this path for OpenAI, vLLM, and other providers that accept:

```json
{
  "model": "polytalk-translation",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.0,
  "max_tokens": 240
}
```

The response must include `choices[0].message.content`.

For non-compatible providers, implement `BaseTranslationService.translate()` and
return `TranslationResult(text=..., success=True)` on success. Do not put
provider-specific prompt rules in the pipeline; keep them in the provider
service or config.

## TTS Providers

The default `TTSService` supports:

- `provider: "piper"` for the local Piper service.
- Any other provider value currently falls back to the OpenAI speech API path.

Configured keys:

```yaml
tts:
  enabled: true
  mock_mode: false
  provider: "piper"
  base_url: "${TTS_BASE_URL}"
  voice: "en_GB-jenny_dioco-medium"
  timeout_seconds: 10
  default_voices:
    ar: "ar_JO-kareem-medium"
    de: "de_DE-karlsson-low"
    en: "en_GB-jenny_dioco-medium"
    es: "es_ES-davefx-medium"
    fr: "fr_FR-siwis-medium"
    hi: "hi_IN-priyamvada-medium"
    it: "it_IT-paola-medium"
    ml: "ml_IN-arjun-medium"
    nl: "nl_NL-ronnie-medium"
    ro: "ro_RO-mihai-medium"
    ru: "ru_RU-denis-medium"
    tr: "tr_TR-dfki-medium"
    zh: "zh_CN-huayan-medium"
```

Piper-compatible providers should expose:

- `GET /voices`
- `POST /` with `text`, `voice`, and optional provider parameters.
- Binary audio content in the HTTP response body.

Custom TTS services should implement `BaseTTSService.synthesize()` and return a
`TTSResult` with either:

- `audio_path` plus `audio_url` for files written under `media/output`, or
- an error result with `success=False` and a useful `error` message.

The frontend plays URLs emitted by the pipeline, so providers that return audio
bytes must still save or expose those bytes through an HTTP-accessible URL.

## Configuration Rules

- `.env` is for deployment-specific values: API keys, base URLs, model names,
  Docker model settings, ports, and CORS origins.
- `config/config.yaml` is for application behavior: mock mode, endpoints,
  prompts, voices, media storage, and timeout/flush settings.
- `config/config.yaml.example` and `.env.example` should stay safe for public
  repositories. Never place real tokens, private URLs, or model registry secrets
  in example files.

## Testing a Provider

Before wiring a provider into a live deployment:

1. Set the relevant service to `mock_mode: false`.
2. Run unit tests:

   ```bash
   pytest tests/ -v
   ```

3. Run the matching benchmark script from `tools/benchmarks/`.
4. Run a full pipeline benchmark if STT, translation, or TTS timing changed.
5. Test microphone input and tab audio separately. Tab audio is cleaner and may
   hide microphone-specific STT issues.

## Current Limits

- Provider selection is not dynamically loaded from entry points or a plugin
  directory.
- The pipeline currently instantiates the built-in service classes.
- Deeper provider swaps may require a small code change in the service wiring.

For open-source deployments, prefer compatibility adapters around existing
provider protocols before adding new pipeline behavior.

## Built-in specialized speech providers

The specialized providers keep the public PolyTalk contracts unchanged. Build
and start the standard Compose stack with:

    HF_TOKEN=hf_read_token_after_accepting_both_gated_model_terms
    docker compose build --progress=plain
    docker compose up -d --no-build

Explicit source languages route through the configured STT provider list.
Unknown-language and conversation sessions stay on faster-whisper unless a
future probing policy is enabled. If a specialized backend fails before it
emits text, buffered PCM is replayed once to the configured Whisper fallback.

Indian target languages route to the Indic Parler adapter, whose binary WAV
response is normalized to the same media URL contract used by Piper and
Supertonic. Punjabi support is unofficial and should be treated as
experimental. Provider errors fall back to Piper.

The specialized STT and Indic Parler images download their configured models
during build. The Indic Parler image clones and fast-forward-pulls the inference
repository, then downloads the gated model and its description tokenizer using
a BuildKit secret. The token is not stored in the image. No host source or model
path is mounted into these containers. Runtime model access is offline.
