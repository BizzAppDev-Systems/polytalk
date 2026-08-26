<p align="center">
  <a href="https://www.polytalk.io/">
    <img src="assets/logo.png" alt="PolyTalk" width="260">
  </a>
</p>

<h1 align="center">PolyTalk Community Edition</h1>

<p align="center">
  <strong>Self-hosted, privacy-first speech-to-speech live translation.</strong><br>
  Speak in one language. Hear it in another. Run the stack on infrastructure you control.
</p>

<p align="center">
  <a href="https://www.polytalk.io/"><strong>Website</strong></a> ·
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="docs/production-deployment.md"><strong>Production Guide</strong></a> ·
  <a href="CONTRIBUTING.md"><strong>Contribute</strong></a>
</p>

<p align="center">
  <a href="https://github.com/PolyTalkIO/polytalk/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/PolyTalkIO/polytalk/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: AGPL-3.0" src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg"></a>
  <a href="https://www.polytalk.io/"><img alt="Website" src="https://img.shields.io/badge/Website-polytalk.io-2ea44f.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-powered-009688.svg">
</p>

PolyTalk CE is an open-source FastAPI application for live speech translation. It records microphone audio in the browser, streams it through transcription, translation, and text-to-speech services, then plays translated speech back in near real time.

Created and maintained by BizzAppDev Systems Pvt. Ltd.

<p align="center">
  <a href="https://www.polytalk.io/"><strong>Visit polytalk.io</strong></a> · <a href="#quick-start"><strong>Run PolyTalk locally</strong></a>
</p>

## Why PolyTalk

- **Own the pipeline**: self-host speech processing and keep user audio under your control.
- **Start safely**: mock mode works without API keys or external services.
- **Deploy practically**: Docker Compose includes the app, faster-whisper STT, and Piper TTS services.
- **Bring your provider**: use OpenAI-compatible, Anthropic-style, Gemini-style, or self-hosted translation APIs.
- **Tune for real conversations**: adjust streaming latency, batching, VAD, and TTS settings for your environment.

## How It Works

1. **Capture**: PolyTalk receives live audio from a microphone, tab, or other browser-supported source.
2. **Listen**: The audio stream is prepared for real-time processing.
3. **Understand**: Speech becomes readable text.
4. **Translate**: The text is converted into the target language.
5. **Respond**: Users receive translated text and translated speech.

```text
Browser audio source -> PolyTalk live pipeline
                           |
                           +-- 1. Listen: live audio stream
                           |
                           +-- 2. Understand: speech -> text
                           |       faster-whisper / Whisper-compatible
                           |
                           +-- 3. Translate: text -> target language
                           |       OpenAI-compatible / Ollama / vLLM / Anthropic / Gemini
                           |
                           +-- 4. Respond
                                   +-- translated text   -> Browser UI
                                   +-- translated speech -> Browser playback
                                       Piper / compatible TTS
```

## Features

- Clean, modular architecture with service-oriented design
- Mock mode for testing without external API keys
- Configurable via YAML config file and environment variables
- Docker and Docker Compose support
- Simple vanilla JavaScript frontend
- Easy to extend for additional providers and workflows

## Use Cases

- Live multilingual meetings, calls, and demos
- Customer support conversations across languages
- Field-team, clinic, and service-desk communication where privacy matters
- Classroom, training, and onboarding translation
- Private or offline speech workflows on controlled infrastructure
- Self-hosted AI prototypes that need a complete speech-to-speech pipeline

## Why Self-Host PolyTalk?

- Keep audio, transcripts, translations, and generated speech on infrastructure you control.
- Avoid hard dependency on a single hosted speech or translation vendor.
- Tune latency, batching, VAD, model size, workers, and translation context for real deployments.
- Run with CPU, GPU, local open-weight models, private APIs, or hosted providers.
- Use mock mode for safe local demos and CI-style checks without API keys.

## Provider Compatibility

PolyTalk is configuration-first and provider-flexible. The default Docker Compose
stack includes local STT and TTS services, while translation can point to hosted
or self-hosted model APIs.

| Pipeline stage | Built-in/default path | Compatible options |
|----------------|-----------------------|--------------------|
| STT | faster-whisper service over WebSocket | Whisper-compatible WebSocket services that accept 16 kHz mono int16 PCM |
| Translation | OpenAI-compatible chat completions | Ollama, vLLM, LM Studio, LiteLLM, OpenAI-compatible Responses, Anthropic Messages-style, Gemini Generate Content-style |
| TTS | Local Piper HTTP service | Piper-compatible HTTP services, configured OpenAI-style TTS fallback |

See [Provider Extension](docs/provider-extension.md) for service contracts,
wire formats, and guidance for adding custom providers.

## Self-Hosted vs Hosted-Only APIs

PolyTalk is designed for teams that want live speech-to-speech translation
without giving up deployment control. Hosted-only translation APIs can be useful
when you want a managed service, but PolyTalk gives you an open-source pipeline
that can run in your own environment, mix local and remote providers, and keep
the browser, WebSocket pipeline, STT, translation, and TTS layers configurable.

The Community Edition is AGPL-3.0 licensed for open-source use, with commercial
licensing available for proprietary deployments.

## Quick Start

### Prerequisites

- Python 3.10+ (for local development)
- Docker and Docker Compose (for containerized deployment)

### Local Development

**Note**: Local development starts safely in mock mode. To use real audio translation without Docker, configure external STT, translation, and TTS services.

1. **Clone and setup**:
   ```bash
   cp .env.example .env
   cp config/config.yaml.example config/config.yaml
   ```

2. **Configure** (edit `config/config.yaml`):
   - Set `mock_mode: true` for testing without API keys
   - Set `mock_mode: false` and add API keys for production use
   - `config/config.yaml.example` defaults to mock mode, so copying it as-is runs the UI flow without real STT, translation, or TTS calls.

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python -m app.main
   ```

5. **Access the app**:
   Open http://localhost:9000 in your browser

### Docker Deployment (Recommended)

The Docker Compose setup includes:
- **STT Service**: Self-hosted faster-whisper transcription service
- **TTS Service**: Self-hosted Piper text-to-speech service
- **PolyTalk App**: Main application with all services

1. **Setup environment**:
   ```bash
   cp .env.example .env
   cp config/config.yaml.example config/config.yaml
   # Edit .env with your API keys (optional for mock mode)
   ```

2. **Build and run**:
   ```bash
   docker compose up -d
   ```

   For GPU-backed STT, use the GPU override:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
   ```

3. **Access the app**:
   Open http://localhost:9000 in your browser

**Note**: Port 9000 is used for the PolyTalk app. The STT and TTS services run on the internal Docker network.

## Configuration

### Config File (`config/config.yaml`)

The main configuration file supports these sections:

#### Whisper (Transcription)
```yaml
whisper:
  enabled: true
  mock_mode: true  # Set false for real API calls
  base_url: "${WHISPER_BASE_URL}"
  ws_endpoint: "${WHISPER_WS_ENDPOINT}"
  max_reconnect_attempts: 3
```

#### Translation
```yaml
translation:
  enabled: true
  mock_mode: true  # Set false for real API calls
  api_format: "${TRANSLATION_API_FORMAT}"
  base_url: "${TRANSLATION_BASE_URL}"
  endpoint: "${TRANSLATION_ENDPOINT}"
  api_key: "${TRANSLATION_API_KEY}"
  model: "${TRANSLATION_MODEL}"
  temperature: 0.0
  max_tokens: "${TRANSLATION_MAX_TOKENS}"
  context_enabled: true
  context_max_chunks: 4
  context_max_chars: 1200
  context_payload_warn_chars: 2000
```

Names under `translation.providers` are operator-defined provider IDs, while
`api_format` selects the request/response adapter. For example, the retained
provider ID `translategamma` uses `api_format: translategemma_chat`. See
`config/config.yaml.example` for the complete multi-provider routing example.

#### TTS (Text-to-Speech)
```yaml
tts:
  enabled: true
  mock_mode: true  # Set false for real API calls
  provider: "piper"
  base_url: "${TTS_BASE_URL}"
  voice: "en_GB-jenny_dioco-medium"
  timeout_seconds: 10
```

#### App
```yaml
app:
  host: "${APP_HOST}"
  port: "${APP_PORT}"
  debug: "${APP_DEBUG}"
  save_media: true
  media_output_dir: "media/output"
```

### Environment Variables

Copy `.env.example` to `.env` and set:

```bash
TRANSLATION_API_FORMAT=openai_chat
TRANSLATION_API_KEY=your_key_here
TRANSLATION_BASE_URL=https://ai.example.com
TRANSLATION_ENDPOINT=/v1/chat/completions
TRANSLATION_MODEL=qwen3-8b
TRANSLATION_MAX_TOKENS=240
WHISPER_BASE_URL=http://stt:8000
TTS_BASE_URL=http://tts:5000
APP_PORT=9000
ALLOWED_ORIGINS=http://localhost:9000,http://127.0.0.1:9000
LOG_LEVEL=INFO
```

Supported translation API formats:

| Format | Base URL | Endpoint | Response parser | Test status |
|--------|----------|----------|-----------------|-------------|
| `openai_chat` | `https://ai.example.com` | `/v1/chat/completions` | `choices[0].message.content` | Fully tested with OpenAI-compatible chat providers |
| `openai_chat` for Ollama | `http://localhost:11434` | `/v1/chat/completions` | OpenAI-compatible chat response | Fully tested as OpenAI-compatible chat format |
| `openai_chat` for vLLM | your vLLM server URL | `/v1/chat/completions` | OpenAI-compatible chat response | Fully tested as OpenAI-compatible chat format |
| `openai_responses` | OpenAI-compatible Responses server URL | `/v1/responses` | `output_text` | Fully tested with OpenAI-compatible Responses format |
| `anthropic_messages` | `https://api.anthropic.com` | `/v1/messages` | text blocks in `content[]` | Adapter unit-tested; live provider not fully tested |
| `gemini_generate_content` | `https://generativelanguage.googleapis.com/v1beta` | `/models/{model}:generateContent` | `candidates[].content.parts[].text` | Adapter unit-tested; live provider not fully tested |

Set `TRANSLATION_BASE_URL` to your self-hosted AI server when using OpenAI-compatible local or private deployments such as Ollama, vLLM, LM Studio, or LiteLLM. `TRANSLATION_ENDPOINT` is used as-is unless it contains `{model}`, in which case PolyTalk substitutes the configured `TRANSLATION_MODEL` before sending the request.

Use `.env` for deployment-specific values such as API keys, service URLs, ports, and Docker model settings. Use `config/config.yaml` for application behavior such as mock mode, translation prompts, voice defaults, media storage, and latency thresholds.

### Shared Voice Activity Detection

PolyTalk can segment conversation, live microphone, and shared/tab audio once in
the CE pipeline before routing to any STT provider. Each profile has four rollout
modes: `off` preserves legacy behavior, `shadow` runs inference without changing
the stream, `boundary` adds server pause boundaries without filtering audio, and
`active` forwards only confirmed speech with pre-roll/post-roll protection.

Keep all profiles `off` until the recorded regression corpus passes. Roll out one
profile at a time through `shadow`, `boundary`, and `active`. The browser switches
to continuous PCM only after CE confirms boundary/active VAD; STT then acknowledges
`input_segmentation=upstream_vad` and bypasses its transport RMS gate. A VAD model
load or inference failure fails open to legacy pass-through audio.

### Latency Tuning

The main latency knobs are:

| Setting | Default | Description |
|---------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Application and STT service log level. Set `DEBUG` to include streaming diagnostics. |
| `VAD_CONVERSATION_MODE` | `off` | Shared CE VAD rollout for conversation audio: `off`, `shadow`, `boundary`, or `active`. |
| `VAD_LIVE_MICROPHONE_MODE` | `off` | Shared CE VAD rollout for live microphone audio. |
| `VAD_SHARE_MODE` | `off` | Shared CE VAD rollout for shared/tab audio. |
| `STT_STREAM_CHUNK_SECONDS` | `3.0` | Audio window processed by the STT service. Lower values reduce first transcript latency; higher values can improve transcript stability. |
| `STT_EMIT_MIN_CHARS` | `120` | Minimum new transcript text before STT emits an update to PolyTalk. Increase this if live chunks are too small. |
| `STT_EMIT_INTERVAL_SECONDS` | `4.5` | Maximum time to hold pending transcript text before emitting it. |
| `STT_PAUSE_FLUSH_SECONDS` | `2.0` | Flush and emit the current speech window after this much trailing silence. Set `0` to disable pause flushing. |
| `STT_LEADING_SILENCE_PREROLL_SECONDS` | `0.2` | Keep this much audio before first detected speech while discarding longer tab-share startup silence. |
| `STT_SILENCE_RMS_THRESHOLD` | `0.003` | Skip STT for very quiet audio windows. Raise this if Whisper hallucinates while nobody is speaking. |
| `STT_NO_SPEECH_PROB_THRESHOLD` | `0.50` | Drop faster-whisper segments classified as likely no-speech. |
| `STT_LOG_PROB_THRESHOLD` | `-1.0` | Drop low-confidence faster-whisper segments. |
| `STT_MAX_CROSS_DELTA_WORD_REPEATS` | `6` | Stop appending the same leading word across transcript updates after this many existing repeats. |
| `STT_VAD_FILTER` | `true` | Enable faster-whisper VAD before decoding. |
| `STT_VAD_MIN_SILENCE_MS` | `500` | Silence duration used by VAD to split speech. Raise for fewer, larger speech regions. |
| `STT_VAD_SPEECH_PAD_MS` | `200` | Padding kept around detected speech. Raise if words are clipped near speech boundaries. |
| `STT_CONDITION_ON_PREVIOUS_TEXT` | `false` | Reuse previous Whisper text as context. Keep disabled for lowest hallucination risk in streaming. |
| `STT_INITIAL_PROMPT` | empty | Optional Whisper prompt for domain terms, names, and expected vocabulary. |
| `whisper.ping_interval_seconds` | `null` | App-to-STT WebSocket ping interval. `null` disables client pings, which avoids timeouts during long local model inference. |
| `whisper.ping_timeout_seconds` | `null` | App-to-STT WebSocket ping timeout. |
| `APP_WORKERS` | `3` | Number of PolyTalk app Gunicorn workers. Increase for more concurrent sessions after checking CPU and memory headroom. |
| `STT_WORKERS` | `1` | Number of STT web workers. Each worker loads its own Whisper model. |
| `STT_PRELOAD_MODEL` | `true` | Load the Whisper model during STT startup instead of delaying the first stream. |
| `STT_CHUNK_OVERLAP_SECONDS` | `0.25` | Audio overlap between STT windows. Helps avoid missing words at chunk boundaries. |
| `STT_TRANSCRIBE_WORKERS` | `2` | Per-stream STT transcription workers. Use more than 1 only when the GPU has spare compute. |
| `STT_TRANSCRIBE_QUEUE_SIZE` | `8` | Max queued audio windows per stream before receiver backpressure. |
| `STT_MODEL_WORKERS` | `2` | faster-whisper/CTranslate2 model workers for concurrent transcribe calls. |
| `VISUAL_CONTEXT_ENABLED` | empty/false | Enable one-time shared tab/page screenshot summarization when tab audio sharing starts. |
| `VISUAL_CONTEXT_BASE_URL` | `TRANSLATION_BASE_URL` | Optional separate base URL for the vision-capable visual context provider. |
| `VISUAL_CONTEXT_API_KEY` | `TRANSLATION_API_KEY` | Optional separate API key for the visual context provider. |
| `VISUAL_CONTEXT_ENDPOINT` | `TRANSLATION_ENDPOINT` | Optional separate endpoint for the visual context provider. |
| `VISUAL_CONTEXT_API_FORMAT` | `TRANSLATION_API_FORMAT` | Optional separate API format for the visual context provider. |
| `VISUAL_CONTEXT_MODEL` | `TRANSLATION_MODEL` | Vision-capable model used to summarize the shared tab/page screenshot. |
| `VISUAL_CONTEXT_MAX_TOKENS` | `240` | Maximum output tokens for the visual context summary. |
| `app.translation_flush_chars` | `300` | Translate buffered text once this many characters are available. |
| `app.translation_flush_seconds` | `6.0` | In paced live/tab sessions, ask the semantic selector for a safe original-text prefix after this age. |
| `app.translation_flush_min_chars` | `120` | Minimum text required for legacy non-session time-based translation flushing. |
| `TRANSLATION_PAUSE_COMMIT_MODE` | `immediate` | `immediate` trusts an explicit acoustic pause and commits the whole buffer; `semantic` asks Qwen for a safe prefix. |
| `TRANSLATION_SEMANTIC_HARD_SECONDS` | `10.0` | Local hard deadline that flushes the buffer without waiting longer for the selector. |
| `SEMANTIC_BOUNDARY_ENABLED` | `true` | Enable Qwen boundary selection. Translation itself still follows the configured language routing. |
| `SEMANTIC_BOUNDARY_PROVIDER` | empty | Optional named Qwen provider used only for boundary selection. Empty uses `translation.default_provider` or the flat translation endpoint. |
| `SEMANTIC_BOUNDARY_MIN_CONFIDENCE` | `0.75` | Reject lower-confidence boundary IDs and keep buffering until a pause or hard deadline. |
| `SEMANTIC_BOUNDARY_CHAT_TEMPLATE_KWARGS_ENABLED` | `true` | Send vLLM's `enable_thinking` chat-template option. Disable for OpenAI-compatible providers that reject this extension. |
| `SEMANTIC_BOUNDARY_PROBE_COOLDOWN_SECONDS` | `3.0` | Minimum per-session delay between selector calls for changing buffer content. |
| `translation.model` | `qwen3-8b` | Use a model supported by your provider or self-hosted server, such as qwen3-8b, TranslateGama, or another open-source/open-weight model. |
| `translation.max_tokens` | `240` | Maximum translation output tokens. Keep bounded for live streaming, but allow enough room for Indic-script targets and longer sentence buffers. |
| `translation.context_enabled` | `true` | Send recent successful source/target translation pairs as read-only context for later chunks. |
| `translation.context_max_chunks` | `4` | Maximum previous translation chunks kept in per-session context. |
| `translation.context_max_chars` | `1200` | Maximum source plus translated characters kept in per-session context. |
| `translation.context_payload_warn_chars` | `2000` | Log a warning when final system plus user prompt text exceeds this many characters. Set `0` to disable. |

Live and shared-tab clients may override `app.translation_flush_seconds` for a
single WebSocket session with the bounded `translation_buffer_seconds` query
parameter (1–10 seconds). They may also update it during the session by sending
`{"type":"translation_buffer_config","translation_buffer_seconds":6}`. The
session value holds translation/TTS across short STT emissions; cumulative
transcription events remain immediate. Conversation mode remains pause-delimited.

A client can send `{"type":"end_of_utterance","reason":"audio_gap"}` after a
simple acoustic gap. PolyTalk forwards a non-terminal `flush` control to STT;
the resulting `force_emit` transcript applies `TRANSLATION_PAUSE_COMMIT_MODE`.
Without a pause, Qwen receives numbered candidate boundaries at the session
interval and returns only a boundary ID, confidence, and reason. PolyTalk slices
the original transcript locally, so the selector cannot rewrite or translate
customer text. TranslateGemma/Qwen language routing is used only by the normal
translation call. At `TRANSLATION_SEMANTIC_HARD_SECONDS`, PolyTalk locally
flushes the remaining buffer even if the selector is slow or finds no boundary.

PolyTalk also sends the session value to the selected STT provider in its initial
WebSocket control message and forwards later updates. For specialized providers,
`STT_STREAM_CHUNK_SECONDS` is therefore the startup fallback and optional
partial-transcript cadence, not a fixed live-session latency floor.
| `tts.timeout_seconds` | `10` | Maximum wait for TTS generation. |
| `TTS_WORKERS` | `4` | Number of Piper Gunicorn workers. Keep `2-4` on small hosts; raise toward `min(8, CPU cores)` only after CPU and memory headroom are confirmed. |

For larger continuous-speech translation chunks, start with:

```bash
STT_STREAM_CHUNK_SECONDS=3.0
STT_EMIT_MIN_CHARS=120
STT_EMIT_INTERVAL_SECONDS=4.5
STT_PAUSE_FLUSH_SECONDS=2.0
STT_LEADING_SILENCE_PREROLL_SECONDS=0.2
```

```yaml
app:
  translation_flush_chars: 300
  translation_flush_seconds: 6.0
  translation_flush_min_chars: 120
```

For lower latency, start with:

```bash
STT_STREAM_CHUNK_SECONDS=0.5
STT_EMIT_MIN_CHARS=24
STT_EMIT_INTERVAL_SECONDS=1.0
STT_PAUSE_FLUSH_SECONDS=1.0
```

```yaml
app:
  translation_flush_chars: 60
  translation_flush_seconds: 0.8
  translation_flush_min_chars: 16
```

For better translation quality, increase those values so the translation model receives more context.

Set `LOG_LEVEL=DEBUG` when diagnosing latency. Streaming debug logs include STT
queue wait, STT inference time, emit delay, ASR-to-translation queue wait,
translation request time, and TTS queue/duration. When `LOG_LEVEL` is unset,
PolyTalk defaults to `INFO`.

### Benchmark Preview

PolyTalk includes small benchmark scripts for measuring each stage of the live
translation path before you tune a deployment.

| Benchmark | What it measures | Script |
|-----------|------------------|--------|
| STT | First transcript timing and transcription service behavior | `tools/benchmarks/benchmark_stt.py` |
| Translation | Translation provider latency for repeated text chunks | `tools/benchmarks/benchmark_translation.py` |
| TTS | Speech synthesis latency and generated audio size | `tools/benchmarks/benchmark_tts.py` |
| Full pipeline | First transcription, first translation, first TTS, event counts, and p50/p95 event arrival times | `tools/benchmarks/benchmark_pipeline.py` |

See [Benchmarking](docs/benchmarking.md) for sample commands, fixture audio,
and guidance on reading results.

## API Endpoints

### `GET /api/health`

Health check endpoint.

### `WS /api/ws/translate`

Streaming translation endpoint used by the frontend.

Query parameters:
- `source_language`: Source language code (default: `en`)
- `target_language`: Target language code (default: `gu`)

Example:

```text
ws://localhost:9000/api/ws/translate?source_language=en&target_language=hi
```

The browser sends raw 16 kHz mono int16 PCM chunks as binary WebSocket messages. The server returns JSON messages with these common `type` values:

- `transcription`: Current STT transcript
- `translation`: Current accumulated translation
- `tts`: URL for generated speech audio
- `language_swapped`: Confirmation after a live source/target swap
- `error`: Pipeline error

## Project Structure

```
.
├── app/
│   ├── main.py              # FastAPI application entry
│   ├── config.py            # Configuration loader
│   ├── schemas/             # Pydantic models
│   ├── routers/             # API and web routers
│   ├── services/            # Business logic services
│   ├── templates/           # Jinja2 templates
│   ├── static/              # CSS and JavaScript
│   └── utils/               # Utility functions
├── config/
│   ├── config.yaml          # Active configuration
│   └── config.yaml.example  # Config template
├── media/
│   └── output/              # Generated media files
├── tests/                   # Test files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example             # Env template
└── README.md
```

## Supported Languages

- English (en)
- Gujarati (gu)
- Hindi (hi)
- Spanish (es)
- Spanish, Mexico (es_MX)
- French (fr)
- German (de)
- Portuguese (pt)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)
- Russian (ru)
- Italian (it)
- Dutch (nl)
- Dutch, Belgium (nl_BE)
- Turkish (tr)
- Tamil (ta)
- Bengali (bn)
- Marathi (mr)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)

## Mock Mode

Mock mode allows testing the full application flow without API keys:

- **Mock Transcription**: Returns pre-defined sample text
- **Mock Translation**: Returns pre-defined translations
- **Mock TTS**: Generates a silence audio file

`config/config.yaml.example` defaults all service sections to `mock_mode: true`
for safe local startup. With those defaults, PolyTalk will not call real STT,
translation, or TTS services. For a real deployment, set `mock_mode: false` in
the `whisper`, `translation`, and `tts` sections and configure the matching
service URLs/API keys in `.env`.

## Extending the Application

The modular service architecture makes it easy to extend PolyTalk:

- **Add new services**: Create service classes in `app/services/` that implement base interfaces
- **Add new providers**: Implement provider-specific services (e.g., TTS providers) and configure them
- **Add features**: Extend the pipeline service to orchestrate new capabilities

## License

PolyTalk Community Edition is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### What This Means:
- **Free to use**: You can run, modify, and distribute this software
- **Copyleft**: Modifications must be licensed under AGPL-3.0
- **Network use**: If you run a modified version as a service, you must provide its source code

### Commercial Licensing

If you need to use PolyTalk under different terms (e.g., for proprietary or closed-source projects), a **commercial license** is available. Contact **BizzAppDev Systems Pvt. Ltd.** for licensing options.

PolyTalk follows a dual-licensing model: AGPL-3.0 for Community Edition and proprietary licensing for Enterprise Edition.

### SaaS Hosting:
If you are using PolyTalk as a hosted service, you must provide users with access to the corresponding source code as required by AGPL-3.0 Section 13.

See [LICENSE](LICENSE) for the full AGPL-3.0 text and [COPYRIGHT](COPYRIGHT) for copyright information.

## Contributing

We welcome contributions from the community!

Start with [CONTRIBUTING.md](CONTRIBUTING.md) for setup, validation, and merge
request expectations. For security issues, follow [SECURITY.md](SECURITY.md)
instead of opening a public issue.

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Requirements:
- Follow Black formatting standards
- Include type hints where practical
- Add docstrings to all public methods
- Write tests for new features
- Ensure all tests pass before submitting PR

> **Contribution Agreement**: By contributing to this project, you agree that:
> 1. Your contributions will be licensed under AGPL-3.0
> 2. You grant BizzAppDev Systems Pvt. Ltd. the right to use, modify, and relicense your contributions as part of commercial or proprietary versions of PolyTalk

## Specialized speech providers

Build and run IndicConformer, Parakeet TDT v3, SenseVoiceSmall, and Indic Parler
TTS as part of the standard Compose stack.
Accept the gated IndicConformer and Indic Parler model terms using the same
Hugging Face account first, then set `HF_TOKEN` in the ignored `.env` file or
export it in the build shell:

    HF_TOKEN=hf_read_token_after_accepting_the_model_terms
    docker compose --progress=plain build
    docker compose up -d --no-build

Indic Parler is the active provider for the configured Indian target languages.
Punjabi support is unofficial and should be treated as experimental.

Source repositories and model weights are fetched inside Docker builds; no
developer-local model directory is used. See
[the provider extension guide](docs/provider-extension.md) for routing details.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows Black formatting standards:

```bash
black app/
```

### Benchmarking

See [Benchmarking](docs/benchmarking.md) for STT, translation, TTS, and full
pipeline latency benchmark scripts.

### Provider Extension

See [Provider Extension](docs/provider-extension.md) for the current STT,
translation, and TTS service contracts, compatible provider expectations, and
guidance for adding custom provider adapters.

## Troubleshooting

- Browser cannot access the microphone: use `localhost` or HTTPS and allow microphone permission.
- WebSocket disconnects behind a proxy: confirm the proxy supports WebSocket upgrade and has a long enough read timeout.
- Translation fails: check `TRANSLATION_API_KEY`, `TRANSLATION_BASE_URL`, `translation.model`, and `translation.mock_mode`.
- STT is slow on CPU: use a smaller model, keep `STT_COMPUTE_TYPE=int8`, or use CUDA with `STT_DEVICE=cuda` and `STT_COMPUTE_TYPE=float16`.
- CUDA fails inside Docker: confirm `docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec stt nvidia-smi` works. If it does not, install/configure NVIDIA Container Toolkit or run with the GPU override file.
- TTS voice is missing: check `tts.default_voices` and the voice files mounted under `tts/voices`.
- Audio files accumulate: generated files are stored in `media/output`; clear or rotate that directory in production.

## Production Notes

- See [Production Deployment](docs/production-deployment.md) for a complete
  Docker, GPU STT, nginx, HTTPS, and translation-provider deployment guide.
- Set `APP_DEBUG=false`.
- Set `ALLOWED_ORIGINS` to the exact browser origins that should access the app.
- Put the app behind a reverse proxy that supports WebSockets.
- Persist `media/output` if generated audio should survive restarts.
- Treat transcripts, translations, and generated audio as user data.
- Review AGPL-3.0 obligations before offering a modified hosted service.

### Shared Tab Visual Context

When `VISUAL_CONTEXT_ENABLED=true`, tab-audio sessions capture one browser-approved shared tab/page screenshot after sharing starts. PolyTalk sends the image for immediate summarization and does not store the raw screenshot. The generated summary is used as a translation hint for visible titles, names, labels, and domain vocabulary; spoken audio remains authoritative if it conflicts with the visual hint.
