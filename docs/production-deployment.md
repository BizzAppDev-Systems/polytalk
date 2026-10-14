# Production Deployment

This guide covers a practical production-style PolyTalk deployment using Docker
Compose, GPU-backed STT, local Piper TTS, an OpenAI-compatible translation
service, and nginx/Let's Encrypt in front of the application.

Use this as an operator runbook for open-source deployments. It intentionally
keeps provider-specific model serving outside this repository so API keys,
private model paths, and GPU-specific launch flags do not leak into source
control.

## Architecture

```text
Browser
  |
  | HTTPS + WebSocket
  v
nginx
  |
  | http://127.0.0.1:9000
  v
polytalk
  |-- ws://stt:8000/v1/stream/transcriptions (faster-whisper fallback)
  |-- ws://indicconformer-stt:8000/v1/stream/transcriptions
  |-- ws://parakeet-stt:8000/v1/stream/transcriptions
  |-- ws://sensevoice-stt:8000/v1/stream/transcriptions
  |-- http://tts:5000
  `-- http(s)://translation-provider/v1/chat/completions
```

The translation provider can be OpenAI, vLLM, Ollama with an OpenAI-compatible
proxy, or any service that implements `/v1/chat/completions`.

## Production Checklist

- Run behind HTTPS. Browsers require a secure context for microphone access on
  non-localhost domains.
- Set `APP_DEBUG=false` and `LOG_LEVEL=INFO` for normal operation.
- Keep `.env`, `config/config.yaml`, model tokens, and provider credentials out
  of git.
- Use `config/config.yaml.example` as a template, then set `mock_mode: false`
  for `whisper`, `translation`, and `tts`.
- Restrict `ALLOWED_ORIGINS` to the real HTTPS origin.
- Put nginx or another reverse proxy in front of the app with WebSocket upgrade
  headers and long read/send timeouts.
- Persist or rotate `media/output`; generated TTS audio is user data.
- Run benchmarks from staging or during an announced maintenance window, not
  during normal demo traffic.

## Server Prerequisites

- Linux server with Docker Engine and Docker Compose plugin.
- A domain pointing to the server.
- Ports `80` and `443` open for nginx and Let's Encrypt.
- NVIDIA driver and NVIDIA Container Toolkit if using GPU STT.
- Enough disk space for STT models, TTS voices, and generated media.

For GPU STT, verify Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Prepare Configuration

```bash
cp .env.example .env
cp config/config.yaml.example config/config.yaml
```

For production, set all service sections in `config/config.yaml` to real mode:

```yaml
whisper:
  mock_mode: false

translation:
  mock_mode: false

tts:
  mock_mode: false
```

Use `.env` for deployment-specific values:

```env
APP_DEBUG=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://polytalk.example.com

WHISPER_BASE_URL=http://stt:8000
WHISPER_WS_ENDPOINT=/v1/stream/transcriptions

TTS_BASE_URL=http://tts:5000
TTS_MODEL=en_GB-jenny_dioco-medium

TRANSLATION_BASE_URL=http://your-translation-service:8000
TRANSLATION_MODEL=polytalk-translation
TRANSLATION_API_KEY=
TRANSLATION_MAX_TOKENS=240
TRANSLATION_TYPED_TEXT_MAX_TOKENS=1536
```

For GPU STT, start with:

```env
STT_MODEL=large-v3
STT_DEVICE=cuda
STT_COMPUTE_TYPE=float16
STT_WORKERS=1
STT_PRELOAD_MODEL=true
STT_TRANSCRIBE_WORKERS=2
STT_MODEL_WORKERS=2
STT_SAMPLE_RATE=16000

STT_STREAM_CHUNK_SECONDS=1.2
STT_CHUNK_OVERLAP_SECONDS=0.25
STT_SILENCE_RMS_THRESHOLD=0.003
STT_NO_SPEECH_PROB_THRESHOLD=0.50
STT_LOG_PROB_THRESHOLD=-1.0
STT_MAX_CROSS_DELTA_WORD_REPEATS=6
STT_VAD_FILTER=true
STT_VAD_MIN_SILENCE_MS=500
STT_VAD_SPEECH_PAD_MS=200
STT_CONDITION_ON_PREVIOUS_TEXT=false
STT_TEMPERATURE=0.0
```

## Conversation STT Routing Rollout

`config/config.yaml` and `.env` are intentionally ignored by git. Pulling or
merging a release therefore does not update an existing production mapping.
Back up both files and merge the `whisper` routing keys from
`config/config.yaml.example` before enabling the new router.

The checked-in mapping is:

| Provider | Languages |
| --- | --- |
| IndicConformer | `as`, `bn`, `brx`, `doi`, `gu`, `hi`, `kn`, `kok`, `ks`, `mai`, `ml`, `mni`, `mr`, `ne`, `or`, `pa`, `sa`, `sat`, `sd`, `ta`, `te`, `ur` |
| SenseVoice | `zh`, `yue`, `ja`, `ko` |
| Parakeet | `en`, `bg`, `hr`, `cs`, `da`, `nl`, `et`, `fi`, `fr`, `de`, `el`, `hu`, `it`, `lv`, `lt`, `mt`, `pl`, `pt`, `ro`, `sk`, `sl`, `es`, `sv`, `ru`, `uk` |
| faster-whisper conversation fallback | `ar`, `id`, `tr`, `vi` |

Locale variants are normalized before lookup, including `nl_BE` to `nl` and
`es_MX` to `es`. Keep `language_detection: true` only for workers that return an
independently detected language code. Parakeet automatically chooses a
transcription language, but does not return that code; it must remain `false`.
Same-script pairs handled only by a provider without language-ID output use the
configured source language as the phase-1 tie-breaker until the separate
language detector is available.

Use a staged rollout so an incomplete provider set cannot reject conversation
turns:

1. Back up `.env` and `config/config.yaml`. Merge the complete `whisper`
   `conversation`, `providers`, and `routing` blocks from the example file.
2. Set `STT_SAMPLE_RATE=16000` for the app and every STT worker. Raw PCM does
   not carry sample-rate metadata, so these values must not differ.
3. Accept the gated IndicConformer model terms and export `HF_TOKEN` in the
   shell that builds its image. Configure all three provider URLs, model IDs,
   enable flags, and the CPU/CUDA device in `.env`.
4. Start with `STT_ROUTING_ENABLED=false`. Build/start the workers and verify
   their loopback health endpoints before exposing routing:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build \
     stt indicconformer-stt parakeet-stt sensevoice-stt
   curl --fail http://127.0.0.1:8000/health
   curl --fail http://127.0.0.1:8011/health
   curl --fail http://127.0.0.1:8012/health
   curl --fail http://127.0.0.1:8013/health
   ```

5. Enable all configured providers and `STT_ROUTING_ENABLED=true`, then
   recreate `polytalk`. Smoke-test at least one cross-provider pair, one
   SenseVoice pair, one Parakeet pair, and the four faster-whisper fallback
   languages.

If any mapped worker is unavailable, disable routing and recreate only the app
to restore the legacy faster-whisper path:

```bash
STT_ROUTING_ENABLED=false docker compose \
  -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate polytalk
```

Increase worker counts only after checking GPU memory and latency under load.
Each STT web worker loads its own Whisper model.

The STT values above are balanced microphone defaults. If quiet speakers are
missed, lower `STT_SILENCE_RMS_THRESHOLD` slightly. If silence produces
hallucinated phrases, raise it slightly or make `STT_NO_SPEECH_PROB_THRESHOLD`
more strict.

If you deploy with Docker Compose, pass `ALLOWED_ORIGINS` to the `polytalk`
service environment or add it to your own production override file. The default
application fallback only allows localhost development origins.

## TTS Voices

The TTS image expects voice files in `tts/voices`. The directory is tracked with
`.gitkeep`, but voice model files are intentionally ignored.

Build the TTS image and download default voices:

```bash
docker compose build tts
./tts/setup-voices.sh
```

The script uses `polytalk-tts:latest` by default. Override it if needed:

```bash
TTS_IMAGE=custom-polytalk-tts:latest ./tts/setup-voices.sh
```

## Start PolyTalk

CPU/default:

```bash
docker compose up -d --build
```

GPU STT:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Check containers:

```bash
docker compose ps
docker compose logs -f polytalk
```

Check app health:

```bash
curl http://127.0.0.1:9000/api/health
```

The base `docker-compose.yml` includes source-code bind mounts to keep local
development convenient. For immutable production images, use your own Compose
override or deployment manifest that removes the `./app`, `./stt/app`,
`./tts/wsgi.py`, and `./config` bind mounts after the image and configuration
are baked or mounted from a controlled release location.

## Translation Provider

PolyTalk talks to translation through an OpenAI-compatible chat completions API.
Keep the translation service outside this repo if it has model-specific GPU,
token, or network requirements.

Example vLLM-style values:

```env
TRANSLATION_BASE_URL=http://polytalk-vllm:8000
TRANSLATION_MODEL=polytalk-translation
TRANSLATION_MAX_TOKENS=240
```

Model notes:

- Keep `TRANSLATION_MAX_TOKENS` bounded for live streaming. `240` gives
  Indic-script targets more room while still limiting runaway generations.
- `TRANSLATION_TYPED_TEXT_MAX_TOKENS` applies only to complete typed-text HTTP
  translations. It does not change live, conversation, or shared-tab translation.
- Qwen3 models may need Qwen-specific thinking/reasoning flags on the vLLM side.
- TranslateGemma/Gemma 3 models require `bfloat16` and must not use Qwen
  reasoning parser flags.
- Keep model-specific vLLM/Hugging Face tokens out of this repository.

## Nginx Reverse Proxy

Install nginx and certbot using your distribution packages, then create a site
similar to this:

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
}

server {
    listen 80;
    server_name polytalk.example.com;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name polytalk.example.com;

    # Certbot can manage these paths automatically after issuance.
    ssl_certificate /etc/letsencrypt/live/polytalk.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/polytalk.example.com/privkey.pem;

    client_max_body_size 250m;

    location ~ /\.(?!well-known(?:/|$)) {
        deny all;
        access_log off;
        log_not_found off;
    }

    location ~* ^/(?:phpinfo|info|php|i|pinfo|test|pi|p|debug|server-status|server-info)\.php(?:$|\?) {
        deny all;
        access_log off;
        log_not_found off;
    }

    location = /api/ws/translate {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Live translation sessions can run for a long time.
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_connect_timeout 60s;
        proxy_buffering off;
    }

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_connect_timeout 60s;
        proxy_buffering off;
    }
}
```

Validate and reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

Issue a certificate:

```bash
sudo certbot --nginx -d polytalk.example.com
```

Verify:

```bash
curl https://polytalk.example.com/api/health
curl -o /dev/null -s -w "%{http_code}\n" https://polytalk.example.com/.env
```

The `.env` check should return `403`.

Also verify WebSocket traffic from the browser dev tools while starting a live
translation. A common reverse-proxy failure mode is a healthy `/api/health`
response but a dropped `/api/ws/translate` connection because upgrade headers or
timeouts are missing.

## Operational Checks

Use these commands after deployment or a restart:

```bash
docker compose ps
docker compose logs --tail 100 polytalk
docker compose logs --tail 100 stt
docker compose logs --tail 100 tts
curl http://127.0.0.1:9000/api/health
```

For GPU deployments:

```bash
nvidia-smi
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec stt nvidia-smi
```

Set `LOG_LEVEL=DEBUG` temporarily when diagnosing latency. Debug logs include
STT queue wait, STT inference time, emit delay, ASR-to-translation queue wait,
translation request time, and TTS queue/duration.

Return to `LOG_LEVEL=INFO` after diagnosis. Debug logs can contain timing,
language, and transcript-adjacent metadata that is too noisy for long-term
production retention.

## Benchmarking and Demo Mode

Use [Benchmarking](benchmarking.md) to run STT, translation, TTS, and pipeline
checks. Benchmarking can consume GPU and translation-provider capacity, so run
it against staging where possible.

If you must benchmark a shared demo system:

- Announce the test window to users.
- Watch `docker compose logs -f stt polytalk` for STT backpressure and provider
  latency.
- Compare results with a known-good JSON output from the same audio fixture.
- Stop tests if live users report degraded microphone or tab-audio behavior.

## Backups and Data Handling

- `.env` contains secrets and must not be committed.
- `media/output` contains generated speech audio. Treat it as user data.
- Transcripts and translations can be sensitive. Control log retention.
- TTS voice files can be re-downloaded, but deployments should still preserve
  `tts/voices` or run `tts/setup-voices.sh` during provisioning.
- Consider a scheduled cleanup job for `media/output` if generated speech does
  not need long-term retention.

Example cleanup policy for files older than seven days:

```bash
find /path/to/polytalk/media/output -type f -mtime +7 -delete
```

## Troubleshooting

- Browser microphone is unavailable: use HTTPS and confirm permission prompts.
- WebSocket disconnects: confirm nginx upgrade headers and long proxy timeouts.
- STT is slow: use CUDA, a smaller model, or lower `STT_STREAM_CHUNK_SECONDS`
  only after checking accuracy.
- STT CUDA fails: verify NVIDIA Container Toolkit and the GPU override compose
  file.
- Translation times out: check provider logs, keep `TRANSLATION_MAX_TOKENS`
  low, and verify model-specific serving flags.
- Missing TTS voice: confirm the voice `.onnx` and `.onnx.json` files exist
  under `tts/voices`.
- CORS errors in the browser: set `ALLOWED_ORIGINS` to the exact HTTPS origin
  and ensure that variable is passed to the running `polytalk` container.
- Microphone quality is worse than tab audio: tune the STT silence and
  no-speech thresholds first. Tab audio is clean digital input; microphones
  include room noise, gain changes, and echo cancellation artifacts.
