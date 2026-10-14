# Specialized STT workers

This build context creates isolated WebSocket-compatible workers for
AI4Bharat IndicConformer, NVIDIA Parakeet TDT v3, and SenseVoiceSmall.
Docker Compose supplies the provider and model build arguments. Models are
downloaded during docker build and stored in the image cache.

The IndicConformer multilingual checkpoint may require Hugging Face access.
Set `HF_TOKEN` in the ignored `.env` file or export it before building; Compose
exposes it to BuildKit as the `hf_token` build secret. Do not put tokens in Docker build arguments.


## CPU and GPU deployment

The base `docker-compose.yml` runs specialized STT workers on CPU for local
functional testing. Staging and production should use the same GPU override as
Faster Whisper:

```bash
# Local CPU
HF_TOKEN=hf_read_token_after_accepting_model_terms
docker compose up -d

# Staging/production GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d
```

The GPU override grants all three workers access to the NVIDIA runtime and sets
`STT_DEVICE=cuda`; the base file defaults them to `cpu`. It also installs a
CUDA-enabled PyTorch wheel for Indic Parler and runs that service on CUDA.

## Finalized utterances

Specialized workers emit finalized utterances after a speech pause by default.
This prevents an offline model's changing cumulative hypotheses from sending
stale partial text into translation and TTS. Silence is controlled by
`STT_PAUSE_FLUSH_SECONDS` and `STT_SILENCE_RMS_THRESHOLD`.

For continuous tab audio, all STT backends use `STT_EMIT_INTERVAL_SECONDS` as
the default pacing interval. The browser's Advanced Translation Pacing value is
sent when a session starts and whenever the user changes it. Specialized workers
finalize a bounded audio window at that interval; Faster Whisper uses it for
transcript emission batching. Set `SPECIALIZED_STT_EMIT_PARTIALS=true` only
when provisional live transcripts are explicitly desired.
