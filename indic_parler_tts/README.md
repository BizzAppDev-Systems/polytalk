# Indic Parler TTS service

This service wraps `ai4bharat/indic-parler-tts` behind a small WAV HTTP API.
The implementation and gated model assets are fetched during the Docker image
build. Runtime inference is offline and does not depend on host model paths.

Before building, accept the model terms on Hugging Face and set a read token as
`HF_TOKEN` in the ignored `.env` file or export it in the build shell. Compose
passes it to the build as a BuildKit secret:

```bash
HF_TOKEN=hf_read_token_after_accepting_the_model_terms
docker compose --progress=plain build indic-parler-tts
docker compose up -d --no-build indic-parler-tts polytalk
```

The model is approximately 3.75 GB before image/runtime overhead. CUDA is
strongly recommended for live translation; CPU mode is intended for functional
testing and is selected automatically when CUDA is unavailable.

Inference is intentionally serialized with a process-local lock because the
model is not used concurrently. Admission is also bounded by
`INDIC_PARLER_MAX_IN_FLIGHT_REQUESTS` (default `1`); requests arriving while all
slots are occupied receive HTTP 503 instead of accumulating in an unbounded
queue. The service runs one Uvicorn worker, and FastAPI executes the synchronous
endpoint in its thread pool, so deployments should scale with additional
service replicas rather than increasing in-process concurrency.

```bash
curl -o marathi.wav http://127.0.0.1:7790/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"तुम्ही कसे आहात?","lang":"mr","gender":"female","pace":"moderate"}'
```
