# STT Service (Speech-to-Text)

A fast, self-hosted speech-to-text service using faster-whisper.

## Features

- Real-time streaming transcription via WebSocket
- Raw 16 kHz mono int16 PCM input from the PolyTalk browser client
- CPU and GPU support
- Multiple Whisper models (small, small-v3, medium, large-v3)

## Quick Start

### Using Docker Compose (Recommended)

The STT service is integrated into the main PolyTalk Docker Compose stack:

```bash
# Start both STT and PolyTalk services
docker compose up -d
```

The STT service will be available at `http://localhost:8000` on the host and internally at `http://stt:8000` in Docker Compose.

### Standalone Docker

```bash
# Build and run
docker build -t polytalk-stt ./stt
docker run -p 8000:8000 polytalk-stt
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL` | `small` | Model to use (small, small-v3, medium, large-v3) |
| `STT_DEVICE` | `cpu` | Device to run on (cpu or cuda) |
| `STT_COMPUTE_TYPE` | `int8` | Compute type (int8 for CPU, float16 for CUDA) |
| `STT_WORKERS` | `1` | Number of web workers. Each worker loads its own Whisper model. |
| `STT_PRELOAD_MODEL` | `true` | Load the Whisper model during service startup instead of on first stream. |
| `STT_STREAM_CHUNK_SECONDS` | `3.0` | Audio window processed per streaming transcription pass. Lower values reduce latency but can reduce transcript stability. |
| `STT_CHUNK_OVERLAP_SECONDS` | `0.25` | Audio overlap between STT windows. Helps avoid missing words at chunk boundaries. |
| `STT_TRANSCRIBE_WORKERS` | `2` | Number of per-stream transcription workers. Use more than 1 only when the GPU has spare compute. |
| `STT_TRANSCRIBE_QUEUE_SIZE` | `8` | Max queued audio windows per stream before WebSocket receiving applies backpressure. |
| `STT_MODEL_WORKERS` | `2` | faster-whisper/CTranslate2 model workers for concurrent transcribe calls. Keep aligned with `STT_TRANSCRIBE_WORKERS`. |
| `STT_EMIT_MIN_CHARS` | `120` | Minimum new transcript text before emitting an update to PolyTalk. Increase this if live chunks are too small. |
| `STT_EMIT_INTERVAL_SECONDS` | `4.5` | Maximum time to hold pending transcript text before emitting it. |
| `STT_PAUSE_FLUSH_SECONDS` | `1.2` | Flush and emit the current speech window after this much trailing silence. Set `0` to disable pause flushing. |
| `STT_IDLE_FLUSH_SECONDS` | `3.0` | Flush and emit the current speech window if audio chunks stop arriving after detected speech. Set `0` to disable idle flushing. |
| `STT_LEADING_SILENCE_PREROLL_SECONDS` | `0.2` | Keep this much audio before first detected speech while discarding longer tab-share startup silence. |
| `STT_SILENCE_RMS_THRESHOLD` | `0.003` | Skip model inference for very quiet audio windows. Raise this if Whisper hallucinates during silence. |
| `STT_NO_SPEECH_PROB_THRESHOLD` | `0.50` | Drop segments classified as likely no-speech by faster-whisper. |
| `STT_LOG_PROB_THRESHOLD` | `-1.0` | Drop low-confidence faster-whisper segments. |
| `STT_CANDIDATE_LANGUAGE_CONFIDENCE` | `0.70` | In conversation mode, trust an in-pair auto-detection only at or above this probability; otherwise decode both selected languages. |
| `STT_MAX_CANDIDATE_DECODES` | `3` | Maximum explicit Whisper decodes for an ambiguous audio window. Normal conversation mode supplies only the selected source and target; the cap bounds malformed or future larger candidate lists. |
| `STT_CANDIDATE_SCRIPT_SCORE_WEIGHT` | `1.50` | Weight expected-script consistency when selecting between ambiguous conversation-language decodes. |
| `STT_MAX_CROSS_DELTA_WORD_REPEATS` | `6` | Stop appending the same leading word across transcript updates after this many existing repeats. |
| `STT_VAD_FILTER` | `true` | Enable faster-whisper VAD before decoding. |
| `STT_VAD_MIN_SILENCE_MS` | `500` | Silence duration used by VAD to split speech. Raise for fewer, larger speech regions. |
| `STT_VAD_SPEECH_PAD_MS` | `200` | Padding kept around detected speech. Raise if words are clipped near speech boundaries. |
| `STT_WORD_TIMESTAMPS` | `true` | Request word timestamps from faster-whisper. |
| `STT_CONDITION_ON_PREVIOUS_TEXT` | `false` | Reuse previous Whisper text as context. Keep disabled for lowest hallucination risk in streaming. |
| `STT_TEMPERATURE` | `0.0` | Decoding temperature. Keep `0.0` for deterministic streaming output. |
| `STT_INITIAL_PROMPT` | empty | Optional Whisper prompt for domain terms, names, and expected vocabulary. |
| `STT_SAMPLE_RATE` | `16000` | Input PCM sample rate expected from the browser client. |
| `STT_CHANNELS` | `1` | Input PCM channel count. |
| `STT_SAMPLE_WIDTH_BYTES` | `2` | Bytes per sample for int16 PCM. |
| `STORAGE_DIR` | `/tmp/stt` | Directory for temporary files |
| `MAX_UPLOAD_MB` | `200` | Maximum stream size in MB |

## API Endpoints

### Health Check

```bash
GET /health
```

### Streaming Transcription (WebSocket)

```bash
WS /v1/stream/transcriptions?language=en&task=transcribe
```

Send audio chunks as binary messages. Receive JSON responses:

```json
{
  "text": "transcribed text so far",
  "is_final": false,
  "language": "en",
  "has_speech": true
}
```

Parameters:
- `language`: Optional language code (e.g., "en", "hi", "es")
- `task`: "transcribe" or "translate"

## GPU Acceleration

For GPU support, set these environment variables:

```bash
STT_DEVICE=cuda
STT_COMPUTE_TYPE=float16
```

Run Docker Compose with the GPU override so the STT container receives the host NVIDIA driver:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build stt
```

Verify GPU visibility from inside the container:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec stt nvidia-smi
```

If `nvidia-smi` works on the host but fails inside the container, install/configure NVIDIA Container Toolkit or confirm the GPU override is being used.

## Model Selection

- **small**: Fast, good quality, ~50MB
- **small-v3**: Improved small model
- **medium**: Better quality, slower, ~800MB
- **large-v3**: Best quality, slowest, ~3GB

First run will download the model automatically.

## License

AGPL-3.0-or-later
