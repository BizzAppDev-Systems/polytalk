# STT Routing and Provider Expansion Plan

Status: planning only. This document does not change runtime behavior.

## Objective

Improve speech-to-text quality by routing each utterance to a model that is
strong for its language while preserving the existing PolyTalk WebSocket
contract and universal fallback behavior.

The first planned expansion is:

- Indian languages: AI4Bharat IndicConformerASR
- European languages: NVIDIA Parakeet TDT v3
- Mandarin, Cantonese, Japanese, and Korean: SenseVoiceSmall
- Other, unsupported, or uncertain languages: faster-whisper Whisper
  `large-v3`

The existing `faster-whisper` service remains the compatibility and fallback
path.

## Current architecture

The application sends audio to the STT service over WebSocket. The STT service
returns normalized transcription events that the pipeline forwards to
translation and TTS.

The pipeline should continue to depend on the normalized transcription result,
not on a model-specific SDK or output format.

## Routing rules

### Explicit-language mode

Normal live translation already receives `source_language` and
`target_language` explicitly. Only `source_language` affects STT routing:

```text
source language in IndicConformer set  -> IndicConformerASR
source language in Parakeet set        -> Parakeet TDT v3
source language in SenseVoice set      -> SenseVoiceSmall
otherwise                              -> faster-whisper large-v3
```

The target language continues to be handled by the translation and TTS
services.

Language sets must be configuration-driven rather than hard-coded in the
pipeline. This allows staging to enable or disable a model without changing
application code.

### Conversation mode

Conversation mode may not know the language before the first utterance. The
router should use a bounded detection/probing flow:

1. Buffer the initial audio for a short detection window.
2. Run the universal `large-v3` detector and, when capacity permits, a
   specialized candidate detector in parallel.
3. Select a backend only when the detected language and confidence meet the
   configured threshold.
4. Use the selected backend for the utterance or direction.
5. If detection disagrees, is below the threshold, or the backend fails, keep
   `large-v3` as the authoritative fallback.

The router must not concatenate transcripts produced by both probe models.
Probe output is discarded; only the selected backend emits user-visible text.
When switching backends, replay a small audio overlap and deduplicate the
result so words are not lost at the handoff.

For pause-flushed conversation turns, routing per turn is preferred over
routing once for the entire session. This supports code-switching and the
existing two-direction conversation flow.

## Backend integration shape

Introduce a provider-neutral STT adapter interface behind the current WebSocket
service:

```text
STT WebSocket API
        |
        v
  STT router / adapter
    |        |          |          |
 Indic    Parakeet  SenseVoice  faster-whisper
Conformer   TDT       Small       large-v3
```

Every backend must produce the same internal result fields:

- transcript text
- detected language
- partial/final state
- segment or word timestamps when available
- speech/no-speech result
- confidence and timing metrics when available
- normalized error information

The existing pipeline, translation buffer, pacing controls, and frontend event
schema should not need to know which STT backend produced a result.

## Runtime and deployment approach

IndicConformer and Parakeet are not Whisper models and cannot be loaded by
`faster-whisper`. They require separate model runtimes and adapters.

Preferred initial deployment:

- Keep faster-whisper as the existing STT service.
- Run IndicConformer, Parakeet, and SenseVoiceSmall as isolated backend workers or services.
- Put routing in a thin STT gateway, or add routing before the existing STT
  worker while preserving the public WebSocket endpoint.
- Keep model-specific Python, NeMo, PyTorch, CUDA, and tokenizer dependencies
  isolated where possible.

Avoid loading all large models into one GPU until memory usage is measured.
Separate services or lazy model loading may be required. Preloading improves
latency but increases GPU memory usage.

## Model-specific considerations

### IndicConformerASR

- Intended for the 22 scheduled Indian languages.
- Official usage is through the AI4Bharat NeMo fork/runtime.
- Use the multilingual model first for a simple router prototype.
- Compare multilingual and language-specific checkpoints for Hindi, Gujarati,
  Telugu, Tamil, and other high-priority languages.
- Confirm timestamp, partial-result, punctuation, and language-detection
  behavior before wiring it into live translation.

### NVIDIA Parakeet TDT v3

- Intended for 25 European languages.
- Use it for explicitly selected supported European source languages.
- Confirm that its output timing and incremental behavior are sufficient for
  PolyTalk's live emission and pacing model.
- Keep large-v3 as fallback for unsupported European variants, mixed-language
  audio, or low-confidence detection.

### SenseVoiceSmall

- Use the released checkpoint for Mandarin (`zh`), Cantonese (`yue`), Japanese (`ja`), and Korean (`ko`). English (`en`) is also supported, but does not need a dedicated route unless benchmarking shows a benefit.
- Treat the broader SenseVoice language claims separately from the released SenseVoiceSmall checkpoint; do not route unsupported languages to it.
- Its low-latency non-autoregressive inference and language-identification output make it a candidate for fast explicit-language transcription and conversation-mode probing.
- Evaluate its pseudo-streaming behavior separately because reduced-context streaming can trade accuracy for latency.
- Confirm its model/runtime license and output timestamps before production use.

### faster-whisper Whisper large-v3

- Universal fallback and initial reference implementation.
- Continue using it when the language is unknown, unsupported, mixed, or when
  a specialized backend is unavailable.
- Preserve the current VAD, silence, overlap, and pacing safeguards during the
  first routing rollout.

## Validation plan

Create a fixed evaluation set containing real or consented recordings for:

- English, German, French, Spanish, Italian, and Dutch
- Mandarin, Cantonese, Japanese, and Korean
- Hindi, Gujarati, Telugu, Tamil, and Malayalam
- noisy microphone audio
- browser/tab audio
- accents and code-switching
- short utterances and pause-flushed conversation turns

For every backend and language, record:

- word error rate or character error rate
- first-result latency
- real-time factor
- final-result latency
- GPU memory and CPU usage
- language-detection accuracy
- duplicate, dropped-word, and hallucination count
- timestamp quality where supported

The router should initially run in shadow mode for selected staging traffic:

- Keep large-v3 as the visible result.
- Run the candidate backend in the background.
- Compare outputs and timings without sending candidate text to translation or
  TTS.

Promote a language/backend pair only after it meets the large-v3 baseline for
latency and improves or maintains transcription quality.

## Rollout controls

Add configuration for:

- backend enablement
- supported language sets
- confidence threshold
- conversation probe duration
- probe timeout
- fallback backend
- shadow-mode logging
- per-backend model name and device

Every request should log the selected backend, requested language, detected
language, fallback reason, and inference timing without logging audio or
transcript content by default.

## Acceptance criteria

- Existing WebSocket clients work without protocol changes.
- Explicit-language sessions route to the configured specialized backend.
- Conversation mode falls back safely when detection is uncertain.
- No duplicate probe transcript reaches translation or TTS.
- A backend failure does not terminate the live session if large-v3 is
  available.
- Specialized routing does not increase end-to-end latency beyond the agreed
  staging baseline.
- Model and runtime licenses are documented before production deployment.

## Follow-up: additional TTS providers

After STT routing is covered, evaluate additional TTS backends using the same
provider-neutral pattern.

Topics to decide:

- Which languages and voices need additional coverage.
- Whether the provider supports streaming audio or only complete files.
- Voice selection and fallback rules per language.
- First-byte latency and browser playback behavior.
- Audio format, sample rate, and media URL requirements.
- GPU/CPU and model memory requirements.
- Commercial and redistribution licensing.
- Whether the provider needs a separate container or can use the existing TTS
  service contract.
