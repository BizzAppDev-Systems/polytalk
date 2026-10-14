# STT Routing and Provider Expansion Plan

Status: phase 1 implemented; phase 2 language detection remains planned.

## Objective

Improve speech-to-text quality by routing each utterance to a model that is
strong for its language while preserving the existing PolyTalk WebSocket
contract. Conversation routing must not assume that faster-whisper supports
either selected language, and fallback eligibility must be language-aware.

The first planned expansion is:

- Indian languages: AI4Bharat IndicConformerASR
- European languages: NVIDIA Parakeet TDT v3
- Mandarin, Cantonese, Japanese, and Korean: SenseVoiceSmall
- Other, unsupported, or uncertain languages: faster-whisper Whisper
  `large-v3`

The existing `faster-whisper` service remains a compatibility backend for
languages that explicitly allow it. It is not the unconditional conversation
fallback.

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

Conversation mode knows the selected language pair but not which language is
spoken in each turn. The current pipeline passes no explicit language to STT,
so the router selects the fallback provider before considering the two
candidate languages. Conversation routing must instead operate on complete,
pause-delimited utterances and resolve providers for both selected languages.

#### Phase 1: pair-aware routing and parallel transcription

Phase 1 contains two paths:

1. When both selected languages resolve to one provider, use that provider.
   Call it once when it can genuinely identify both languages. If it requires
   an explicit language hint, run two concurrent decodes against the same
   provider, one for each language hypothesis.
2. When the selected languages resolve to different providers, send identical
   utterance audio to both configured providers concurrently. Adjudicate the
   completed candidate results and expose only the winner to translation and
   TTS.

Each invocation represents a language hypothesis. `candidate_language` is the
language supplied to the provider, while `detected_language` is populated only
when the model independently identifies a language. An echoed request hint
must not be treated as detection evidence.

Use this decision order:

1. Discard failed, empty, no-speech, or below-threshold candidates.
2. If both candidates independently identify the same language, choose the
   successful output from the provider configured for that language.
3. If candidates identify different languages, choose the candidate with the
   highest normalized confidence when it exceeds the other by a configured
   margin.
4. If confidence is unavailable, low, or too close, choose the configured
   conversation default. Initially this is the original source language's
   configured provider.
5. If only one valid candidate remains, use it.
6. Use faster-whisper as a fallback only when the winning language explicitly
   permits it. Otherwise return an uncertain/error result rather than a likely
   incorrect transcript.

Raw confidence values from different model families are not directly
comparable. Normalize them using provider-specific calibration plus available
signals such as expected-script consistency, speech probability, transcript
quality, and hallucination or repetition penalties. Until calibration data is
available, treat confidence as pass/fail per provider and send close decisions
to the configured default.

Example with different providers:

```text
one pause-delimited utterance
        |---------------------> IndicConformer (Gujarati hypothesis) --|
        |---------------------> Parakeet (English hypothesis) ---------|
                                                                       v
                                                                 adjudicator
                                                                       |
                                                           one visible result
```

The router must never concatenate candidate transcripts or allow both results
to reach translation or TTS.

#### Phase 2: dedicated language detector

Add a spoken-language detector constrained to the two selected languages. A
high-confidence result routes the utterance to only the configured provider
for the detected language. A low-confidence or failed detection falls back to
the phase 1 parallel path:

```text
utterance -> pair-constrained language detector
              | high confidence -> one configured STT
              | low confidence  -> phase 1 parallel STT + adjudicator
```

Phase 2 is an optimization and accuracy improvement, not a replacement for the
phase 1 fallback path.

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
- candidate/requested language
- independently detected language, when supported
- partial/final state
- segment or word timestamps when available
- speech/no-speech result
- provider-native confidence and normalized confidence
- timing metrics
- normalized error information

A conversation candidate result should have a provider-neutral shape similar
to:

```json
{
  "provider": "indicconformer",
  "candidate_language": "gu",
  "detected_language": "gu",
  "text": "નમસ્તે",
  "provider_confidence": 0.84,
  "normalized_confidence": 0.88,
  "has_speech": true,
  "success": true
}
```

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
- Allow large-v3 fallback for unsupported European variants or mixed-language
  audio only when that language is explicitly fallback-eligible.

### SenseVoiceSmall

- Use the released checkpoint for Mandarin (`zh`), Cantonese (`yue`), Japanese (`ja`), and Korean (`ko`). English (`en`) is also supported, but does not need a dedicated route unless benchmarking shows a benefit.
- Treat the broader SenseVoice language claims separately from the released SenseVoiceSmall checkpoint; do not route unsupported languages to it.
- Its low-latency non-autoregressive inference and language-identification output make it a candidate for fast explicit-language transcription and conversation-mode probing.
- Evaluate its pseudo-streaming behavior separately because reduced-context streaming can trade accuracy for latency.
- Confirm its model/runtime license and output timestamps before production use.

### faster-whisper Whisper large-v3

- Compatibility fallback and initial reference implementation.
- Use it only when the selected or detected language is configured as
  supported by faster-whisper, or when an operator explicitly enables it as a
  fallback for that language.
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
- provider confidence distribution and calibration error
- pairwise winner accuracy for conversation utterances
- adjudicator default/tie rate
- percentage of utterances requiring two STT calls

Conversation evaluation must include pairs that map to the same provider and
pairs that map to different providers. It must verify that both candidates
receive byte-identical utterance audio and that exactly one result reaches
translation and TTS.

The router should initially run in shadow mode for selected staging traffic:

- Keep the configured default result visible.
- Run the parallel candidates and adjudicator in the background.
- Compare the proposed winner, outputs, and timings without sending shadow
  candidate text to translation or TTS.

Promote a language/backend pair or adjudication rule only after it meets the
configured baseline for latency and improves or maintains transcription
quality.

## Rollout controls

Add configuration for:

- backend enablement
- supported language sets
- confidence threshold
- confidence normalization/calibration per provider
- minimum winning-confidence margin
- conversation default language/provider
- language-aware fallback eligibility
- provider language-detection capability
- parallel-candidate timeout
- conversation probe duration
- probe timeout
- fallback backend
- shadow-mode logging
- per-backend model name and device

Every request should log the selected backend, requested language, detected
language, fallback reason, candidate outcomes, and inference timing without
logging audio or transcript content by default. Timeouts are logged separately
from provider errors so an external log pipeline can aggregate them when needed.

## Acceptance criteria

- Existing WebSocket clients work without protocol changes.
- Explicit-language sessions route to the configured specialized backend.
- Conversation mode resolves both selected languages before selecting a
  backend.
- A same-provider pair uses that provider and handles providers that require
  explicit language hints.
- A different-provider pair calls both candidates concurrently with identical
  utterance audio and emits exactly one adjudicated transcript.
- Agreement selects the provider configured for the agreed language;
  disagreement uses normalized confidence and then the configured default.
- An echoed language hint is never counted as independent language detection.
- No duplicate candidate transcript reaches translation or TTS.
- A backend failure uses the other valid candidate or a language-compatible
  fallback without terminating the session.
- faster-whisper is not used for a language unless configuration permits it.
- Specialized routing does not increase end-to-end latency beyond the agreed
  staging baseline.
- Phase 2 high-confidence detection calls one STT; uncertain detection falls
  back to the phase 1 parallel path.
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
