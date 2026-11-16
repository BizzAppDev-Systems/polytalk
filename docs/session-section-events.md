# Section events (T12368)

Streaming clients may consume additive `section` events. Existing cumulative
transcription and translation messages remain available for older clients.

Each pipeline run has a UUID prefix. A section ID appends the committed block
sequence (live/tab) or turn ID (conversation). IDs remain stable through translation
and TTS. A pending section contains the exact translation input, source/target
languages and kind (`live` or `conversation`). A later event with the same ID
contains status `completed` and `translated_text`, or status `failed`. A source-only
block has status `unavailable` because translation was not requested.

TTS events carry `section_id`; consumers must correlate on this field rather than
assuming the speech sequence equals the translation sequence. Translation-only
blocks and speech failures can make those sequences differ. Future multi-part
speech must add `chunk_order` starting at zero for each section.

Translation commits source before speech is generated. Live completion waits for
the speech worker's sentinel, ensuring the final audio event is delivered. Buffered
blocks that fail translation remain failed sections and are consumed; they are not
silently duplicated into a later buffered block. Transcript-only sections preserve
short deltas and commit at sentence/final boundaries. Translation-enabled streams
also retain short deltas in the buffer: existing sentence/pause/cadence thresholds
control translation, so words such as "I" are not discarded. Final flushes retain
short utterances. An unexpected speech-worker failure emits an error and closes
the result stream; cancellation does not enqueue into an abandoned result queue.

This is a transport contract. No Enterprise schema, authentication, UI or storage
implementation is imported into Core. See `tests/test_section_protocol.py` for
mapping and completion regressions.
