"""Section identity survives buffered translation and asynchronous speech."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.pipeline_service import TranslationPipelineService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "translation_enabled,tts_enabled", [(True, True), (True, False), (False, False)]
)
async def test_live_section_links_exact_source_and_audio(
    translation_enabled, tts_enabled
):
    pipeline = TranslationPipelineService(warm_connections=False)

    async def audio():
        yield b"audio"
        yield b"__END_SIGNAL__"

    async def transcript(*args, **kwargs):
        yield MagicMock(
            success=True,
            text="Hello world.",
            language="en",
            is_partial=False,
            metrics={},
        )

    translate = AsyncMock(
        return_value=MagicMock(success=True, text="Hallo Welt.", metrics={})
    )
    speech = AsyncMock(
        return_value=MagicMock(
            success=True, audio_url="/media/output/test.wav", duration=1
        )
    )
    with (
        patch.object(pipeline.whisper, "stream_transcribe", transcript),
        patch.object(pipeline.translation, "translate", translate),
        patch.object(pipeline, "_synthesize", speech),
    ):
        events = [
            event
            async for event in pipeline.process_streaming(
                audio(),
                "en",
                "de",
                translation_enabled=translation_enabled,
                tts_enabled=tts_enabled,
            )
        ]
    sections = [event for event in events if event["type"] == "section"]
    assert sections
    assert sections[0]["source_text"] == "Hello world."
    assert len({event["section_id"] for event in sections}) == 1
    if translation_enabled:
        assert sections[-1]["status"] == "completed"
        assert sections[-1]["translated_text"] == "Hallo Welt."
    else:
        assert sections[-1]["status"] == "unavailable"
        translate.assert_not_called()
    audio_events = [event for event in events if event["type"] == "tts"]
    if tts_enabled:
        assert audio_events[0]["section_id"] == sections[0]["section_id"]
        assert speech.await_args.args[1] == "de"
    else:
        assert not audio_events


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_failed_translation_preserves_source_section(partial):
    pipeline = TranslationPipelineService(warm_connections=False)

    async def audio():
        yield b"audio"
        yield b"__END_SIGNAL__"

    async def transcript(*args, **kwargs):
        yield MagicMock(
            success=True,
            text="Keep this source.",
            language="en",
            is_partial=partial,
            metrics={},
        )

    with (
        patch.object(pipeline.whisper, "stream_transcribe", transcript),
        patch.object(
            pipeline.translation,
            "translate",
            AsyncMock(
                return_value=MagicMock(success=False, text="", error="unavailable")
            ),
        ),
    ):
        events = [
            event
            async for event in pipeline.process_streaming(
                audio(), "en", "de", tts_enabled=False
            )
        ]
    failed = [
        event
        for event in events
        if event["type"] == "section" and event["status"] == "failed"
    ]
    assert failed and failed[0]["source_text"] == "Keep this source."
    assert all("translated_text" not in event for event in failed)


@pytest.mark.asyncio
async def test_transcript_only_sections_keep_short_deltas_without_cumulative_duplicates():
    pipeline = TranslationPipelineService(warm_connections=False)

    async def audio():
        yield b"audio"

    async def transcript(*_args, **_kwargs):
        for value, partial in [
            ("Hi", True),
            ("Hi there.", True),
            ("Hi there. Goodbye", False),
        ]:
            yield MagicMock(
                success=True, text=value, language="en", is_partial=partial, metrics={}
            )

    with patch.object(pipeline.whisper, "stream_transcribe", transcript):
        events = [
            event
            async for event in pipeline.process_streaming(
                audio(), "en", "de", translation_enabled=False, tts_enabled=False
            )
        ]
    sections = [event for event in events if event["type"] == "section"]
    assert [event["source_text"] for event in sections] == ["Hi there.", "Goodbye"]
    assert len({event["section_id"] for event in sections}) == 2


@pytest.mark.asyncio
async def test_conversation_sections_keep_turn_direction_and_audio_identity():
    pipeline = TranslationPipelineService(warm_connections=False)

    async def audio():
        yield b"audio"

    async def transcript(*_args, **_kwargs):
        for value, language in [("Hello there.", "en"), ("Guten Tag.", "de")]:
            yield MagicMock(
                success=True,
                text=value,
                language=language,
                is_partial=False,
                metrics={"conversation_utterance": True},
            )

    with (
        patch.object(pipeline.whisper, "stream_transcribe_conversation", transcript),
        patch.object(
            pipeline.translation,
            "translate",
            AsyncMock(
                return_value=MagicMock(success=True, text="Translation", metrics={})
            ),
        ),
        patch.object(
            pipeline,
            "_synthesize",
            AsyncMock(
                return_value=MagicMock(success=True, audio_url="/media/output/turn.wav")
            ),
        ),
    ):
        events = [
            event
            async for event in pipeline.process_streaming(
                audio(), "en", "de", mode="conversation"
            )
        ]
    sections = [
        event
        for event in events
        if event["type"] == "section" and event["status"] == "completed"
    ]
    audio_events = [event for event in events if event["type"] == "tts"]
    assert [
        (event["source_language"], event["target_language"]) for event in sections
    ] == [("en", "de"), ("de", "en")]
    assert [event["section_id"] for event in audio_events] == [
        event["section_id"] for event in sections
    ]


@pytest.mark.asyncio
async def test_tts_outer_failure_terminates_stream():
    """An unexpected TTS queue-item failure cannot leave the consumer waiting."""
    from app.services.pipeline_service import TranslatedSentence

    pipeline = TranslationPipelineService(warm_connections=False)

    async def audio():
        yield b"audio"

    async def transcript(*args, **kwargs):
        yield MagicMock(
            success=True,
            text="Hello world.",
            language="en",
            is_partial=False,
            metrics={},
        )

    def broken_item(*args, **kwargs):
        item = TranslatedSentence(*args, **kwargs)
        item.queued_at = "invalid timestamp"
        return item

    async def collect():
        return [
            event async for event in pipeline.process_streaming(audio(), "en", "de")
        ]

    with (
        patch.object(pipeline.whisper, "stream_transcribe", transcript),
        patch.object(
            pipeline.translation,
            "translate",
            AsyncMock(
                return_value=MagicMock(success=True, text="Hallo Welt.", metrics={})
            ),
        ),
        patch(
            "app.services.pipeline_service.TranslatedSentence", side_effect=broken_item
        ),
    ):
        events = await asyncio.wait_for(collect(), timeout=2)
    assert any(event["type"] == "error" for event in events)
    assert any(
        event["type"] == "section" and event["status"] == "completed"
        for event in events
    )


@pytest.mark.asyncio
async def test_translation_accumulates_short_deltas_without_extra_speech():
    """Short source words remain in the sentence rather than being discarded."""
    pipeline = TranslationPipelineService(warm_connections=False)

    async def audio():
        yield b"audio"

    async def transcript(*args, **kwargs):
        for text, partial in [("I", True), ("I am", True), ("I am here.", False)]:
            yield MagicMock(
                success=True, text=text, language="en", is_partial=partial, metrics={}
            )

    translate = AsyncMock(
        return_value=MagicMock(success=True, text="Ich bin hier.", metrics={})
    )
    speech = AsyncMock(
        return_value=MagicMock(
            success=True, audio_url="/media/output/one.wav", duration=1
        )
    )
    with (
        patch.object(pipeline.whisper, "stream_transcribe", transcript),
        patch.object(pipeline.translation, "translate", translate),
        patch.object(pipeline, "_synthesize", speech),
    ):
        events = [
            event async for event in pipeline.process_streaming(audio(), "en", "de")
        ]
    assert translate.await_count == 1
    assert translate.await_args.args[0] == "I am here."
    assert speech.await_count == 1
    assert (
        len(
            [e for e in events if e["type"] == "section" and e["status"] == "completed"]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_full_speech_queue_does_not_block_cancellation():
    """Disconnect reaches the speech timeout even with a full pending queue."""
    pipeline = TranslationPipelineService(warm_connections=False)
    full = asyncio.Event()
    synth_cancelled = asyncio.Event()
    queue_class = asyncio.Queue

    class ObservedQueue(queue_class):
        async def put(self, item):
            from app.services.pipeline_service import TranslatedSentence

            if isinstance(item, TranslatedSentence) and self.full():
                full.set()
            return await super().put(item)

    async def audio():
        yield b"audio"

    async def transcript(*args, **kwargs):
        text = ""
        for number in range(110):
            text += f"Sentence number {number}. "
            yield MagicMock(
                success=True,
                text=text.strip(),
                language="en",
                is_partial=True,
                metrics={},
            )

    async def synthesize(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            synth_cancelled.set()

    async def collect():
        async for _ in pipeline.process_streaming(audio(), "en", "de"):
            pass

    with (
        patch("app.services.pipeline_service.asyncio.Queue", ObservedQueue),
        patch.object(pipeline.whisper, "stream_transcribe", transcript),
        patch.object(
            pipeline.translation,
            "translate",
            AsyncMock(
                return_value=MagicMock(
                    success=True, text="Translated sentence.", metrics={}
                )
            ),
        ),
        patch.object(pipeline, "_synthesize", synthesize),
    ):
        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(full.wait(), timeout=3)
            # Let the consumer drain already-produced events before disconnecting.
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=7)
            assert synth_cancelled.is_set()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["context", "speech_queue"])
async def test_post_translation_error_does_not_reverse_completed_section(failure_point):
    """A downstream failure must not produce completed then failed for one ID."""
    pipeline = TranslationPipelineService(warm_connections=False)

    async def audio():
        yield b"audio"

    async def transcript(*args, **kwargs):
        yield MagicMock(
            success=True,
            text="Keep completed text",
            language="en",
            is_partial=False,
            metrics={},
        )

    failing = patch(
        "app.services.pipeline_service.TranslationContextWindow.remember"
        if failure_point == "context"
        else "app.services.pipeline_service.TranslatedSentence",
        side_effect=RuntimeError("downstream failure"),
    )
    with (
        patch.object(pipeline.whisper, "stream_transcribe", transcript),
        patch.object(
            pipeline.translation,
            "translate",
            AsyncMock(
                return_value=MagicMock(success=True, text="Translation", metrics={})
            ),
        ),
        failing,
    ):
        events = [
            event async for event in pipeline.process_streaming(audio(), "en", "de")
        ]
    terminal = [
        event
        for event in events
        if event["type"] == "section" and event["status"] in {"completed", "failed"}
    ]
    assert len(terminal) == 1
    assert terminal[0]["status"] == "completed"
    assert terminal[0]["translated_text"] == "Translation"
