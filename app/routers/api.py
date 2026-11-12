# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
API router for PolyTalk application.

Provides REST endpoints for audio translation.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field, field_validator

from ..pipeline_modes import (
    PIPELINE_MODE_DEFAULT,
    normalize_pipeline_mode,
    pipeline_mode_capabilities,
)
from ..services.audio_vad import VAD_RESET, VadMode
from ..schemas.translation import (
    TextSpeechRequest,
    TextTranslationRequest,
    TextTranslationResponse,
)
from ..services.pipeline_service import TranslationPipelineService
from ..services.visual_context_service import VisualContextService
from ..utils.config import get_custom_instruction_max_chars
from ..utils.audio import estimate_audio_duration
from ..utils.logger import get_logger
from ..utils.sanitize import normalize_instruction
from ..utils.translation_buffer import clamp_translation_buffer_seconds
from ..version import __version__

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["api"])

pipeline_service: Optional[TranslationPipelineService] = None
visual_context_service: Optional[VisualContextService] = None


class TextSpeakRequest(BaseModel):
    """One bounded selected-text chunk to translate and synthesize."""

    text: str = Field(min_length=1, max_length=500)
    source_language: str = Field(
        default="auto", pattern=r"^(?:auto|[a-z]{2,3}(?:_[A-Z]{2})?)$"
    )
    target_language: str = Field(pattern=r"^[a-z]{2,3}(?:_[A-Z]{2})?$")
    sequence: int = Field(default=0, ge=0, le=10000)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        """Normalize whitespace without retaining or logging the text."""
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized


def sanitize_custom_instruction(value: str | None) -> str:
    """Normalize and bound user-provided translation guidance."""
    return normalize_instruction(value, get_custom_instruction_max_chars())


def replace_queue_item(queue: asyncio.Queue, item) -> None:
    """Replace any pending item in a maxsize=1 queue (latest-wins).

    asyncio.Queue.put_nowait() raises QueueFull on a full queue instead of
    overwriting, so the pending item must be removed first.

    The check-then-get-then-put sequence is not atomic — another task could
    theoretically add an item between get_nowait() and put_nowait(). This is
    safe in practice because the app runs on a single-threaded asyncio event
    loop with no ``await`` between these calls, so no other task can run during
    this window.
    """
    if not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(item)


def get_pipeline_service() -> TranslationPipelineService:
    """
    Get or create the translation pipeline service singleton.

    Returns:
        TranslationPipelineService instance
    """
    global pipeline_service
    if pipeline_service is None:
        pipeline_service = TranslationPipelineService()
    return pipeline_service


async def close_pipeline_service() -> None:
    """Close and reset the shared translation pipeline."""
    global pipeline_service
    if pipeline_service is None:
        return

    await pipeline_service.close()
    pipeline_service = None


def get_visual_context_service() -> VisualContextService:
    """Get or create the visual context service singleton."""
    global visual_context_service
    if visual_context_service is None:
        visual_context_service = VisualContextService()
    return visual_context_service


async def close_visual_context_service() -> None:
    """Close and reset the visual context service singleton."""
    global visual_context_service
    if visual_context_service is None:
        return

    await visual_context_service.close()
    visual_context_service = None


def should_start_visual_context_request(
    image_data_url: str, in_flight: bool, ready: bool
) -> bool:
    """Return whether a visual context request should be accepted."""
    return bool(image_data_url) and not (in_flight or ready)


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "version": __version__,
        "service": "PolyTalk API",
    }


@router.post("/text/speak")
async def text_speak(
    payload: TextSpeakRequest,
    translation_routing: str = Header(
        default="default", alias="X-PolyTalk-Translation-Routing"
    ),
) -> Response:
    """Translate one selected-text chunk and return transient speech audio."""
    if translation_routing not in {"default", "custom"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid translation routing mode",
        )
    result = await get_pipeline_service().translate_text_to_speech(
        payload.text,
        payload.source_language,
        payload.target_language,
        custom_translation_routing=translation_routing == "custom",
    )
    if not result.success:
        logger.warning("Selected-text translation or speech synthesis failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Text translation or speech synthesis failed",
        )

    headers = {
        "Cache-Control": "no-store",
        "X-PolyTalk-Sequence": str(payload.sequence),
    }
    if result.duration is not None:
        headers["X-PolyTalk-Audio-Duration"] = str(result.duration)
    return Response(
        content=result.audio,
        media_type=result.media_type,
        headers=headers,
    )


def _normalized_text(value: str) -> str:
    """Trim outer whitespace while preserving the user's Unicode content."""
    text = value.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text must not be empty",
        )
    return text


@router.post("/text/translations", response_model=TextTranslationResponse)
async def translate_text(
    request: TextTranslationRequest,
    translation_routing: str | None = Header(
        default=None, alias="X-PolyTalk-Translation-Routing"
    ),
) -> TextTranslationResponse:
    """Translate user-provided text without requiring an audio stream."""
    pipeline = get_pipeline_service()
    result = await pipeline.translation.translate(
        _normalized_text(request.text),
        request.source_language,
        request.target_language,
        custom_instruction=sanitize_custom_instruction(request.custom_instruction),
        custom_translation_routing=(translation_routing or "").lower() == "custom",
        preserve_source_structure=True,
    )
    if not result.success or not result.text.strip():
        logger.warning(
            "Text translation failed: source=%s target=%s",
            request.source_language,
            request.target_language,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Translation is currently unavailable",
        )
    return TextTranslationResponse(
        translated_text=result.text,
        source_language=result.source_language or request.source_language,
        target_language=result.target_language or request.target_language,
    )


def _audio_media_type(audio_path: Path) -> str:
    """Infer the response media type from a generated audio artifact."""
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
        ".aac": "audio/aac",
        ".m4a": "audio/mp4",
    }
    return media_types.get(audio_path.suffix.lower(), "application/octet-stream")


@router.get("/text/voices")
async def text_voices(language: str) -> dict:
    """List voices actually installed for the requested language."""
    if not language or len(language) > 12:
        raise HTTPException(422, "Invalid language")
    return {"voices": await get_pipeline_service().tts.available_text_voices(language)}


@router.post("/text/speech")
async def synthesize_text(request: TextSpeechRequest) -> Response:
    """Generate playable speech bytes for translated text."""
    pipeline = get_pipeline_service()
    if request.voice:
        result = await pipeline.tts.synthesize_selected_voice(
            _normalized_text(request.text), request.language, request.voice
        )
    else:
        result = await pipeline.tts.synthesize(
            _normalized_text(request.text), request.language
        )
    if not result.success or result.audio_path is None:
        logger.warning("Text speech generation failed: language=%s", request.language)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Speech generation is currently unavailable",
        )

    audio_path = result.audio_path
    try:
        audio_content = audio_path.read_bytes()
        if not audio_content:
            raise OSError("Generated speech file is empty")
    except OSError as exc:
        logger.warning(
            "Generated speech could not be read: language=%s", request.language
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Generated speech is unavailable",
        ) from exc
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Temporary text speech file cleanup failed", exc_info=True)

    headers = {
        "Cache-Control": "no-store",
        # Providers may omit duration metadata; estimate WAV output so the
        # Enterprise billing boundary can still charge generated speech.
        "X-Audio-Duration-Seconds": str(
            result.duration
            if result.duration is not None
            else estimate_audio_duration(audio_content)
        ),
    }
    return Response(
        content=audio_content,
        media_type=_audio_media_type(audio_path),
        headers=headers,
    )


@router.websocket("/ws/translate")
async def websocket_translate(
    websocket: WebSocket,
    source_language: str = "en",
    target_language: str = "gu",
    mode: str = "live",
    input_type: str = "microphone",
    custom_instruction: str | None = None,
    translation_buffer_seconds: float | None = None,
    pipeline_mode: str | None = None,
):
    """
    WebSocket endpoint for real-time audio translation streaming (2-thread pipeline).

    Args:
        websocket: WebSocket connection
        source_language: Source language code
        target_language: Target language code
        mode: Translation mode, either live or conversation.
        input_type: Browser audio source, either microphone or share.
        custom_instruction: Optional user-provided translation guidance.
        translation_buffer_seconds: Optional per-session translation context window.
        pipeline_mode: Result mode: transcript, translation, or translation_tts.
    """
    normalized_pipeline_mode = normalize_pipeline_mode(pipeline_mode)
    if pipeline_mode is not None and pipeline_mode != normalized_pipeline_mode:
        logger.warning("Invalid pipeline_mode value, using default: %s", pipeline_mode)
    enable_translation, enable_tts = pipeline_mode_capabilities(
        normalized_pipeline_mode
    )

    routing_mode = (
        websocket.headers.get("x-polytalk-translation-routing", "default")
        .strip()
        .lower()
    )
    custom_translation_routing = routing_mode == "custom"
    if routing_mode not in {"default", "custom"}:
        logger.warning(
            "Unknown translation routing mode; using legacy default provider: "
            f"mode={routing_mode!r}"
        )

    await websocket.accept()
    logger.info(
        "WebSocket connection established: %s -> %s mode=%s input_type=%s",
        source_language,
        target_language,
        mode,
        input_type,
    )

    if mode not in {"live", "conversation"}:
        await websocket.send_json(
            {"type": "error", "error": "mode must be live or conversation"}
        )
        await websocket.close()
        return
    if input_type not in {"microphone", "share"}:
        await websocket.send_json(
            {
                "type": "error",
                "error": "input_type must be microphone or share",
            }
        )
        await websocket.close()
        return

    pipeline = get_pipeline_service()
    audio_chunks = []
    client_disconnected = False
    pause_event = asyncio.Event()
    language_swap_queue = asyncio.Queue()
    visual_context_queue = asyncio.Queue(maxsize=1)
    custom_instruction_queue = asyncio.Queue(maxsize=1)
    glossary_terms_queue = asyncio.Queue(maxsize=1)
    translation_buffer_config_queue = asyncio.Queue(maxsize=1)
    stt_emit_interval_config_queue = asyncio.Queue(maxsize=1)
    pipeline_mode_queue = asyncio.Queue(maxsize=1)
    pipeline_mode_queue.put_nowait(normalized_pipeline_mode)
    visual_context_tasks: set[asyncio.Task] = set()
    visual_context_in_flight = False
    visual_context_ready = False

    connection_start = time.time()
    idle_timeout_seconds = 300

    async def send_pipeline_status(
        stage: str,
        status: str,
        message: str,
    ) -> None:
        """Send a compact connection/readiness update to the frontend."""
        await send_result(
            {
                "type": "pipeline_status",
                "stage": stage,
                "status": status,
                "message": message,
            }
        )

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        """Generate audio chunks from WebSocket messages."""
        nonlocal client_disconnected, connection_start, visual_context_in_flight
        is_paused = False
        try:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                    connection_start = time.time()
                except asyncio.TimeoutError:
                    idle_time = time.time() - connection_start
                    if idle_time > idle_timeout_seconds:
                        logger.warning(
                            f"Connection idle for {idle_time:.1f}s, closing connection"
                        )
                        client_disconnected = True
                        return
                    logger.debug("Audio generator timeout, continuing...")
                    continue

                if "bytes" in message:
                    if not is_paused:
                        audio_chunk = message["bytes"]
                        audio_chunks.append(audio_chunk)
                        yield audio_chunk
                    else:
                        logger.debug("Audio chunk received while paused, discarding")
                elif "text" in message:
                    data = json.loads(message["text"])
                    if data.get("type") == "end":
                        client_disconnected = True
                        logger.info("Client sent 'end' signal, stopping pipeline")
                        yield b"__END_SIGNAL__"
                        return
                    elif data.get("type") == "end_of_utterance":
                        logger.info("Client reported an acoustic pause")
                        yield b"__END_OF_UTTERANCE__"
                    elif data.get("type") == "pause":
                        is_paused = True
                        pause_event.set()
                        yield VAD_RESET
                        logger.info(
                            "Client sent 'pause' signal, stopping audio transmission"
                        )
                    elif data.get("type") == "resume":
                        is_paused = False
                        pause_event.clear()
                        logger.info(
                            "Client sent 'resume' signal, resuming audio transmission"
                        )
                    elif data.get("type") == "custom_instruction":
                        instruction = sanitize_custom_instruction(
                            data.get("custom_instruction")
                        )
                        while True:
                            try:
                                custom_instruction_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        custom_instruction_queue.put_nowait(instruction)
                        logger.info(
                            "Custom translation instruction updated: "
                            f"chars={len(instruction)}"
                        )
                    elif data.get("type") in ("glossary_terms", "glossary_words"):
                        source_key = (
                            "terms" if data["type"] == "glossary_terms" else "words"
                        )
                        items = data.get(source_key, [])
                        if not isinstance(items, list):
                            logger.warning(
                                "Glossary message: expected list, got %s",
                                type(items).__name__,
                            )
                            continue
                        items = [t for t in items if isinstance(t, str) and t.strip()]
                        replace_queue_item(glossary_terms_queue, items)
                        logger.info(
                            "Glossary %s updated: %d items",
                            source_key,
                            len(items),
                        )
                    elif data.get("type") == "translation_buffer_config":
                        seconds = clamp_translation_buffer_seconds(
                            data.get("translation_buffer_seconds")
                        )
                        if seconds is None:
                            logger.warning(
                                "Ignoring invalid translation buffer configuration"
                            )
                            continue
                        while True:
                            try:
                                translation_buffer_config_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        translation_buffer_config_queue.put_nowait(seconds)
                        while True:
                            try:
                                stt_emit_interval_config_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        stt_emit_interval_config_queue.put_nowait(seconds)
                        logger.info(
                            "Session translation/STT pacing updated: "
                            f"seconds={seconds:.1f}"
                        )
                    elif data.get("type") == "visual_context":
                        image_data_url = data.get("image_data_url") or ""
                        if should_start_visual_context_request(
                            image_data_url,
                            visual_context_in_flight,
                            visual_context_ready,
                        ):
                            visual_context_in_flight = True
                            task = asyncio.create_task(
                                summarize_visual_context(image_data_url)
                            )
                            visual_context_tasks.add(task)
                            task.add_done_callback(visual_context_tasks.discard)
                        elif image_data_url:
                            logger.debug("Ignoring duplicate visual context request")
                    elif data.get("type") == "pipeline_mode":
                        value = data.get("value", PIPELINE_MODE_DEFAULT)
                        while True:
                            try:
                                pipeline_mode_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        mode_value = normalize_pipeline_mode(value)
                        if value != mode_value:
                            logger.warning("Invalid pipeline_mode value: %r", value)
                        pipeline_mode_queue.put_nowait(mode_value)
                        logger.debug("Pipeline mode updated: %s", mode_value)
                        try:
                            await send_result(
                                {
                                    "type": "pipeline_mode",
                                    "value": mode_value,
                                }
                            )
                        except Exception:
                            logger.warning("Failed to send pipeline_mode confirmation")
                    elif data.get("type") == "swap_languages":
                        logger.info(
                            "Ignoring language swap request; runtime swap is disabled"
                        )
                        try:
                            await websocket.send_json(
                                {
                                    "type": "swap_disabled",
                                    "message": "Language swap is disabled for active sessions",
                                }
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to send swap disabled response: {e}"
                            )
        except WebSocketDisconnect:
            client_disconnected = True
            logger.info("Client disconnected")
        except GeneratorExit:
            logger.info("GeneratorExit received")
            raise
        except Exception as e:
            logger.error(f"Error receiving audio chunk: {e}")

    async def send_result(result: dict):
        """Send result to WebSocket client."""
        try:
            await websocket.send_json(result)
        except Exception as e:
            logger.debug(f"Error sending result: {e}")

    async def summarize_visual_context(image_data_url: str) -> None:
        """Summarize a shared tab/page screenshot without blocking audio receive."""
        nonlocal visual_context_in_flight, visual_context_ready
        try:
            await send_pipeline_status(
                "visual_context",
                "active",
                "Reading shared tab context",
            )
            service = get_visual_context_service()
            summary = await service.summarize_screenshot(
                image_data_url,
                source_language,
                target_language,
            )
            if not summary:
                await send_pipeline_status(
                    "visual_context",
                    "warning",
                    "Shared tab context unavailable",
                )
                return

            while True:
                try:
                    visual_context_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            visual_context_queue.put_nowait(summary)
            visual_context_ready = True
            logger.info(
                "Visual context summary received: chars=%d",
                len(summary),
            )
            await send_pipeline_status(
                "visual_context",
                "done",
                "Shared tab context ready",
            )
        except Exception as exc:
            logger.warning(f"Visual context service call failed: {exc}")
            await send_pipeline_status(
                "visual_context",
                "warning",
                "Shared tab context unavailable",
            )
        finally:
            visual_context_in_flight = False

    audio_gen = audio_generator()

    try:
        await send_pipeline_status(
            "server_connected",
            "done",
            "Server connected",
        )
        vad_mode = pipeline.vad.effective_mode(mode, input_type)
        if vad_mode in {VadMode.BOUNDARY, VadMode.ACTIVE}:
            await send_result(
                {
                    "type": "audio_segmentation",
                    "mode": vad_mode.value,
                    "input_type": input_type,
                    "continuous_audio": True,
                }
            )
        await send_pipeline_status(
            "pipeline_warming",
            "active",
            "Preparing translation pipeline",
        )
        try:
            await pipeline.warm_connections()
            await send_pipeline_status(
                "pipeline_ready",
                "done",
                "Pipeline ready",
            )
        except Exception as warm_error:
            logger.warning(f"Pipeline warm-up failed: {warm_error}")
            await send_pipeline_status(
                "pipeline_ready",
                "warning",
                "Pipeline ready with warnings",
            )

        async for result in pipeline.process_streaming(
            audio_gen,
            source_language,
            target_language,
            mode=mode,
            input_type=input_type,
            pause_event=pause_event,
            language_swap_queue=language_swap_queue,
            visual_context_queue=visual_context_queue,
            custom_instruction=sanitize_custom_instruction(custom_instruction),
            custom_instruction_queue=custom_instruction_queue,
            translation_buffer_seconds=clamp_translation_buffer_seconds(
                translation_buffer_seconds
            ),
            translation_buffer_config_queue=translation_buffer_config_queue,
            stt_emit_interval_config_queue=stt_emit_interval_config_queue,
            custom_translation_routing=custom_translation_routing,
            tts_enabled=enable_tts,
            translation_enabled=enable_translation,
            pipeline_mode_queue=pipeline_mode_queue,
            glossary_terms_queue=glossary_terms_queue,
        ):
            if client_disconnected:
                logger.info("Client disconnected, stopping pipeline")
                break

            try:
                await send_result(result)
            except Exception:
                if result.get("type") != "complete":
                    logger.info("Client disconnected during streaming")
                    break

            if result.get("type") == "complete":
                logger.info("Translation pipeline complete")
                break

    except Exception as e:
        logger.error(f"WebSocket translation error: {e}")
        if not client_disconnected:
            try:
                await send_result({"type": "error", "error": str(e)})
            except Exception:
                pass

    finally:
        connection_duration = time.time() - connection_start
        logger.info(f"WebSocket connection closed after {connection_duration:.2f}s")

        try:
            await audio_gen.aclose()
        except Exception as e:
            logger.error(f"Error closing audio generator: {e}")

        for task in visual_context_tasks:
            task.cancel()
        if visual_context_tasks:
            await asyncio.gather(*visual_context_tasks, return_exceptions=True)

        if not client_disconnected:
            try:
                await websocket.close()
            except Exception:
                pass
