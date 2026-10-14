"""HTTP service for the AI4Bharat Indic Parler-TTS model."""

import io
import os
import threading
from contextlib import asynccontextmanager

import librosa
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from audio import normalize_generated_audio


MODEL_PATH = os.getenv("INDIC_PARLER_MODEL_PATH", "/models/indic-parler-tts")
DESCRIPTION_TOKENIZER_PATH = os.getenv(
    "INDIC_PARLER_DESCRIPTION_TOKENIZER_PATH", "/models/description-tokenizer"
)
REQUESTED_DEVICE = os.getenv("INDIC_PARLER_DEVICE", "auto").lower()
PRELOAD_MODEL = os.getenv("INDIC_PARLER_PRELOAD_MODEL", "true").lower() == "true"
MAX_TEXT_CHARS = int(os.getenv("INDIC_PARLER_MAX_TEXT_CHARS", "1000"))
MAX_IN_FLIGHT_REQUESTS = max(
    1, int(os.getenv("INDIC_PARLER_MAX_IN_FLIGHT_REQUESTS", "1"))
)

LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

# Recommended named speakers from the model card. Languages without a listed
# speaker use a descriptive gender prompt rather than borrowing another accent.
RECOMMENDED_VOICES = {
    "as": {"male": "Amit", "female": "Sita"},
    "bn": {"male": "Arjun", "female": "Aditi"},
    "brx": {"male": "Bikram", "female": "Maya"},
    "doi": {"male": "Karan", "female": "Karan"},
    "en": {"male": "Thoma", "female": "Mary"},
    "gu": {"male": "Yash", "female": "Neha"},
    "hi": {"male": "Rohit", "female": "Divya"},
    "kn": {"male": "Suresh", "female": "Anu"},
    "ml": {"male": "Harish", "female": "Anjali"},
    "mni": {"male": "Ranjit", "female": "Laishram"},
    "mr": {"male": "Sanjay", "female": "Sunita"},
    "ne": {"male": "Amrita", "female": "Amrita"},
    "or": {"male": "Manas", "female": "Debjani"},
    "pa": {"male": "Divjot", "female": "Gurpreet"},
    "sa": {"male": "Aryan", "female": "Aryan"},
    "ta": {"male": "Jaya", "female": "Jaya"},
    "te": {"male": "Prakash", "female": "Lalitha"},
}

model = None
prompt_tokenizer = None
description_tokenizer = None
device = None
model_lock = threading.Lock()
generation_lock = threading.Lock()
request_slots = threading.BoundedSemaphore(MAX_IN_FLIGHT_REQUESTS)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    lang: str = "hi"
    gender: str = "female"
    voice: str = "auto"
    pace: str = "moderate"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    description: str | None = None


def _select_device() -> str:
    if REQUESTED_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if REQUESTED_DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("INDIC_PARLER_DEVICE=cuda but CUDA is unavailable")
    if REQUESTED_DEVICE not in {"cpu", "cuda"}:
        raise RuntimeError("INDIC_PARLER_DEVICE must be auto, cpu, or cuda")
    return REQUESTED_DEVICE


def _load_model() -> None:
    global model, prompt_tokenizer, description_tokenizer, device
    if model is not None:
        return
    with model_lock:
        if model is not None:
            return
        device = _select_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        prompt_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True
        )
        description_tokenizer = AutoTokenizer.from_pretrained(
            DESCRIPTION_TOKENIZER_PATH, local_files_only=True
        )


def _description(request: TTSRequest, lang: str) -> str:
    if request.description:
        return request.description.strip()
    gender = request.gender.lower()
    if gender not in {"male", "female"}:
        raise HTTPException(status_code=400, detail="gender must be male or female")
    voice = request.voice.strip()
    if not voice or voice.lower() == "auto":
        voice = RECOMMENDED_VOICES.get(lang, {}).get(gender, "")
    subject = f"{voice}'s voice" if voice else f"A {gender} speaker"
    language = LANGUAGE_NAMES.get(lang, lang)
    return (
        f"{subject} speaks {language} at a {request.pace.strip()} pace with a "
        "natural, slightly expressive tone and balanced pitch. The recording "
        "is very clear, close-sounding, and has no background noise or reverberation."
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if PRELOAD_MODEL:
        _load_model()
    yield


app = FastAPI(title="Indic Parler TTS", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "success",
        "service": "indic-parler-tts",
        "device": device or _select_device(),
        "model_loaded": model is not None,
    }


@app.post("/v1/tts")
def synthesize(request: TTSRequest) -> Response:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be blank")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds {MAX_TEXT_CHARS} characters",
        )
    lang = request.lang.replace("-", "_").split("_", 1)[0].lower()
    description = _description(request, lang)
    if not request_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Indic Parler TTS is busy; retry after the active request completes",
        )

    try:
        _load_model()

        with generation_lock, torch.inference_mode():
            description_inputs = description_tokenizer(
                description, return_tensors="pt"
            ).to(device)
            prompt_inputs = prompt_tokenizer(text, return_tensors="pt").to(device)
            generation = model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )
    finally:
        request_slots.release()

    try:
        audio = normalize_generated_audio(generation.detach().float().cpu().numpy())
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if abs(request.speed - 1.0) > 0.001:
        audio = librosa.effects.time_stretch(audio, rate=request.speed)
    output = io.BytesIO()
    sf.write(output, audio, model.config.sampling_rate, format="WAV", subtype="PCM_16")
    return Response(
        content=output.getvalue(),
        media_type="audio/wav",
        headers={
            "X-TTS-Voice": request.voice,
            "X-TTS-Language": lang,
            "X-TTS-Speed": str(request.speed),
        },
    )
