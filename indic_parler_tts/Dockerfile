FROM python:3.10-slim

ARG PARLER_TTS_REPOSITORY=https://github.com/huggingface/parler-tts.git
ARG PARLER_TTS_REF=main
ARG INDIC_PARLER_MODEL=ai4bharat/indic-parler-tts
ARG INDIC_PARLER_REVISION=main
ARG INDIC_PARLER_TORCH_VERSION=2.5.1
ARG INDIC_PARLER_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/.cache \
    INDIC_PARLER_MODEL_PATH=/models/indic-parler-tts \
    INDIC_PARLER_DESCRIPTION_TOKENIZER_PATH=/models/description-tokenizer \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Pin Torch before installing Parler so pip does not select an unrelated newer
# CUDA runtime. Use the PyTorch CUDA wheel index as an extra source
# for GPU deployments.
RUN pip install --no-cache-dir --extra-index-url "${INDIC_PARLER_TORCH_INDEX_URL}" \
        "torch==${INDIC_PARLER_TORCH_VERSION}" \
        "torchaudio==${INDIC_PARLER_TORCH_VERSION}"

# Fetch the inference implementation during the image build. PARLER_TTS_REF can
# be pinned to a release or commit in production.
RUN git clone --branch "${PARLER_TTS_REF}" --single-branch \
        "${PARLER_TTS_REPOSITORY}" /opt/parler-tts \
    && cd /opt/parler-tts \
    && git pull --ff-only \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir fastapi==0.116.1 uvicorn[standard]==0.35.0 \
        soundfile==0.13.1 huggingface-hub==0.34.4

COPY download_models.py /usr/local/bin/download_indic_parler_models.py

# Indic Parler is gated. The token is available only to this build step and is
# never copied into an image layer.
RUN --mount=type=secret,id=hf_token,required=true \
    python /usr/local/bin/download_indic_parler_models.py \
        --repo-id "${INDIC_PARLER_MODEL}" \
        --revision "${INDIC_PARLER_REVISION}" \
        --model-dir "${INDIC_PARLER_MODEL_PATH}" \
        --description-tokenizer-dir "${INDIC_PARLER_DESCRIPTION_TOKENIZER_PATH}" \
        --token-file /run/secrets/hf_token \
    && rm -rf /models/.cache

COPY app.py audio.py /app/

EXPOSE 7790

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7790", "--workers", "1"]
