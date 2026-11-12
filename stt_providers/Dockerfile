# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ARG STT_PROVIDER
ARG STT_PROVIDER_MODEL
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    MODELSCOPE_CACHE=/models/modelscope \
    STT_PROVIDER=${STT_PROVIDER} \
    STT_PROVIDER_MODEL=${STT_PROVIDER_MODEL}

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs libsndfile1 python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r /app/requirements.txt \
    && case "${STT_PROVIDER}" in \
        indicconformer) pip3 install --no-cache-dir torch torchaudio transformers sentencepiece huggingface_hub onnxruntime-gpu==1.23.2 ;; \
        parakeet) pip3 install --no-cache-dir torch torchaudio "nemo_toolkit[asr]" ;; \
        sensevoice) pip3 install --no-cache-dir torch torchaudio funasr modelscope huggingface_hub ;; \
        *) echo "Unsupported STT_PROVIDER=${STT_PROVIDER}" >&2; exit 1 ;; \
    esac

COPY app/__init__.py app/download_model.py /app/app/
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
    python3 -m app.download_model

COPY app/main.py /app/app/main.py

EXPOSE 8000
CMD ["sh", "-c", "exec gunicorn app.main:app -w ${STT_WORKERS:-1} -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600"]
