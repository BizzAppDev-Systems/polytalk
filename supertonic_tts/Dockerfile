FROM python:3.12-slim

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV XDG_CACHE_HOME=/data/cache
ENV HOME=/data/home

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gosu \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir 'supertonic[serve]==1.3.1'

RUN addgroup --system supertonic \
    && adduser --system --ingroup supertonic --home /data/home supertonic \
    && mkdir -p /data/cache /data/home /app \
    && chown -R supertonic:supertonic /data /app

COPY entrypoint.sh /app/entrypoint.sh
COPY text_filter.py sitecustomize.py /usr/local/lib/python3.12/site-packages/
RUN chmod +x /app/entrypoint.sh

EXPOSE 7788

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["supertonic", "serve", "--host", "0.0.0.0", "--port", "7788"]
