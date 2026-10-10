FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x deploy/entrypoint.sh \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/data && chown -R app:app /app/data
USER app

# Public-demo defaults. Secrets (NVIDIA_API_KEY, ...) are injected at
# runtime and are never baked into the image.
ENV DEMO_MODE=1 \
    LLM_PROVIDER=nvidia \
    EMBEDDING_PROVIDER=minilm \
    MEMORY_BACKEND=memory \
    CHROMA_PERSIST_DIR=/app/data/chroma

EXPOSE 8501
ENTRYPOINT ["deploy/entrypoint.sh"]
