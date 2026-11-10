# syntax=docker/dockerfile:1
#
# Moxa web UI (Streamlit) image.
#
#   docker compose up --build            # see compose.yaml for keys, sign-in secrets and volumes
#
# Secrets are never baked in: .env and .streamlit/secrets.toml are excluded by .dockerignore and
# supplied at run time.

# ---------- build: dependencies, app, and the retrieval models ----------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    FASTEMBED_CACHE_PATH=/app/models

WORKDIR /app

# Dependencies first, so code changes don't reinstall them
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Download the BM25 and re-ranker models (~560 MB) at build time, so containers start without
# fetching them. Build with --build-arg RERANK_MODEL=... to bake in a different re-ranker; setting
# RERANK_MODEL only at run time downloads that model on first use instead.
ARG RERANK_MODEL=BAAI/bge-reranker-v2-m3-int8
RUN /app/.venv/bin/python -c "\
from fastembed import SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
SparseTextEmbedding('Qdrant/bm25'); \
TextCrossEncoder('${RERANK_MODEL}')"

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---------- runtime ----------
FROM python:3.12-slim

ARG RERANK_MODEL=BAAI/bge-reranker-v2-m3-int8

RUN useradd --create-home --uid 1000 moxa

WORKDIR /app
COPY --from=builder --chown=moxa:moxa /app /app
# Writable state: embedded Qdrant index and logs (mount volumes here to keep them)
RUN mkdir -p /app/data/qdrant /app/log && chown -R moxa:moxa /app/data /app/log

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    RERANK_MODEL=${RERANK_MODEL} \
    QDRANT_PATH=/app/data/qdrant \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER moxa
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["moxa-ui"]
