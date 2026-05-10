# syntax=docker/dockerfile:1.6
#
# tideon.ai — API server image (FastAPI / OpenAI-compatible)
#
# Listens on $PORT (Railway convention) or 8000 by default.
# Multi-stage: builder compiles wheels, runtime is slim and non-root.
# Installs CPU-only torch — Railway has no GPU, and the CUDA build adds 4 GB.
#
# Build:   docker build -f Dockerfile -t tideon-ai/teffgen-api:0.2.0 .
# Run:     docker run -p 8000:8000 -e TEFFGEN_API_KEY=secret tideon-ai/teffgen-api:0.2.0
# Railway: dockerfilePath = "Dockerfile" in railway.toml — Railway injects $PORT.

# ---- builder ----
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml setup.py requirements.txt README.md ./
COPY teffgen ./teffgen
RUN pip wheel --wheel-dir=/wheels .

# ---- runtime ----
FROM python:3.11-slim

LABEL org.opencontainers.image.title="tideon.ai-api" \
      org.opencontainers.image.description="tideon.ai (teffgen) OpenAI-compatible API server" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.source="https://github.com/tideon-ai/teffgen"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    TEFFGEN_RATE_LIMIT=60

# curl is for the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 teffgen

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels

USER teffgen

# Default port; Railway overrides via $PORT at runtime.
ENV PORT=8000
EXPOSE 8000

# Smoke-check at build time so a broken install fails the build, not the deploy.
RUN python -c "import teffgen; print('teffgen', teffgen.__version__)"

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# CMD only (no ENTRYPOINT) so `docker run image <other-cmd>` cleanly overrides.
# Default uses sh -c to expand $PORT at container start.
CMD ["sh", "-c", "exec teffgen serve --host 0.0.0.0 --port ${PORT}"]
