# syntax=docker/dockerfile:1

# ---------------------------------------------------------
# Stage 1: Build virtual environment with uv
# ---------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Install uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation and copy mode
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Build argument for CPU or GPU target (defaults to cpu)
ARG BUILD_TARGET=cpu

COPY pyproject.toml uv.lock README.md ./

# Install dependencies into /app/.venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy source tree and install project package
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------
# Stage 2: Minimal runtime image
# ---------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ARG BUILD_TARGET=cpu
ENV BUILD_TARGET=${BUILD_TARGET}

# Runtime environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/home/appuser/.cache/huggingface" \
    TORCH_HOME="/home/appuser/.cache/torch" \
    RAG_DEVICE="cpu"

# Security: Create non-root system user
RUN groupadd -r -g 1000 appuser && \
    useradd -r -g appuser -u 1000 -m -d /home/appuser -s /bin/bash appuser

WORKDIR /app

# Copy virtualenv and application from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser data/golden/ /app/data/golden/
COPY --chown=appuser:appuser pyproject.toml README.md /app/

# Ensure runtime directories exist with proper non-root ownership
RUN mkdir -p /app/data/raw /app/data/qdrant /home/appuser/.cache && \
    chown -R appuser:appuser /app/data /home/appuser/.cache

USER appuser

EXPOSE 8000

# Container healthcheck querying FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: launch production ASGI server
CMD ["uvicorn", "rag.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
