FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install into a target directory
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip wheel --no-deps --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app

# Create non-root user (handle case where group might already exist)
RUN (getent group proxy >/dev/null 2>&1 || addgroup --system proxy) && \
    (getent passwd proxy >/dev/null 2>&1 || adduser --system --ingroup proxy proxy)

# Create data dir owned by runtime user
RUN mkdir -p /data && chown proxy:proxy /data

# Copy installed wheels and install
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt || true

# Copy application files
COPY --chown=proxy:proxy . /app

USER proxy

# Environment defaults (can be overridden at runtime)
# Note: OLLAMA_BASE_URL should point to a separately running Ollama instance
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV PROXY_HOST=0.0.0.0
ENV PROXY_PORT=8128
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8028
ENV CLASSIFIER_MODEL=qwen2.5:8b
ENV REQUEST_TIMEOUT=120
ENV FALLBACK_TIMEOUT=30

EXPOSE 8128 8028

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8128/health || exit 1

CMD ["python", "-u", "ollama_router.py"]
