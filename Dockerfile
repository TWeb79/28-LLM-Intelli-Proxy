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

# Create non-root user
RUN addgroup --system proxy && adduser --system --ingroup proxy proxy

# Create data dir owned by runtime user
RUN mkdir -p /data && chown proxy:proxy /data

# Copy installed wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt || true

# Copy application files
COPY --chown=proxy:proxy . /app

USER proxy

# Environment defaults (can be overridden at runtime)
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV PROXY_HOST=0.0.0.0
ENV PROXY_PORT=8128
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8028
ENV CLASSIFIER_MODEL=qwen2.5:7b

EXPOSE 8128 8028

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8128/health || exit 1

CMD ["python", "-u", "ollama_router.py"]
