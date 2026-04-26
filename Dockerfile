FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Requirements zuerst (für Cache)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App kopieren
COPY . .

# Non-root user
RUN (getent group proxy >/dev/null 2>&1 || addgroup --system proxy) && \
    (getent passwd proxy >/dev/null 2>&1 || adduser --system --ingroup proxy proxy)

# Datenverzeichnis
RUN mkdir -p /data && chown proxy:proxy /data

USER proxy

# Environment Variablen
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV PROXY_HOST=0.0.0.0
ENV PROXY_PORT=8128
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8028
ENV CLASSIFIER_MODEL=qwen2.5:8b
ENV REQUEST_TIMEOUT=120
ENV FALLBACK_TIMEOUT=30

EXPOSE 8128 8028

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8128/health || exit 1

# 🚀 WICHTIG: Starte sowohl die API als auch das Dashboard
# Verwende start_servers.py um beide Uvicorn-Instanzen zu starten.
COPY --chown=proxy:proxy start_servers.py /app/start_servers.py
RUN chmod +x /app/start_servers.py
CMD ["/usr/bin/env", "python3", "/app/start_servers.py"]