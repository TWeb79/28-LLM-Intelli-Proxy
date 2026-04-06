FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ollama_router.py ollama_router.py
COPY router_client.py .

# Copy static files (dashboard)
COPY static/ ./static/

# Environment variables
# Ollama is on same docker network at http://ollama:11434
ENV OLLAMA_BASE_URL=http://ollama:11434

# API Server (port 8128 - external applications connect here)
ENV PROXY_HOST=0.0.0.0
ENV PROXY_PORT=8128

# Web Dashboard (port 8028 - for monitoring)
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8028

# Classifier model
ENV CLASSIFIER_MODEL=qwen2.5:7b

# Expose ports
EXPOSE 8128 8028

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8128/health || exit 1

# Run the router (both API and Web servers)
CMD ["python", "-u", "ollama_router.py"]
