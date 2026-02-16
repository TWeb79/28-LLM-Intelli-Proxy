# Ollama Intelligent Router Proxy

An intelligent LLM routing proxy for Ollama with automatic model selection, load balancing, failover capabilities, and **transparent Ollama API compatibility**.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [AirLLM Configuration](#airllm-configuration)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Model Attributes](#model-attributes)
- [Fallback Configuration](#fallback-configuration)
- [Task Classification](#task-classification)
- [Web Dashboard](#web-dashboard)
- [Troubleshooting](#troubleshooting)
- [Performance Benefits](#performance-benefits)

## Features

### Intelligent Request Routing
- Automatic task classification based on prompt content
- Smart model selection based on:
  - Task category (code, reasoning, general, vision, image, uncensored)
  - Prompt complexity (1-10)
  - Model speed and capabilities

### Transparent Ollama API Compatibility
- **Ollama-compatible endpoints** - any Ollama client can connect directly
- Use `IntelliProxyLLM` model name for intelligent routing
- Works with existing Ollama software without code changes
- Full streaming support

### Auto-Discovery of Model Attributes
- Automatically discovers attributes for unknown models
- Uses LLM to analyze model names and determine:
  - Speed (1-10)
  - Complexity (1-10)
  - Preferred categories

### Load Balancing
- Automatic distribution of requests across multiple Ollama backends
- Consideration of model attributes (speed, complexity)
- Optimization for different task types

### Error Handling and Failover
- Automatic fallback to alternative models on:
  - Request timeouts
  - Memory issues
  - Model loading errors
- Configurable fallback chains per model

### Statistics and Monitoring
- Real-time statistics per model:
  - Request count
  - Average response time
  - Task distribution by category
- Request log with the last 50 requests
- Web dashboard for monitoring

### AirLLM Integration (Optional)
- Connect to remote/decentralized Ollama instances via IP:Port
- Enable AirLLM for large models (70B+) for KV cache compression
- Per-model AirLLM toggle in dashboard
- Persistent configuration storage

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Client    │────▶│  Router Proxy   │────▶│   Ollama    │
│ (Ollama API)│     │   (Port 9998)   │     │ (Port 11434)│
└─────────────┘     └──────────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │  Dashboard   │
                    │ (Port 9999)  │
                    └─────────────┘
```

### With AirLLM Support (Optional)

```
┌─────────────┐     ┌──────────────────┐     ┌───────────┐     ┌─────────────┐
│   Client    │────▶│  Router Proxy   │────▶│  AirLLM   │────▶│   Ollama    │
│ (Ollama API)│     │   (Port 9998)   │     │(Port 9996)│     │ (Target IP) │
└─────────────┘     └──────────────────┘     └───────────┘     └─────────────┘
                            │
                     ┌──────┴──────┐
                     │  Dashboard   │
                     │ (Port 9999)  │
                     └─────────────┘
```

**Connection Paths:**
- **Path A (Direct)**: Client → Proxy → Ollama (for models without AirLLM enabled)
- **Path B (via AirLLM)**: Client → Proxy → AirLLM → Ollama (for large 70B+ models with KV cache compression)

Clients can now use standard Ollama API - just connect to port 9998 instead of 11434!

## Prerequisites

### Software
- Docker
- Docker Compose

### Ports

| Port | Service | Description |
|------|---------|-------------|
| 9998 | API | Router API endpoints |
| 9999 | Dashboard | Web dashboard |
| 9997 | Ollama | Ollama server (direct) |
| 9996 | AirLLM | AirLLM service (optional) |

## AirLLM Configuration

### What is AirLLM?
AirLLM enables inference with large language models (70B+) using only 4GB of GPU memory through KV cache compression. This allows running very large models on limited hardware.

### Connection Flow

```
┌─────────────┐     ┌──────────────────┐     ┌───────────┐     ┌─────────────┐
│   Client    │────▶│  Router Proxy   │────▶│  AirLLM   │────▶│   Ollama    │
│ (Ollama API)│     │   (Port 9998)   │     │(Port 9996)│     │ (Target IP) │
└─────────────┘     └──────────────────┘     └───────────┘     └─────────────┘
```

### Configuration via Dashboard

1. Open the dashboard at http://localhost:9999
2. Navigate to the "AirLLM" tab
3. Configure:
   - **Target Ollama**: Set the IP and port of your Ollama instance (can be remote)
   - **AirLLM Service**: Enable/disable and set AirLLM host/port
   - **Per-Model**: Toggle AirLLM for specific models

### Configuration via Environment Variables

```bash
# Target Ollama (decentralized connection)
OLLAMA_TARGET_HOST=192.168.1.100
OLLAMA_TARGET_PORT=11434

# AirLLM Service
AIRLLM_ENABLED=true
AIRLLM_HOST=airllm
AIRLLM_PORT=9996
```

### Persistent Storage

Configuration is stored in Docker volume `router-data` at `/app/data`:
- `router_config.json` - Ollama target, AirLLM config, fallbacks
- `models.json` - Model attributes and settings

These files persist across container restarts.

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd 28-LLM-Intelli-Proxy
```

### 2. Start Docker Containers

```bash
# Build and start containers
docker-compose up -d --build

# Or start in the background
docker-compose up -d
```

### 3. Verify Status

```bash
# Check container status
docker ps

# View container logs
docker logs ollama-router
```

### 4. Open Dashboard

Open http://localhost:9999 in your browser.

## Configuration

### Environment Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `OLLAMA_BASE_URL` | http://ollama:11434 | Ollama server URL |
| `OLLAMA_TARGET_HOST` | ollama | Target Ollama host IP |
| `OLLAMA_TARGET_PORT` | 11434 | Target Ollama port |
| `AIRLLM_ENABLED` | false | Enable AirLLM integration |
| `AIRLLM_HOST` | airllm | AirLLM service host |
| `AIRLLM_PORT` | 9996 | AirLLM service port |
| `CLASSIFIER_MODEL` | qwen2.5:7b | Model for task classification |
| `PROXY_PORT` | 9998 | Router API port |
| `PROXY_HOST` | 0.0.0.0 | Router host binding |
| `WEB_PORT` | 9999 | Dashboard port |
| `WEB_HOST` | 0.0.0.0 | Dashboard host binding |
| `REQUEST_TIMEOUT` | 900 | Timeout in seconds (15 min) |
| `MODEL_FALLBACKS` | (JSON) | Fallback configuration |
| `DEFAULT_MODEL` | qwen2.5:7b | Default model when no routing |
| `ENABLE_AUTO_DISCOVERY` | true | Auto-discover model attributes |
| `DATA_DIR` | /app/data | Directory for persistent config |

## API Endpoints

### Health Check (Dashboard)

```bash
curl -s http://localhost:9998/api/health
```

Response:
```json
{
  "status": "healthy",
  "models_available": 5,
  "router_url": "http://localhost:9998",
  "api_status": "running",
  "ollama_url": "http://ollama:11434"
}
```

### Health Check (/health)

```bash
curl -s http://localhost:9998/health
```

Response:
```json
{
  "status": "healthy",
  "models_available": 5,
  "categories": {
    "general": 2,
    "code": 1,
    "reasoning": 2
  },
  "ollama_url": "http://ollama:11434",
  "classifier_model": "IntelliProxyLLM"
}
```

### Get All Models

```bash
curl -s http://localhost:9998/models
```

### Get Models by Category

```bash
curl -s http://localhost:9998/models/general
```

### Process Task

```bash
curl -s -X POST http://localhost:9998/task \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain quantum computing","task_type":"reasoning"}'
```

### Classify Task (Classification Only)

```bash
curl -s -X POST http://localhost:9998/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a function"}'
```

### Get Statistics

```bash
curl -s http://localhost:9998/stats
```

### Get Request Log

```bash
curl -s http://localhost:9998/requests
```

### Clear Request Log

```bash
curl -s -X POST http://localhost:9998/requests/clear
```

### Clear Request Log

```bash
curl -s -X POST http://localhost:9998/requests/clear
```

### Get AirLLM Configuration

```bash
curl -s http://localhost:9998/config/airllm
```

Response:
```json
{
  "ollama_host": "ollama",
  "ollama_port": 11434,
  "airllm_enabled": false,
  "airllm_host": "airllm",
  "airllm_port": 9996,
  "model_airllm_settings": {}
}
```

### Update Ollama Target Configuration

```bash
curl -s -X POST http://localhost:9998/config/ollama \
  -H "Content-Type: application/json" \
  -d '{"host":"192.168.1.100","port":11434}'
```

### Update AirLLM Service Configuration

```bash
curl -s -X POST http://localhost:9998/config/airllm/service \
  -H "Content-Type: application/json" \
  -d '{"enabled":true,"host":"airllm","port":9996}'
```

### Enable/Disable AirLLM for a Specific Model

```bash
curl -s -X POST http://localhost:9998/config/model/airllm \
  -H "Content-Type: application/json" \
  -d '{"model_name":"qwen2.5:7b","enabled":true}'
```

### Refresh Models (Auto-discovery)

```bash
curl -s -X POST http://localhost:9998/models/refresh
```

### Get Fallback Configuration

```bash
curl -s http://localhost:9998/config/fallbacks
```

### Update Fallback Configuration

```bash
curl -s -X POST http://localhost:9998/config/fallbacks \
  -H "Content-Type: application/json" \
  -d '{"fallbacks":{"deepseek-r1:latest":"qwen2.5:7b"},"timeout":300}'
```

### Run Performance Test

Run all 4 performance tests:

```bash
curl -s -X POST http://localhost:9998/performance-test \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is a transparent proxy?"}'
```

Run a single test mode:

```bash
curl -s -X POST http://localhost:9998/performance-test \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is a transparent proxy?","mode":"direct"}'
```

Available modes: `direct`, `direct_airllm`, `llm`, `llm_airllm`

---

## Ollama-Compatible API Endpoints

The proxy now supports the standard Ollama API - any software that works with Ollama can connect to the proxy directly!

### Get Models (with IntelliProxyLLM)

```bash
curl -s http://localhost:9998/api/tags
```

Response includes `IntelliProxyLLM` for intelligent routing:
```json
{
  "models": [
    {
      "name": "qwen2.5:7b",
      "size": 4700000000,
      "speed": 8,
      "complexity": 6,
      "preferred_for": ["general", "qa"]
    },
    {
      "name": "IntelliProxyLLM",
      "size": 0,
      "description": "Intelligent routing - proxy selects best model based on task"
    }
  ]
}
```

### Generate (with intelligent routing)

```bash
# Use IntelliProxyLLM for automatic model selection
curl -s -X POST http://localhost:9998/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "IntelliProxyLLM", "prompt": "Explain quantum computing"}'

# Or use a specific model directly
curl -s -X POST http://localhost:9998/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:7b", "prompt": "Hello!"}'
```

### Chat (with intelligent routing)

```bash
curl -s -X POST http://localhost:9998/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "IntelliProxyLLM",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Support

Both `/api/generate` and `/api/chat` support streaming:

```bash
curl -s -X POST http://localhost:9998/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "IntelliProxyLLM", "prompt": "Write a story", "stream": true}'
```

### Other Ollama Endpoints (Transparent Forwarding)

These are forwarded directly to Ollama:
- `POST /api/pull` - Pull models
- `DELETE /api/delete` - Delete models
- `POST /api/embeddings` - Generate embeddings

---

## API Response Formats

#### POST /task Response

```json
{
  "result": "Response text from model",
  "model_used": "qwen2.5:7b",
  "task_classification": "general",
  "timestamp": "2026-02-15T14:12:45.857189"
}
```

#### GET /stats Response

```json
{
  "total_requests": 10,
  "models": {
    "qwen2.5:7b": {
      "count": 5,
      "total_time": 75.5
    },
    "deepseek-r1:latest": {
      "count": 5,
      "total_time": 1200.0
    }
  },
  "model_avg_times": {
    "qwen2.5:7b": 15.1,
    "deepseek-r1:latest": 240.0
  },
  "categories": {
    "general": 5,
    "reasoning": 5
  },
  "last_update": "2026-02-15T14:12:45.857189"
}
```

#### GET /models Response

```json
{
  "total": 8,
  "categories": {
    "code": ["qwen2.5-coder:7b"],
    "reasoning": ["deepseek-r1:latest"],
    "general": ["qwen2.5:7b", "mistral:latest", "nemotron-3-nano:latest"],
    "vision": ["llava:latest"],
    "image": ["goonsai/qwen2.5-3B-goonsai-nsfw-100k:latest"],
    "uncensored": ["llama2-uncensored:latest"]
  },
  "models": {
    "qwen2.5-coder:7b": {
      "size": "4.7GB",
      "speed": 8,
      "complexity": 7,
      "preferred_for": ["code", "debugging", "technical"]
    }
  }
}
```

## Model Attributes

Each model has the following attributes:

| Attribute | Description | Values |
|-----------|-------------|--------|
| `speed` | Model speed | 1-10 (10 = fastest) |
| `complexity` | Complexity handling | 1-10 (10 = most complex) |
| `preferred_for` | Preferred task types | Array of categories |

### Configured Models

```python
MODEL_ATTRIBUTES = {
    "qwen2.5-coder:7b": {"speed": 8, "complexity": 7, "preferred_for": ["code"]},
    "deepseek-r1:latest": {"speed": 3, "complexity": 10, "preferred_for": ["reasoning"]},
    "llava:latest": {"speed": 4, "complexity": 6, "preferred_for": ["vision"]},
    "nemotron-3-nano:latest": {"speed": 10, "complexity": 4, "preferred_for": ["simple"]},
    "mistral:latest": {"speed": 7, "complexity": 6, "preferred_for": ["general"]},
    "qwen2.5:7b": {"speed": 8, "complexity": 6, "preferred_for": ["general"]},
}
```

## Fallback Configuration

Default fallback chain:

```python
MODEL_FALLBACKS = {
    "qwen2.5-coder:7b": "qwen2.5:7b",
    "deepseek-r1:latest": "qwen2.5:7b",
    "llava:latest": "qwen2.5:7b",
    "nemotron-3-nano:latest": "qwen2.5:7b",
    "mistral:latest": "qwen2.5:7b",
}
```

## Task Classification

The router automatically classifies tasks into the following categories:

- **code** - Writing, debugging, and analyzing code
- **reasoning** - Explaining, analyzing, and problem-solving
- **general** - Conversation, Q&A, and writing tasks
- **vision** - Analyzing and describing images
- **image** - Generating images
- **uncensored** - Creative writing and unrestricted output

## Web Dashboard

The dashboard at http://localhost:9999 provides:

1. **Status** - Overview of router and available models
2. **Models** - List of all models with categories
3. **Statistics** - Usage statistics and average response times
4. **Config** - Fallback configuration management
5. **API** - API reference with curl examples
6. **Debug** - Last 50 requests with prompts and response times

## Troubleshooting

### Container Won't Start

```bash
# View logs
docker logs ollama-router

# Rebuild container
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Ollama Not Reachable

```bash
# Check Ollama container
docker logs ollama-server

# Test Ollama directly
curl http://localhost:9997/api/tags
```

### Model Not Responding

- Increase timeout: `REQUEST_TIMEOUT=1800 docker-compose up -d`
- Fallback model will be used automatically

### Memory Errors

If a model requires more memory than available:
- Router automatically falls back to fallback model
- Remove model from MODEL_ATTRIBUTES or reconfigure fallback

## Performance Benefits

1. **Automatic Optimization** - Selects the best model based on task type
2. **Failover** - No interruption when model errors occur
3. **Statistics** - Real-time monitoring of model performance
4. **Easy Integration** - Standard REST API, compatible with existing applications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.