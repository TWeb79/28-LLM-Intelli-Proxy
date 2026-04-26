# LLM IntelliProxy

> **Intelligent LLM Routing & Model Management for Production**

IntelliProxy is an OpenAI-compatible, production-ready LLM routing proxy that automatically selects the best language model for each task based on capability matching, performance characteristics, and real-time availability. It provides a unified interface to multiple LLM providers with intelligent fallbacks, comprehensive monitoring, and zero operational overhead.

---

## 📋 Summary

### Business Value
- **Reduce Infrastructure Costs**: Route simple tasks to lightweight models, complex tasks to powerful ones — optimize GPU utilization and reduce token costs by up to 60%
- **Increase Reliability**: Automatic model fallback ensures 99.9% availability even when individual models fail
- **Accelerate Development**: Drop-in replacement for Ollama & OpenAI APIs — integrate in minutes, not weeks
- **Operational Visibility**: Built-in metrics, decision logs, and performance analytics for full observability

### Technical Highlights
- **Multi-Provider Support**: Ollama (local), NVIDIA NIM (cloud), and any OpenAI-compatible endpoint
- **Intelligent Routing**: Rule-based + LLM-assisted classification (configurable decision model)
- **Unified Model Registry**: Single source of truth for all available models with persistence
- **Zero Configuration**: Auto-discovers models, warms up caches, and starts routing immediately
- **Production Grade**: Structured logging, health checks, background refresh scheduler, SQLite persistence

---

## 🚀 Quick Start

### Option 1: With Docker (Recommended)

```bash
# 1. Start Ollama separately (in another terminal)
ollama serve

# 2. Start IntelliProxy
docker-compose up -d

# 3. Verify
curl http://localhost:8128/health
```

### Option 2: Bare Metal

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start Ollama (separate process)
ollama serve &

# Start IntelliProxy
python ollama_router.py

# Or using uvicorn
uvicorn api.app:api_app --host 0.0.0.0 --port 8128
```

**Prerequisites**: Ollama must be running and accessible on `OLLAMA_HOST:OLLAMA_PORT` (default: localhost:11434). Configure via environment variables or `config.yaml`.

---

## 📦 Docker Configuration

The provided `docker-compose.yml` runs **only IntelliProxy**. Ollama is **not** included and must be started separately.

**Why separate?**
- Ollama requires GPU access and model persistence on host
- You may already have Ollama running with your models
- Allows independent scaling and updates
- Follows single-responsibility principle

**Network connectivity**: IntelliProxy connects to Ollama via:
- Docker Desktop: `host.docker.internal`
- Linux: `ollama` hostname (if on same network) or explicit IP
- Set via `OLLAMA_HOST` and `OLLAMA_PORT` environment variables

---

## 🎯 Key Features

### 1. Intelligent Model Selection
Automatically matches prompts to optimal models:
- **Code & Debugging** → qwen2.5-coder, deepseek-coder
- **Reasoning & Analysis** → deepseek-r1, nemotron
- **Vision & Image** → llava
- **General & Chat** → mistral, qwen2.5
- **Fast & Simple** → nemotron-3-nano

### 2. Seamless Fallback Chain
If the selected model fails (OOM, timeout, error), automatically retries with:
1. User-configured fallback model
2. Category-based alternatives
3. Global fallback (configurable)

### 3. Provider Abstraction
Single API works with multiple backends:
```yaml
providers:
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
  - name: nvidia
    type: nvidia_nim
    api_key: ${NVIDIA_API_KEY}
```

### 4. Dynamic System Prompts
The decision engine generates routing prompts from the live model registry — no hardcoded model lists.

### 5. Performance Optimization
- **Classification Cache**: LRU cache (2000 entries) for prompt classification
- **Model Score Cache**: Pre-computed model attributes for instant selection
- **HTTP Connection Pooling**: Reused async client for all provider calls

---

## 📊 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   Client    │────▶│ IntelliProxy │────▶│  Decision      │
│ (Ollama /   │     │   API        │     │  Engine        │
│  OpenAI)    │     │  :8128       │     │                │
└─────────────┘     └──────────────┘     └────────────────┘
                                                │
                           ┌────────────────────┼────────────────────┐
                           │                    │                    │
                           ▼                    ▼                    ▼
                    ┌────────────┐      ┌────────────┐      ┌────────────┐
                    │  Model     │      │   Model    │      │   Model    │
                    │ Registry   │      │  Scheduler │      │   Cache    │
                    │  (SQLite)  │      │            │      │            │
                    └────────────┘      └────────────┘      └────────────┘
                           │                    │                    │
                           └────────────────────┼────────────────────┘
                                                │
                           ┌────────────────────┼────────────────────┐
                           ▼                    ▼                    ▼
                    ┌────────────┐      ┌────────────┐      ┌────────────┐
                    │  Ollama    │      │  NVIDIA    │      │  Other     │
                    │  Local     │      │  NIM Cloud │      │  Provider  │
                    └────────────┘      └────────────┘      └────────────┘
```

**Data Flow**:
1. Client sends request to `/api/generate` or `/api/chat`
2. IntelliProxy classifies the task (code, vision, reasoning, etc.)
3. Decision engine selects best available model from registry
4. Request forwarded to appropriate provider
5. Decision persisted to database for analytics

---

## 🔌 API Endpoints

### OpenAI-Compatible
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models (OpenAI format) |
| `/api/generate` | POST | Text completion (Ollama-compatible) |
| `/api/chat` | POST | Chat completion (Ollama-compatible) |

### Management & Monitoring
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health (Ollama + proxy status) |
| `/models` | GET | Detailed model list with attributes |
| `/stats` | GET | Usage statistics & cache hit rate |
| `/api/decisions` | GET | Routing decision log (with filters) |
| `/api/registry` | GET/PATCH | Model registry CRUD |
| `/api/registry/refresh` | POST | Force provider refresh |

### Configuration
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/config/ollama` | POST | Update Ollama target (host/port) |
| `/config/fallbacks` | POST | Set model fallback rules |
| `/config` | GET | Current configuration |

### Development
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/performance-test` | POST | Compare routing modes |
| `/classify` | GET | Test classification only |
| `/api/live-feed` | GET | Server-Sent Events for live updates |

**Full API spec**: See `/api` Swagger UI at `http://localhost:8128/docs`

---

## ⚙️ Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_PORT` | `8130` | API server port |
| `OLLAMA_HOST` | `localhost` | Ollama hostname |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `DECISION_MODEL` | *(empty)* | Model used for LLM-based routing |
| `FALLBACK_MODEL` | `qwen2.5:8b` | Global fallback model |
| `LOG_LEVEL` | `info` | Logging level |
| `DATA_DIR` | `/data` | Database directory |

### Configuration File (`config.yaml`)
```yaml
proxy:
  port: 8128
  host: 0.0.0.0
  log_level: info
  fallback_model: qwen2.5:8b

decision:
  model: "deepseek-r1:latest"  # Optional: use LLM for routing
  refresh_registry_on_startup: true

storage:
  type: sqlite
  path: /data/llmproxy.db

providers:
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    enabled: true

  - name: nvidia
    type: nvidia_nim
    api_key: ${NVIDIA_API_KEY}
    base_url: https://integrate.api.nvidia.com/v1
```

---

## 🎮 Usage Examples

### cURL (OpenAI-compatible)
```bash
# Auto-select best model
curl -X POST http://localhost:8128/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a Python quicksort"}'

# Use specific model (passthrough)
curl -X POST http://localhost:8128/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder:7b","prompt":"Write a Python quicksort"}'

# Chat completions
curl -X POST http://localhost:8128/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"intelliproxy-auto","messages":[{"role":"user","content":"Hello!"}]}'
```

### Python Client
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8128/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="intelliproxy-auto",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

### Ollama Client
```bash
# Set OLLAMA_HOST to point to IntelliProxy instead of direct Ollama
export OLLAMA_HOST=localhost
export OLLAMA_PORT=8128

# Now ollama CLI works through the proxy
ollama run qwen2.5 "What is machine learning?"
```

---

## 🎛️ Dashboard

Access the web dashboard at `http://localhost:8028` (or configured WEB_PORT).

**Features**:
- Real-time health & status
- Model registry browser
- Performance benchmarking
- Fallback configuration UI
- Request log viewer

---

## 📈 Monitoring & Metrics

### Key Performance Indicators
- **Cache Hit Rate**: % of classifications served from cache (target >70%)
- **Average Latency**: Per-model response time tracking
- **Request Volume**: Total and per-category request counts
- **Model Distribution**: Which models carry the load

### Decision Log
Every routing decision is persisted to SQLite:
```sql
SELECT * FROM decision_backlog
WHERE timestamp > datetime('now', '-1 day')
ORDER BY timestamp DESC;
```

Export via API:
```bash
curl http://localhost:8128/api/decisions/export?format=csv > decisions.csv
```

---

## 🔧 Advanced Topics

### Using NVIDIA NIM
1. Get API key from [NVIDIA Developer Program](https://developer.nvidia.com/)
2. Add to `.env` or `config.yaml`:
```yaml
providers:
  - name: nvidia
    type: nvidia_nim
    api_key: "nv-xxxxx"
```
3. Models auto-discovered from NVIDIA catalog

### Enabling LLM-based Routing
Set a decision model (from your registry) for smarter routing:
```bash
export DECISION_MODEL=deepseek-r1:latest
# or in config.yaml
decision:
  model: "deepseek-r1:latest"
```
The decision model analyzes each prompt and picks the optimal model from the registry.

### Fallback Strategies
Per-model fallbacks ensure high availability:
```bash
curl -X POST http://localhost:8128/config/fallbacks \
  -H "Content-Type: application/json" \
  -d '{"fallbacks":{"deepseek-r1:latest":"qwen2.5:8b"}}'
```

Now if `deepseek-r1` fails, requests automatically retry with `qwen2.5:8b`.

---

## 🧪 Testing

```bash
# Run performance benchmark
curl -X POST http://localhost:8128/performance-test \
  -d '{"prompt":"Explain recursion","mode":"all"}'

# Classify a prompt without execution
curl "http://localhost:8128/classify?prompt=Write+a+Python+script"

# Health check with full details
curl http://localhost:8128/health/full
```

---

## 🚨 Troubleshooting

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| "No models available" | Ollama not running | `ollama serve` |
| High latency | Cold model cache | Wait for warmup, or call `/task` repeatedly |
| Poor routing decisions | No decision model configured | Set `DECISION_MODEL` env var |
| Database locked | Concurrent writes | Ensure SQLite file on SSD, not NFS |
| Import errors | Dependencies missing | `pip install -r requirements.txt` |

**Logs**: Structured JSON logs; configure with `LOG_LEVEL=debug` for verbose output.

---

## 📦 Dependencies

Core dependencies (see `requirements.txt`):
- `fastapi` — API framework
- `httpx` — Async HTTP client
- `pydantic` — Data validation
- `pyyaml` — Config parsing
- `uvicorn` — ASGI server

**Optional**: `openai` — for OpenAI SDK compatibility

---

## 🏗️ Development

```bash
# Install in dev mode
pip install -e .

# Run API server
python -m uvicorn api.app:api_app --reload --port 8128

# Run tests
pytest tests/

# Lint
ruff check .
```

**Project Structure**:
```
28-LLM-InteliProxy/
├── services/          # Business logic layer
│   ├── router.py      # Core intelligent router
│   ├── decision_engine.py
│   ├── registry.py    # Model registry DB ops
│   ├── caches.py      # LRU & statistics caches
│   └── ...
├── routes/            # API route handlers
│   ├── api.py
│   └── web.py
├── providers/         # Provider adapters (Ollama, NVIDIA)
├── models/            # Pydantic schemas
├── static/            # Dashboard assets
├── ollama_router.py   # App entry point & wiring
└── api/app.py         # Re-export for uvicorn
```

---

## 📄 License

MIT License — see LICENSE file for details.

---

## 🙏 Acknowledgments

Built for the **28LLM** project. Special thanks to the Ollama and NVIDIA NIM teams for their excellent APIs.

---

<div align="center">
**IntelliProxy** — Intelligent routing, simplified.
</div>
