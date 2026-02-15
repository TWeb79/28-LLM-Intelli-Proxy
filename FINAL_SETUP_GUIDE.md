# 🚀 Ollama Router with Integrated Ollama - Final Setup Guide

## Architecture Overview

Your final setup includes everything in one Docker container:

```
┌─────────────────────────────────────────────────────────┐
│                    DOCKER CONTAINER                      │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Ollama Server (Internal 11434)           │  │
│  │            ↓ Exposed as 9997                     │  │
│  │     (All LLM Models run here)                    │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↑                                 │
│                        │ (Internal Network)              │
│                        │                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │      Router Proxy Server (Port 9998)              │  │
│  │   - Auto-detects models on Ollama 9997           │  │
│  │   - Intelligent task routing                     │  │
│  │   - Statistics tracking                          │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↑                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Web Dashboard (Port 9999)                       │  │
│  │   - View models & statistics                     │  │
│  │   - Copy code examples                           │  │
│  │   - Connectivity info                            │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
         ↑                          ↑                ↑
    YOUR APPLICATION          WEB BROWSER       CLI/cURL
    Port 9998                  Port 9999        Port 9998
```

---

## 🎯 Port Configuration

| Component | Port | Access From | Purpose |
|-----------|------|-------------|---------|
| **Proxy API** | 9998 | External (Your apps) | Send tasks to intelligent router |
| **Web Dashboard** | 9999 | External (Browser) | Monitor, view stats, copy examples |
| **Ollama** | 9997 | External (Direct access) | Optional direct model access |
| **Ollama Internal** | 11434 | Docker internal only | Used by proxy internally |

---

## ✨ What Your Applications Do

```
Your Application Code (Python, Node, etc.)
                    ↓
        POST http://localhost:9998/task
        {
            "prompt": "Write a Python function",
            "task_type": null,
            "stream": false
        }
                    ↓
         Ollama Router (Port 9998)
         - Classifies the task
         - Selects best model
         - Sends to Ollama on 9997
                    ↓
         Ollama Server (Port 9997 externally)
         - Runs the selected LLM
         - Returns result
                    ↓
        Response from 9998
        {
            "result": "def sort_list(lst): return sorted(lst)",
            "model_used": "qwen3-coder",
            "task_classification": "code",
            "timestamp": "2024-01-15T10:30:45"
        }
```

---

## 🐳 Docker Compose Setup

### File: `docker-compose.yml`

```yaml
version: '3.8'

services:
  # Ollama (Ports: 9997 external, 11434 internal)
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    ports:
      - "9997:11434"  # External:Internal
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    restart: unless-stopped
    networks:
      - ollama-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Router Proxy (Port: 9998)
  router:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ollama-router
    ports:
      - "9998:9998"    # API for external apps
      - "9999:9999"    # Web Dashboard
    volumes:
      - ./static:/app/static
    environment:
      # Connect to Ollama on same docker network
      - OLLAMA_BASE_URL=http://ollama:11434
      
      # API for external apps
      - PROXY_HOST=0.0.0.0
      - PROXY_PORT=9998
      
      # Web Dashboard
      - WEB_HOST=0.0.0.0
      - WEB_PORT=9999
      
      # Classifier
      - CLASSIFIER_MODEL=qwen2.5:7b
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - ollama-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9998/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  ollama-data:
    driver: local

networks:
  ollama-network:
    driver: bridge
```

---

## 🚀 Quick Start

### Step 1: Prepare Files

Ensure you have these files:
```
project/
├── docker-compose.yml              (NEW - final version)
├── Dockerfile                      (UPDATED - use final)
├── ollama_router_9998.py          (NEW - router with port 9998)
├── router_client.py
├── requirements.txt
└── static/
    ├── index.html
    └── app.js
```

### Step 2: Build and Start

```bash
# Build the containers
docker-compose build

# Start both services
docker-compose up -d

# View logs
docker-compose logs -f
```

You should see:
```
ollama-server  | 2024-01-15T10:00:00.000 "GET /api/tags HTTP/1.1" 200
ollama-router  | 🚀 Ollama Intelligent Router Starting...
ollama-router  | ✅ Discovered 10 models
ollama-router  | ✅ API running at http://0.0.0.0:9998
ollama-router  | ✅ Dashboard at http://0.0.0.0:9999
```

### Step 3: Access Services

- **Web Dashboard**: http://localhost:9999
- **Router API**: http://localhost:9998
- **Ollama Direct**: http://localhost:9997

---

## 💻 Using the Router from Your Applications

### Python Example

```python
import requests

# Your application sends requests to port 9998
response = requests.post(
    "http://localhost:9998/task",
    json={
        "prompt": "Write a Python function to sort a list",
        "task_type": None,  # Auto-classify
        "stream": False
    }
)

result = response.json()
print(f"Model: {result['model_used']}")
print(f"Result: {result['result']}")
```

### Node.js Example

```javascript
const response = await fetch('http://localhost:9998/task', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        prompt: "Explain machine learning",
        task_type: null,
        stream: false
    })
});

const result = await response.json();
console.log(`Model: ${result.model_used}`);
console.log(`Result: ${result.result}`);
```

### cURL Example

```bash
curl -X POST http://localhost:9998/task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "task_type": null,
    "stream": false
  }'
```

---

## 🔍 Router API Endpoints (Port 9998)

### Health Check
```bash
curl http://localhost:9998/health
```

### List All Models
```bash
curl http://localhost:9998/models
```

### List Models by Category
```bash
curl http://localhost:9998/models/code
# Categories: code, vision, reasoning, general, uncensored
```

### Process Task (Main Endpoint)
```bash
curl -X POST http://localhost:9998/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your task", "task_type": null}'
```

### Classify Only (Without Processing)
```bash
curl -X POST http://localhost:9998/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your task"}'
```

### Get Statistics
```bash
curl http://localhost:9998/stats
```

---

## 📊 Web Dashboard (Port 9999)

Open http://localhost:9999 to see:

- **📊 Dashboard Tab** - Real-time statistics
- **🧠 Models Tab** - All available models with features
- **🐍 Python Guide** - Copy-paste code examples
- **✍️ Prompts** - Optimized prompt templates
- **🔌 Connectivity** - API details and troubleshooting

The dashboard updates automatically every 5 seconds!

---

## 🎯 How the Router Discovers Models

1. **On startup**, router connects to Ollama at `http://ollama:11434` (internal)
2. **Requests**: `GET http://ollama:11434/api/tags`
3. **Gets list** of all installed Ollama models
4. **Auto-categorizes** them (code, vision, reasoning, etc.)
5. **Now ready** to route tasks to the right model

---

## 🔄 Internal Docker Network

**Within the Docker container:**
- Router communicates with Ollama via internal network name: `ollama`
- URL: `http://ollama:11434` (internal)
- Very fast, no external routing needed

**From your machine:**
- Access Ollama directly at: `http://localhost:9997`
- Access Router at: `http://localhost:9998`
- Access Dashboard at: `http://localhost:9999`

---

## 📝 Environment Variables

All set in `docker-compose.yml`:

```bash
# Ollama location (internal to docker)
OLLAMA_BASE_URL=http://ollama:11434

# Router API (external apps connect here)
PROXY_HOST=0.0.0.0
PROXY_PORT=9998

# Web Dashboard
WEB_HOST=0.0.0.0
WEB_PORT=9999

# LLM for task classification
CLASSIFIER_MODEL=qwen2.5:7b
```

---

## ✅ Verification Checklist

### 1. Check Docker Containers Running

```bash
docker ps

# Should show:
# - ollama-server
# - ollama-router
```

### 2. Check Ollama Health

```bash
curl http://localhost:9997/api/tags

# Should return JSON with list of models
```

### 3. Check Router Health

```bash
curl http://localhost:9998/health

# Should return:
# {"status": "healthy", "models_available": 10, ...}
```

### 4. Check Dashboard

Open http://localhost:9999 in browser

Should show all models and their stats.

### 5. Send Test Request

```bash
curl -X POST http://localhost:9998/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "task_type": null}'

# Should return a result from a model
```

---

## 🆘 Troubleshooting

### "Connection refused on 9998"

```bash
# Check if containers are running
docker ps

# Check router logs
docker-compose logs router

# Restart
docker-compose restart router
```

### "No models found"

```bash
# Check if Ollama has models
curl http://localhost:9997/api/tags

# If empty, pull a model from Ollama:
docker exec ollama-server ollama pull qwen2.5:7b

# Restart router to detect
docker-compose restart router
```

### "Can't access dashboard"

```bash
# Try direct IP
http://127.0.0.1:9999

# Check port is exposed
docker port ollama-router

# Should show:
# 9998/tcp -> 0.0.0.0:9998
# 9999/tcp -> 0.0.0.0:9999
```

### "Ollama not connecting to router"

```bash
# The docker-compose.yml has the fix:
# depends_on with condition: service_healthy

# Manually check connection
docker exec ollama-router curl http://ollama:11434/api/tags

# Should work if containers are on same network
```

---

## 🐳 Docker Management Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Just router
docker-compose logs -f router

# Just ollama
docker-compose logs -f ollama
```

### Stop Services
```bash
docker-compose stop
```

### Restart Services
```bash
docker-compose restart

# Or just one
docker-compose restart router
```

### Remove All (Including Data)
```bash
# Stop and remove
docker-compose down

# Also remove volumes (deletes models!)
docker-compose down -v
```

### Rebuild After Changes
```bash
docker-compose build --no-cache
docker-compose up -d
```

---

## 📁 File Structure

```
your-project/
├── docker-compose.yml          (Combined services)
├── Dockerfile                  (Final version)
├── ollama_router_9998.py      (Router with port 9998)
├── router_client.py            (Python client library)
├── requirements.txt            (Dependencies)
├── static/
│   ├── index.html             (Dashboard UI)
│   └── app.js                 (Dashboard JS)
└── README.md
```

---

## 🚀 Complete Example Workflow

### Setup
```bash
# 1. Start everything
docker-compose up -d

# 2. Wait for startup (10-30 seconds for model loading)
docker-compose logs router

# 3. Verify connection
curl http://localhost:9998/health
```

### Monitor
```bash
# Open dashboard
# http://localhost:9999
```

### Use from Python App
```python
import requests

response = requests.post('http://localhost:9998/task', json={
    'prompt': 'Write a Python function',
    'task_type': None
})

result = response.json()
print(result['model_used'])
print(result['result'])
```

### View Statistics
```bash
curl http://localhost:9998/stats | python -m json.tool
```

---

## 💡 Key Points

✅ **Ollama inside Docker** - Models run in container at 9997
✅ **Router inside Docker** - Proxy at 9998 connects to Ollama internally  
✅ **Your Apps Connect to 9998** - Simple integration
✅ **Dashboard at 9999** - Monitor everything
✅ **Auto Model Discovery** - Router finds all models on startup
✅ **Internal Network** - Fast communication within Docker

---

## 🎯 Summary

Your setup:
1. **Ollama runs inside Docker** (models at port 9997)
2. **Router runs in same Docker** (API at port 9998)
3. **Router detects all Ollama models** on startup
4. **Your apps send requests to 9998** (router proxy)
5. **Router intelligently routes** to best model on 9997
6. **Dashboard monitors everything** on port 9999

**Result:** One Docker container with everything, super simple! 🎉

---

**Ready to go! Start with `docker-compose up -d`** 🚀
