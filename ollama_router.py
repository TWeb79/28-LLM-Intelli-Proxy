#!/usr/bin/env python3
"""
Intelligent Ollama Router Proxy with Web Dashboard
- Auto-discovers installed models
- Uses lightweight LLM to classify tasks
- Routes to appropriate model
- Tracks statistics and serves web UI
- API on port 9998, Dashboard on 9999, Ollama on 9997
"""

import os
import json
import time
import requests
import asyncio
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from collections import defaultdict
import threading

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")  # Default to docker internal
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "qwen2.5:7b")
PROXY_PORT = int(os.getenv("PROXY_PORT", "9998"))  # Changed from 8000 to 9998
PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "9999"))
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "900"))  # Default 15 minutes timeout

# Fallback configuration for each model
MODEL_FALLBACKS = {
    "qwen2.5-coder:7b": "qwen2.5:7b",
    "deepseek-r1:latest": "qwen2.5:7b",
    "llava:latest": "qwen2.5:7b",
    "nemotron-3-nano:latest": "qwen2.5:7b",
    "mistral:latest": "qwen2.5:7b",
    "goonsai/qwen2.5-3B-goonsai-nsfw-100k:latest": "qwen2.5:7b",
    "llama2-uncensored:latest": "qwen2.5:7b",
}

# Fallback from environment variable (JSON format)
FALLBACK_ENV = os.getenv("MODEL_FALLBACKS", "")
if FALLBACK_ENV:
    try:
        import json
        MODEL_FALLBACKS.update(json.loads(FALLBACK_ENV))
    except:
        pass

# Request/Response models
class TaskRequest(BaseModel):
    prompt: str
    task_type: Optional[str] = None
    stream: bool = False

class TaskResponse(BaseModel):
    result: str
    model_used: str
    task_classification: str
    timestamp: str

# Statistics tracking
class Statistics:
    def __init__(self):
        self.total_requests = 0
        self.models = defaultdict(lambda: {"count": 0, "total_time": 0})
        self.categories = defaultdict(int)
        self.last_update = datetime.now()
        self.lock = threading.Lock()
    
    def record_request(self, model: str, category: str, execution_time: float = 0):
        with self.lock:
            self.total_requests += 1
            self.models[model]["count"] += 1
            self.models[model]["total_time"] += execution_time
            self.categories[category] += 1
            self.last_update = datetime.now()
    
    def to_dict(self):
        with self.lock:
            return {
                "total_requests": self.total_requests,
                "models": dict(self.models),
                "categories": dict(self.categories),
                "last_update": self.last_update.isoformat()
            }

# Initialize statistics
stats = Statistics()

# Initialize FastAPI apps
api_app = FastAPI(
    title="Ollama Intelligent Router",
    description="Intelligent task routing proxy for Ollama models",
    version="1.0"
)

web_app = FastAPI(
    title="Ollama Router Dashboard",
    description="Web dashboard for router management",
    version="1.0"
)

# Global state
available_models = {}
model_categories = {}

class OllamaRouter:
    """Intelligent routing engine"""
    
    def __init__(self, base_url: str, classifier_model: str):
        self.base_url = base_url
        self.classifier_model = classifier_model
        self.available_models = {}
        self.model_categories = {}
        self.connection_attempts = 0
        self.max_attempts = 5
    
    async def discover_models(self) -> Dict:
        """Scan Ollama for available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                
                for model in models:
                    name = model["name"]
                    size = model.get("size", 0)
                    self.available_models[name] = {
                        "name": name,
                        "size": size,
                        "modified": model.get("modified_at", ""),
                    }
                
                print(f"✅ Discovered {len(self.available_models)} models from {self.base_url}")
                self._categorize_models()
                return self.available_models
            else:
                raise Exception(f"Failed to connect to Ollama (status {response.status_code})")
        
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Cannot connect to Ollama at {self.base_url}")
            print(f"   Error: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error discovering models: {e}")
            return {}
    
    def _categorize_models(self):
        """Categorize models by capability (heuristic)"""
        self.model_categories = {
            "code": [],
            "vision": [],
            "reasoning": [],
            "general": [],
            "uncensored": [],
        }
        
        keywords = {
            "code": ["coder", "code"],
            "vision": ["llava", "vision"],
            "reasoning": ["deepseek", "r1"],
            "uncensored": ["uncensored", "goonsai", "nsfw"],
        }
        
        for model_name in self.available_models.keys():
            categorized = False
            for category, words in keywords.items():
                if any(word in model_name.lower() for word in words):
                    self.model_categories[category].append(model_name)
                    categorized = True
                    break
            
            if not categorized:
                self.model_categories["general"].append(model_name)
        
        print("\n📊 Model Categories:")
        for category, models in self.model_categories.items():
            if models:
                print(f"  {category}: {', '.join(models)}")
    
    async def classify_task(self, prompt: str) -> str:
        """Use lightweight LLM to classify the task"""
        classification_prompt = f"""Analyze this task and respond with ONLY one category:

Task: "{prompt}"

Categories:
- code: Writing, debugging, or analyzing code
- vision: Analyzing, describing, or processing images/visual content
- reasoning: Explaining, analyzing, problem-solving with step-by-step thinking
- uncensored: Creative writing, adult content, or unrestricted output
- general: General conversation, Q&A, writing, summarization

Respond with ONLY the category name, nothing else."""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.classifier_model,
                    "prompt": classification_prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                classification = response.json()["response"].strip().lower()
                valid_categories = ["code", "vision", "reasoning", "uncensored", "general"]
                if classification in valid_categories:
                    return classification
                else:
                    return "general"
            else:
                return "general"
        
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return "general"
    
    def _select_best_model(self, category: str) -> Optional[str]:
        """Select best model for category"""
        if category in self.model_categories and self.model_categories[category]:
            return self.model_categories[category][0]
        
        if self.model_categories["general"]:
            return self.model_categories["general"][0]
        
        for models in self.model_categories.values():
            if models:
                return models[0]
        
        return None
    
    async def route_and_execute(self, prompt: str, task_type: Optional[str] = None, stream: bool = False) -> Dict:
        """Classify task, select model, and execute with fallback on timeout"""
        import time
        start_time = time.time()
        
        if task_type:
            classification = task_type
        else:
            classification = await self.classify_task(prompt)
        
        selected_model = self._select_best_model(classification)
        
        if not selected_model:
            raise HTTPException(status_code=503, detail="No models available")
        
        print(f"\n🎯 Task Classification: {classification}")
        print(f"📍 Selected Model: {selected_model}")
        
        # Try to execute with fallback on timeout
        result = await self._execute_with_fallback(prompt, selected_model, stream, start_time, classification)
        
        return result
    
    async def _execute_with_fallback(self, prompt: str, model: str, stream: bool, start_time: float, classification: str) -> Dict:
        """Execute request with fallback to alternative model on timeout"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": stream
                },
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()["response"]
                execution_time = time.time() - start_time
                
                # Record statistics
                stats.record_request(model, classification, execution_time)
                
                return {
                    "result": result,
                    "model_used": model,
                    "task_classification": classification,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="Model execution failed")
        
        except requests.exceptions.Timeout:
            # Timeout occurred - try fallback model
            fallback_model = MODEL_FALLBACKS.get(model)
            if fallback_model and fallback_model in self.available_models:
                print(f"⚠️ Timeout with model {model}, trying fallback: {fallback_model}")
                return await self._execute_with_fallback(prompt, fallback_model, stream, start_time, classification)
            else:
                print(f"❌ Timeout with model {model} and no fallback available")
                raise HTTPException(status_code=504, detail=f"Model execution timeout. No fallback model configured for {model}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize router
router = OllamaRouter(OLLAMA_BASE_URL, CLASSIFIER_MODEL)

# ============================================================================
# API ENDPOINTS (Port 9998)
# ============================================================================

@api_app.on_event("startup")
async def startup_event():
    """Discover models on startup"""
    print("\n" + "="*70)
    print("🚀 Ollama Intelligent Router Starting...")
    print("="*70)
    print(f"Connecting to Ollama at: {OLLAMA_BASE_URL}")
    await router.discover_models()
    print(f"\n✅ API running at http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"✅ Dashboard at http://{WEB_HOST}:{WEB_PORT}")
    print(f"✅ Ollama at {OLLAMA_BASE_URL}")
    print("="*70 + "\n")

@api_app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "available_models": len(router.available_models),
        "classifier_model": CLASSIFIER_MODEL,
        "ollama_url": OLLAMA_BASE_URL
    }

@api_app.get("/models")
async def list_models():
    """List all discovered models"""
    return {
        "total": len(router.available_models),
        "models": router.available_models,
        "categories": router.model_categories
    }

@api_app.get("/models/{category}")
async def list_models_by_category(category: str):
    """List models in a specific category"""
    if category not in router.model_categories:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    
    return {
        "category": category,
        "models": router.model_categories[category]
    }

@api_app.post("/task", response_model=TaskResponse)
async def process_task(request: TaskRequest):
    """Main endpoint: Process a task with intelligent routing"""
    result = await router.route_and_execute(
        prompt=request.prompt,
        task_type=request.task_type,
        stream=request.stream
    )
    return TaskResponse(**result)

@api_app.post("/classify")
async def classify_only(request: TaskRequest):
    """Just classify a task without executing"""
    classification = await router.classify_task(request.prompt)
    selected_model = router._select_best_model(classification)
    
    return {
        "prompt": request.prompt,
        "classification": classification,
        "recommended_model": selected_model,
        "available_models_in_category": router.model_categories.get(classification, [])
    }

@api_app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if router.available_models else "degraded",
        "models_available": len(router.available_models),
        "categories": {k: len(v) for k, v in router.model_categories.items()},
        "ollama_url": OLLAMA_BASE_URL,
        "classifier_model": CLASSIFIER_MODEL
    }

@api_app.get("/stats")
async def get_statistics():
    """Get usage statistics"""
    return stats.to_dict()

@api_app.get("/config/fallbacks")
async def get_fallbacks():
    """Get model fallback configuration"""
    return {
        "fallbacks": MODEL_FALLBACKS,
        "timeout": REQUEST_TIMEOUT
    }

@api_app.post("/config/fallbacks")
async def update_fallbacks(request: dict):
    """Update model fallback configuration"""
    global MODEL_FALLBACKS
    if "fallbacks" in request:
        MODEL_FALLBACKS.update(request["fallbacks"])
    if "timeout" in request:
        global REQUEST_TIMEOUT
        REQUEST_TIMEOUT = int(request["timeout"])
    return {
        "status": "updated",
        "fallbacks": MODEL_FALLBACKS,
        "timeout": REQUEST_TIMEOUT
    }

# ============================================================================
# WEB DASHBOARD ENDPOINTS (Port 9999)
# ============================================================================

@web_app.on_event("startup")
async def web_startup():
    """Initialize web app"""
    print(f"🌐 Dashboard running at http://{WEB_HOST}:{WEB_PORT}")

@web_app.get("/")
async def serve_dashboard():
    """Serve main dashboard"""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(os.path.join(static_dir, "index.html")):
        return FileResponse(os.path.join(static_dir, "index.html"))
    else:
        return {"error": "Dashboard not found. Ensure static/index.html exists."}

@web_app.get("/api/stats")
async def api_get_stats():
    """API endpoint for statistics (for dashboard)"""
    return stats.to_dict()

@web_app.get("/api/config/fallbacks")
async def web_get_fallbacks():
    """API endpoint for fallback configuration (for dashboard)"""
    return {
        "fallbacks": MODEL_FALLBACKS,
        "timeout": REQUEST_TIMEOUT
    }

@web_app.get("/api/models")
async def api_list_models():
    """API endpoint for models (for dashboard)"""
    return {
        "total": len(router.available_models),
        "models": router.available_models,
        "categories": router.model_categories
    }

@web_app.get("/api/health")
async def api_health():
    """API health check for dashboard"""
    return {
        "status": "healthy" if router.available_models else "degraded",
        "models_available": len(router.available_models),
        "router_url": f"http://{PROXY_HOST}:{PROXY_PORT}",
        "api_status": "running",
        "ollama_url": OLLAMA_BASE_URL
    }

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    web_app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================================
# RUN SERVERS
# ============================================================================

def run_api_server():
    """Run API server on port 9998"""
    uvicorn.run(
        api_app,
        host=PROXY_HOST,
        port=PROXY_PORT,
        log_level="info"
    )

def run_web_server():
    """Run web dashboard on port 9999"""
    uvicorn.run(
        web_app,
        host=WEB_HOST,
        port=WEB_PORT,
        log_level="info"
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Run only web dashboard
        run_web_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run only API
        run_api_server()
    else:
        # Run both servers in separate threads
        print("\n🚀 Starting both API and Web Dashboard servers...\n")
        
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        web_thread = threading.Thread(target=run_web_server, daemon=True)
        
        api_thread.start()
        web_thread.start()
        
        try:
            api_thread.join()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            sys.exit(0)
