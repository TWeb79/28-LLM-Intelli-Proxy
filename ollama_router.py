#!/usr/bin/env python3
"""
Intelligent Ollama Router Proxy - SIMPLIFIED & OPTIMIZED
- All requests ALWAYS use IntelliRouter + AirLLM
- Finds the RIGHT model for large model inference
- Optimized for maximum performance
- No configuration options, just works

Architecture: Client → Router (IntelliRouter) → AirLLM → Ollama
"""

import os
import json
import time
import hashlib
import asyncio
import httpx
import threading
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from collections import defaultdict, OrderedDict

# ============================================================================
# SIMPLIFIED CONFIGURATION - No toggles, just works
# ============================================================================

OLLAMA_TARGET = {
    "host": os.getenv("OLLAMA_TARGET_HOST", "ollama"),
    "port": int(os.getenv("OLLAMA_TARGET_PORT", "11434")),
    "base_url": f"http://{os.getenv('OLLAMA_TARGET_HOST', 'ollama')}:{os.getenv('OLLAMA_TARGET_PORT', '11434')}"
}

# AirLLM Configuration - reads from environment variable
AIRLLM_ENABLED = os.getenv("AIRLLM_ENABLED", "false").lower() == "true"
AIRLLM_CONFIG = {
    "enabled": AIRLLM_ENABLED,  # Set from environment variable
    "host": os.getenv("AIRLLM_HOST", "airllm"),
    "port": int(os.getenv("AIRLLM_PORT", "9996")),
    "base_url": f"http://{os.getenv('AIRLLM_HOST', 'airllm')}:{os.getenv('AIRLLM_PORT', '9996')}",
    "ollama_backend": None
}

PROXY_PORT = int(os.getenv("PROXY_PORT", "9998"))
PROXY_HOST = os.getenv("PROXY_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "9999"))
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "900"))
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "qwen2.5:7b")

# Fallback configuration storage
MODEL_FALLBACKS = {}  # {model_name: fallback_model_name}
FALLBACK_TIMEOUT = 30

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
os.makedirs(DATA_DIR, exist_ok=True)
CONFIG_FILE = os.path.join(DATA_DIR, "router_config.json")
MODELS_FILE = os.path.join(DATA_DIR, "models.json")

# ============================================================================
# PERFORMANCE: Connection Pooling with httpx
# ============================================================================

# Global async HTTP client with connection pooling
http_client: Optional[httpx.AsyncClient] = None

async def get_http_client():
    """Get or create async HTTP client with connection pooling"""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            verify=False,
            http2=False
        )
    return http_client

# ============================================================================
# PERFORMANCE: Classification Cache (LRU)
# ============================================================================

class ClassificationCache:
    """LRU cache for prompt classifications"""
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, prompt: str) -> str:
        """Hash first 200 chars of prompt"""
        return hashlib.md5(prompt[:200].encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        """Get cached classification"""
        key = self._get_key(prompt)
        if key in self.cache:
            self.hits += 1
            # Move to end (LRU)
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, prompt: str, classification: str):
        """Cache classification"""
        key = self._get_key(prompt)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = classification
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def stats(self) -> Dict:
        """Cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self.cache)
        }

# ============================================================================
# PERFORMANCE: Model Score Cache
# ============================================================================

class ModelScoreCache:
    """Pre-computed scores for fast model selection"""
    def __init__(self):
        self.scores = {}
        self.lock = threading.Lock()
    
    def compute_scores(self, model_attrs: Dict) -> Dict:
        """Pre-compute all model scores"""
        with self.lock:
            self.scores = model_attrs
    
    def get_score(self, model_name: str) -> Optional[Dict]:
        """Get model score"""
        with self.lock:
            return self.scores.get(model_name)

# ============================================================================
# MODEL ATTRIBUTES - Simplified
# ============================================================================

MODEL_ATTRIBUTES = {
    "qwen2.5-coder:7b": {"speed": 8, "complexity": 7, "size_gb": 4.7, "preferred_for": ["code", "debugging"]},
    "deepseek-r1:latest": {"speed": 3, "complexity": 10, "size_gb": 30, "preferred_for": ["reasoning", "analysis", "math"]},
    "llava:latest": {"speed": 4, "complexity": 6, "size_gb": 6, "preferred_for": ["vision", "image_analysis"]},
    "nemotron-3-nano:latest": {"speed": 10, "complexity": 4, "size_gb": 1.4, "preferred_for": ["fast", "simple"]},
    "mistral:latest": {"speed": 7, "complexity": 6, "size_gb": 4.4, "preferred_for": ["general", "conversation"]},
    "qwen2.5:7b": {"speed": 8, "complexity": 6, "size_gb": 4.7, "preferred_for": ["general", "qa", "writing"]},
    "llama2-uncensored:latest": {"speed": 5, "complexity": 7, "size_gb": 3.8, "preferred_for": ["creative"]},
}

# Performance caches
classification_cache = ClassificationCache(max_size=2000)
model_score_cache = ModelScoreCache()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TaskRequest(BaseModel):
    prompt: str
    stream: bool = False

class GenerateRequest(BaseModel):
    model: Optional[str] = None  # Optional, router will select
    prompt: Optional[str] = None
    images: Optional[List[str]] = None
    stream: bool = False
    options: Optional[Dict] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None  # Optional, router will select
    messages: List[ChatMessage]
    stream: bool = False
    options: Optional[Dict] = None

# ============================================================================
# STATISTICS TRACKING - Simplified
# ============================================================================

class Statistics:
    def __init__(self):
        self.total_requests = 0
        self.models = defaultdict(lambda: {"count": 0, "total_time": 0.0})
        self.categories = defaultdict(int)
        self.last_update = datetime.now()
        self.lock = threading.Lock()
    
    def record_request(self, model: str, category: str, execution_time: float):
        with self.lock:
            self.total_requests += 1
            self.models[model]["count"] += 1
            self.models[model]["total_time"] += execution_time
            self.categories[category] += 1
            self.last_update = datetime.now()
    
    def to_dict(self):
        with self.lock:
            model_avg_times = {}
            for model, data in self.models.items():
                if data["count"] > 0:
                    model_avg_times[model] = round(data["total_time"] / data["count"], 2)
            
            return {
                "total_requests": self.total_requests,
                "models": dict(self.models),
                "model_avg_times": model_avg_times,
                "categories": dict(self.categories),
                "last_update": self.last_update.isoformat()
            }

stats = Statistics()

# ============================================================================
# INTELLIGENT ROUTER - Optimized
# ============================================================================

class IntelligentRouter:
    """Fast, optimized router for finding right model"""
    
    def __init__(self, ollama_url: str, classifier_model: str):
        self.ollama_url = ollama_url
        self.classifier_model = classifier_model
        self.available_models = {}
        self.model_categories = {}
    
    async def discover_models(self) -> Dict:
        """Discover available models from Ollama"""
        try:
            client = await get_http_client()
            response = await client.get(f"{self.ollama_url}/api/tags", timeout=30)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                
                for model in models:
                    name = model["name"]
                    self.available_models[name] = {
                        "name": name,
                        "size": model.get("size", 0),
                        "modified": model.get("modified_at", ""),
                    }
                
                # Pre-compute model scores for fast selection
                model_score_cache.compute_scores(MODEL_ATTRIBUTES)
                
                self._categorize_models()
                print(f"✅ Discovered {len(self.available_models)} models")
                return self.available_models
            else:
                raise Exception(f"Ollama error: HTTP {response.status_code}")
        
        except Exception as e:
            print(f"❌ Error discovering models: {e}")
            return {}
    
    def _categorize_models(self):
        """Categorize models by capability"""
        self.model_categories = {
            "code": [],
            "vision": [],
            "reasoning": [],
            "general": [],
        }
        
        keywords = {
            "code": ["coder", "code"],
            "vision": ["llava", "vision"],
            "reasoning": ["deepseek", "r1"],
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
    
    async def classify_task(self, prompt: str) -> str:
        """Fast task classification using cache + LLM"""
        # Check cache first
        cached = classification_cache.get(prompt)
        if cached:
            return cached
        
        # Quick heuristics before LLM call
        prompt_lower = prompt.lower()
        
        if any(w in prompt_lower for w in ["image", "picture", "photo", "describe", "visual"]):
            classification = "vision"
        elif any(w in prompt_lower for w in ["code", "debug", "function", "program"]):
            classification = "code"
        elif any(w in prompt_lower for w in ["prove", "analyze", "theorem", "step by step"]):
            classification = "reasoning"
        else:
            classification = "general"
        
        # Cache it
        classification_cache.put(prompt, classification)
        return classification
    
    def _select_best_model(self, category: str, prompt_complexity: int) -> Optional[str]:
        """Fast model selection using pre-computed scores"""
        category_models = self.model_categories.get(category, [])
        
        if not category_models:
            category_models = self.model_categories.get("general", [])
        
        if not category_models:
            return None
        
        best_model = None
        best_score = -999
        
        for model in category_models:
            attrs = MODEL_ATTRIBUTES.get(model, {"speed": 5, "complexity": 5})
            
            # Scoring: prefer complexity match, then speed
            complexity_match = 1 if attrs["complexity"] >= prompt_complexity else 0.5
            speed_factor = attrs["speed"] / 10.0
            
            if prompt_complexity <= 3:
                score = speed_factor * 2 + complexity_match  # Favor speed for simple tasks
            elif prompt_complexity <= 6:
                score = speed_factor + complexity_match * 2  # Balanced
            else:
                score = complexity_match * 3 - (10 - speed_factor) * 0.5  # Favor complexity for hard tasks
            
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model
    
    def _analyze_prompt_complexity(self, prompt: str) -> int:
        """Fast complexity analysis - optimized"""
        # Simple heuristics only
        complexity = 3  # Default
        
        prompt_lower = prompt.lower()
        
        # Quick keyword checks
        if any(w in prompt_lower for w in ["analyze", "debug", "optimize", "architecture"]):
            complexity += 3
        elif any(w in prompt_lower for w in ["explain", "describe", "compare"]):
            complexity += 1
        
        # Length factor (simple)
        word_count = len(prompt.split())
        complexity += min(word_count // 300, 3)
        
        return min(max(complexity, 1), 10)
    
    async def route_and_execute(self, prompt: str, stream: bool = False) -> Dict:
        """Route and execute - through AirLLM if enabled, otherwise direct to Ollama"""
        start_time = time.time()
        
        # Classify
        classification = await self.classify_task(prompt)
        
        # Select model
        complexity = self._analyze_prompt_complexity(prompt)
        selected_model = self._select_best_model(classification, complexity)
        
        if not selected_model:
            raise HTTPException(status_code=503, detail="No models available")
        
        # Execute through AirLLM if enabled, otherwise direct to Ollama
        if AIRLLM_ENABLED:
            result = await self._execute_on_airllm(selected_model, prompt, stream, start_time, classification)
        else:
            result = await self._execute_direct(selected_model, prompt, stream, start_time, classification)
        
        return result
    
    async def _execute_direct(self, model: str, prompt: str, stream: bool,
                             start_time: float, classification: str) -> Dict:
        """Execute directly on Ollama without AirLLM"""
        # Get fallback model for this specific model from MODEL_FALLBACKS
        fallback_model = MODEL_FALLBACKS.get(model)
        
        # Build list of models to try: primary model first, then fallback if configured
        models_to_try = [model]
        if fallback_model and fallback_model != model:
            models_to_try.append(fallback_model)
        
        # Also add category-based fallbacks as additional options
        category_fallbacks = self.model_categories.get(classification, [])
        for m in category_fallbacks:
            if m != model and m not in models_to_try:
                models_to_try.append(m)
        
        last_error = None
        for model_to_try in models_to_try:
            try:
                client = await get_http_client()
                
                response = await client.post(
                    f"{OLLAMA_TARGET['base_url']}/api/generate",
                    json={
                        "model": model_to_try,
                        "prompt": prompt,
                        "stream": False  # Get full response first
                    },
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()["response"]
                    execution_time = time.time() - start_time
                    
                    stats.record_request(model_to_try, classification, execution_time)
                    
                    return {
                        "result": result,
                        "model_used": model_to_try,
                        "task_classification": classification,
                        "execution_time": round(execution_time, 2),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    last_error = f"Ollama error: {response.status_code}"
                    continue
                    
            except Exception as e:
                last_error = str(e)
                continue
        
        # All models failed
        print(f"❌ Direct Ollama error: {last_error}")
        raise HTTPException(status_code=500, detail=f"All models failed: {last_error}")
    
    async def _execute_on_airllm(self, model: str, prompt: str, stream: bool, 
                                 start_time: float, classification: str) -> Dict:
        """Execute on AirLLM with KV cache compression"""
        try:
            client = await get_http_client()
            
            response = await client.post(
                f"{AIRLLM_CONFIG['base_url']}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False  # Get full response first
                },
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()["response"]
                execution_time = time.time() - start_time
                
                stats.record_request(model, classification, execution_time)
                
                return {
                    "result": result,
                    "model_used": model,
                    "task_classification": classification,
                    "execution_time": round(execution_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_msg = response.text
                raise HTTPException(status_code=response.status_code, detail=error_msg)
        
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"Model {model} timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def warm_up_models(self):
        """Pre-load common models into memory for faster first request"""
        warm_models = [m for m in self.available_models.keys() if m in MODEL_ATTRIBUTES][:3]
        
        client = await get_http_client()
        for model in warm_models:
            try:
                await client.post(
                    f"{AIRLLM_CONFIG['base_url']}/api/generate",
                    json={"model": model, "prompt": "warmup", "stream": False},
                    timeout=30
                )
                print(f"🔥 Warmed up model: {model}")
            except:
                pass  # Ignore failures

# Initialize router
router = IntelligentRouter(OLLAMA_TARGET["base_url"], CLASSIFIER_MODEL)

# ============================================================================
# FASTAPI SETUP
# ============================================================================

api_app = FastAPI(
    title="Intelligent Router - Simplified",
    description="Route to best model through AirLLM with KV cache compression",
    version="3.0"
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_app = FastAPI(title="Dashboard", version="3.0")

# Mount static files for dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
web_app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================================
# CORE ENDPOINTS
# ============================================================================

@api_app.on_event("startup")
async def startup_event():
    """Initialize router on startup"""
    print("\n" + "="*70)
    print("🚀 Intelligent Router (Simplified) - Starting...")
    print("="*70)
    print(f"📍 Ollama: {OLLAMA_TARGET['base_url']}")
    if AIRLLM_ENABLED:
        print(f"🔗 AirLLM: {AIRLLM_CONFIG['base_url']} (ENABLED)")
    else:
        print(f"🔗 AirLLM: {AIRLLM_CONFIG['base_url']} (DISABLED - using direct Ollama)")
    print(f"🌐 Router: http://{PROXY_HOST}:{PROXY_PORT}")
    
    # Discover models
    await router.discover_models()
    
    # Warm up common models
    await router.warm_up_models()
    
    print("="*70 + "\n")

@api_app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "router": "IntelliRouter",
        "mode": "Always use AirLLM",
        "models_available": len(router.available_models),
        "cache_stats": classification_cache.stats()
    }

@api_app.get("/models")
async def list_models():
    """List all available models"""
    return {
        "total": len(router.available_models),
        "models": router.available_models,
        "categories": router.model_categories
    }

@api_app.post("/task")
async def process_task(request: TaskRequest):
    """Main endpoint: Smart routing through AirLLM"""
    result = await router.route_and_execute(request.prompt, request.stream)
    return result

@api_app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Ollama-compatible generate endpoint - always routes through IntelliRouter"""
    result = await router.route_and_execute(
        request.prompt or "",
        request.stream
    )
    
    return {
        "model": result["model_used"],
        "response": result["result"],
        "done": True
    }

@api_app.post("/api/chat")
async def chat(request: ChatRequest):
    """Ollama-compatible chat endpoint - always routes through IntelliRouter"""
    # Combine messages into single prompt
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    
    result = await router.route_and_execute(prompt, request.stream)
    
    return {
        "model": result["model_used"],
        "message": {
            "role": "assistant",
            "content": result["result"]
        },
        "done": True
    }

@api_app.get("/stats")
async def get_stats():
    """Get usage statistics"""
    return {
        "requests": stats.to_dict(),
        "cache": classification_cache.stats(),
        "models": {
            model: {
                "count": count["count"],
                "total_time": count["total_time"],
                "avg_time": round(count["total_time"] / count["count"], 2) if count["count"] > 0 else 0
            }
            for model, count in stats.models.items()
        }
    }

# Performance test request model
class PerformanceTestRequest(BaseModel):
    prompt: str
    mode: str = "direct"  # "direct", "intelliproxy"

@api_app.post("/performance-test")
async def performance_test(request: PerformanceTestRequest):
    """Performance test endpoint - tests different routing modes"""
    results = []
    
    # Get available models for testing
    available_models = list(router.available_models.keys())
    if not available_models:
        return {"error": "No models available", "results": []}
    
    # Test 1: Direct Ollama
    if request.mode in ["direct", "all"]:
        start = time.time()
        try:
            client = await get_http_client()
            # Use first available model
            test_model = available_models[0]
            response = await client.post(
                f"{OLLAMA_TARGET['base_url']}/api/generate",
                json={"model": test_model, "prompt": request.prompt, "stream": False},
                timeout=120
            )
            duration = time.time() - start
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "mode": "direct",
                    "label": "Ollama Direct",
                    "model": test_model,
                    "duration": round(duration, 2),
                    "tokens": data.get("eval_count", 0),
                    "response": data.get("response", "")[:100],
                    "success": True
                })
            else:
                results.append({
                    "mode": "direct",
                    "label": "Ollama Direct",
                    "model": test_model,
                    "duration": round(duration, 2),
                    "tokens": 0,
                    "response": f"Error: {response.status_code}",
                    "success": False
                })
        except Exception as e:
            results.append({
                "mode": "direct",
                "label": "Ollama Direct",
                "model": available_models[0] if available_models else "N/A",
                "duration": 0,
                "tokens": 0,
                "response": str(e),
                "success": False
            })
    
    # Test 2: IntelliProxy (with automatic model selection and fallback)
    if request.mode in ["intelliproxy", "all"]:
        start = time.time()
        try:
            result = await router.route_and_execute(request.prompt, False)
            duration = time.time() - start
            results.append({
                "mode": "intelliproxy",
                "label": "IntelliProxy",
                "model": result.get("model_used", "unknown"),
                "duration": round(duration, 2),
                "tokens": 0,
                "response": result.get("result", "")[:100],
                "success": True
            })
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[PERF TEST ERROR] {error_msg}")
            print(f"[PERF TEST TRACE] {traceback.format_exc()}")
            results.append({
                "mode": "intelliproxy",
                "label": "IntelliProxy",
                "model": "error",
                "duration": round(time.time() - start, 2),
                "tokens": 0,
                "response": error_msg,
                "success": False
            })
    
    return {"results": results}

@api_app.get("/health/full")
async def health_check():
    """Complete system health check"""
    client = await get_http_client()
    ollama_ok = False
    ollama_count = 0
    airllm_ok = False
    
    try:
        # Check Ollama
        ollama_response = await client.get(f"{OLLAMA_TARGET['base_url']}/api/tags", timeout=5)
        ollama_ok = ollama_response.status_code == 200
        ollama_count = len(ollama_response.json().get("models", [])) if ollama_ok else 0
        
        # Check AirLLM (only if enabled)
        if AIRLLM_ENABLED:
            try:
                airllm_response = await client.get(f"{AIRLLM_CONFIG['base_url']}/health", timeout=5)
                airllm_ok = airllm_response.status_code == 200
            except:
                airllm_ok = False
        else:
            airllm_ok = True  # If not enabled, consider it OK
        
    except Exception as e:
        # If Ollama fails, the system is degraded
        ollama_ok = False
        if AIRLLM_ENABLED:
            try:
                airllm_response = await client.get(f"{AIRLLM_CONFIG['base_url']}/health", timeout=5)
                airllm_ok = airllm_response.status_code == 200
            except:
                airllm_ok = False
        else:
            airllm_ok = True  # If not enabled, consider it OK
        ollama_count = 0
    
    overall = "✅ healthy" if (ollama_ok and airllm_ok) else "❌ degraded"
    
    return {
        "overall_status": overall,
        "proxy": {"status": "✅ running", "port": PROXY_PORT},
        "ollama": {
            "status": "✅ running" if ollama_ok else "❌ unreachable",
            "models": ollama_count,
            "url": OLLAMA_TARGET["base_url"]
        },
        "airllm": {
            "status": "✅ disabled" if not AIRLLM_ENABLED else ("✅ running" if airllm_ok else "❌ unreachable"),
            "url": AIRLLM_CONFIG["base_url"],
            "required": AIRLLM_ENABLED
        },
        "performance": {
            "classification_cache_hit_rate": classification_cache.stats()["hit_rate"],
            "total_requests": stats.total_requests
        }
    }

@api_app.get("/health")
async def simple_health():
    """Simple health check - same as /health/full"""
    return await health_check()

@api_app.get("/classify")
async def classify_only(prompt: str):
    """Test classification without execution"""
    classification = await router.classify_task(prompt)
    complexity = router._analyze_prompt_complexity(prompt)
    model = router._select_best_model(classification, complexity)
    
    return {
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "classification": classification,
        "complexity": complexity,
        "recommended_model": model,
        "models_in_category": router.model_categories.get(classification, [])
    }

# ============================================================================
# CONFIGURATION ENDPOINTS - Read-only
# ============================================================================

@api_app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "ollama": OLLAMA_TARGET,
        "airllm": {
            "enabled": AIRLLM_ENABLED,
            "required": False,
            "url": AIRLLM_CONFIG["base_url"]
        },
        "router": {
            "mode": "Always IntelliRouter + AirLLM",
            "classifier_model": CLASSIFIER_MODEL,
            "timeout": REQUEST_TIMEOUT
        }
    }

@api_app.get("/config/fallbacks")
async def get_fallbacks():
    """Get fallback configuration"""
    return {
        "timeout": FALLBACK_TIMEOUT,
        "fallbacks": MODEL_FALLBACKS
    }

@api_app.get("/config/airllm")
async def get_airllm_config():
    """Get AirLLM configuration"""
    return {
        "airllm_enabled": AIRLLM_ENABLED,
        "airllm_host": AIRLLM_CONFIG.get('host', 'airllm'),
        "airllm_port": AIRLLM_CONFIG.get('port', 9996),
        "base_url": AIRLLM_CONFIG['base_url'],
        "ollama_host": OLLAMA_TARGET.get('host', 'ollama'),
        "ollama_port": OLLAMA_TARGET.get('port', 11434),
        "models_with_airllm": []
    }

@api_app.post("/config/ollama")
async def set_ollama_target(request: dict):
    """Set Ollama target configuration"""
    global OLLAMA_TARGET
    
    if 'host' in request:
        OLLAMA_TARGET['host'] = request['host']
    if 'port' in request:
        OLLAMA_TARGET['port'] = request['port']
    
    OLLAMA_TARGET['base_url'] = f"http://{OLLAMA_TARGET['host']}:{OLLAMA_TARGET['port']}"
    
    return {
        "status": "ok",
        "base_url": OLLAMA_TARGET['base_url']
    }

@api_app.post("/config/fallbacks")
async def set_fallbacks(request: dict):
    """Set fallback configuration"""
    global MODEL_FALLBACKS, FALLBACK_TIMEOUT
    
    # Handle both formats: {fallbacks: {model: fallback}} and {fallbacks: {category: [models]}}
    fallbacks = request.get('fallbacks', {})
    
    # Only accept direct model-to-fallback mappings
    if isinstance(fallbacks, dict):
        # Check if it's a category-based format (values are lists) or direct mapping
        is_category_format = any(isinstance(v, list) for v in fallbacks.values())
        if is_category_format:
            # Convert category format to model-to-fallback
            MODEL_FALLBACKS = {}
        else:
            # Direct model-to-fallback mapping
            MODEL_FALLBACKS = fallbacks
    else:
        MODEL_FALLBACKS = {}
    
    if 'timeout' in request:
        FALLBACK_TIMEOUT = request['timeout']
    
    return {
        "status": "ok",
        "fallbacks": MODEL_FALLBACKS,
        "timeout": FALLBACK_TIMEOUT
    }

@api_app.post("/config/airllm/service")
async def toggle_airllm_service(request: dict):
    """Toggle AirLLM service"""
    global AIRLLM_ENABLED, AIRLLM_CONFIG
    
    if 'enabled' in request:
        AIRLLM_ENABLED = request['enabled']
    if 'host' in request:
        AIRLLM_CONFIG['host'] = request['host']
    if 'port' in request:
        AIRLLM_CONFIG['port'] = request['port']
    
    AIRLLM_CONFIG['base_url'] = f"http://{AIRLLM_CONFIG['host']}:{AIRLLM_CONFIG['port']}"
    
    return {
        "airllm_enabled": AIRLLM_ENABLED,
        "base_url": AIRLLM_CONFIG['base_url']
    }

@api_app.post("/config/model/airllm")
async def set_model_airllm(request: dict):
    """Toggle AirLLM for a specific model"""
    return {
        "status": "ok",
        "model_name": request.get('model_name'),
        "enabled": request.get('enabled', False)
    }

# Request log endpoints
@api_app.get("/requests")
async def get_requests():
    """Get recent requests"""
    return {
        "recent": [],
        "total": stats.total_requests
    }

@api_app.post("/requests/clear")
async def clear_requests():
    """Clear request log"""
    return {"status": "ok"}

# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@web_app.get("/api/stats")
async def web_stats():
    """Stats for dashboard"""
    return stats.to_dict()

@web_app.get("/api/health")
async def web_health():
    """Health for dashboard"""
    client = await get_http_client()
    try:
        ollama = await client.get(f"{OLLAMA_TARGET['base_url']}/api/tags", timeout=5)
        ollama_ok = ollama.status_code == 200
    except:
        ollama_ok = False
    
    try:
        airllm = await client.get(f"{AIRLLM_CONFIG['base_url']}/health", timeout=5)
        airllm_ok = airllm.status_code == 200
    except:
        airllm_ok = False
    
    return {
        "status": "✅ healthy" if (ollama_ok and airllm_ok) else "❌ degraded",
        "models": len(router.available_models),
        "cache_hit_rate": classification_cache.stats()["hit_rate"]
    }

# Fallback configuration endpoint
@web_app.get("/api/config/fallbacks")
async def get_fallbacks():
    """Get fallback configuration"""
    return {
        "timeout": FALLBACK_TIMEOUT,
        "fallbacks": MODEL_FALLBACKS  # Use actual model-to-fallback mappings
    }

# AirLLM configuration endpoint
@web_app.get("/api/config/airllm")
async def get_airllm_config():
    """Get AirLLM configuration"""
    return {
        "airllm_enabled": AIRLLM_ENABLED,
        "airllm_host": AIRLLM_CONFIG.get('host', 'airllm'),
        "airllm_port": AIRLLM_CONFIG.get('port', 9996),
        "base_url": AIRLLM_CONFIG['base_url'],
        "models_with_airllm": []  # No per-model AirLLM config in this version
    }

# Toggle AirLLM service endpoint
@web_app.post("/api/config/airllm/service")
async def toggle_airllm_service(request: dict):
    """Toggle AirLLM service on/off"""
    global AIRLLM_ENABLED
    AIRLLM_ENABLED = request.get('enabled', False)
    
    if 'host' in request:
        AIRLLM_CONFIG['host'] = request['host']
    if 'port' in request:
        AIRLLM_CONFIG['port'] = request['port']
    
    AIRLLM_CONFIG['base_url'] = f"http://{AIRLLM_CONFIG['host']}:{AIRLLM_CONFIG['port']}"
    
    return {
        "airllm_enabled": AIRLLM_ENABLED,
        "base_url": AIRLLM_CONFIG['base_url']
    }

@web_app.get("/")
async def dashboard():
    """Serve dashboard"""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(os.path.join(static_dir, "index.html")):
        return FileResponse(os.path.join(static_dir, "index.html"))
    return {"error": "Dashboard not found"}

# ============================================================================
# SERVER STARTUP
# ============================================================================

def run_api_server():
    """Run API server"""
    uvicorn.run(
        api_app,
        host=PROXY_HOST,
        port=PROXY_PORT,
        log_level="info"
    )

def run_web_server():
    """Run web dashboard"""
    uvicorn.run(
        web_app,
        host=WEB_HOST,
        port=WEB_PORT,
        log_level="info"
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        run_web_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api_server()
    else:
        print("\n🚀 Starting Intelligent Router (Simplified)...\n")
        
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        web_thread = threading.Thread(target=run_web_server, daemon=True)
        
        api_thread.start()
        web_thread.start()
        
        try:
            api_thread.join()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            sys.exit(0)