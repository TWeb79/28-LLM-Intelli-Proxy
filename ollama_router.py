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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from collections import defaultdict
import threading

# Model attributes configuration
# Each model can have:
# - speed: 1-10 (higher is faster)
# - complexity: 1-10 (higher can handle more complex tasks)
# - preferred_for: list of task types it's best at
MODEL_ATTRIBUTES = {
    "qwen2.5-coder:7b": {"speed": 8, "complexity": 7, "preferred_for": ["code", "debugging", "technical"]},
    "deepseek-r1:latest": {"speed": 3, "complexity": 10, "preferred_for": ["reasoning", "analysis", "math"]},
    "llava:latest": {"speed": 4, "complexity": 6, "preferred_for": ["vision", "image_analysis"]},
    "nemotron-3-nano:latest": {"speed": 10, "complexity": 4, "preferred_for": ["simple", "fast", "basic"]},
    "mistral:latest": {"speed": 7, "complexity": 6, "preferred_for": ["general", "conversation"]},
    "goonsai/qwen2.5-3B-goonsai-nsfw-100k:latest": {"speed": 9, "complexity": 5, "preferred_for": ["image", "generate", "creative"]},
    "llama2-uncensored:latest": {"speed": 5, "complexity": 7, "preferred_for": ["uncensored", "creative"]},
    "qwen2.5:7b": {"speed": 8, "complexity": 6, "preferred_for": ["general", "qa", "writing"]},
}

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")  # Default to docker internal
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "qwen2.5:7b")
PROXY_PORT = int(os.getenv("PROXY_PORT", "9998"))  # Changed from 8000 to 9998
PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "9999"))
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "900"))  # Default 15 minutes timeout
ENABLE_AUTO_DISCOVERY = os.getenv("ENABLE_AUTO_DISCOVERY", "true").lower() == "true"

# ============================================================================
# AirLLM CONFIGURATION - Decentralized Ollama + AirLLM Support
# ============================================================================

# Target Ollama connection (for decentralized Ollama)
# This allows connecting to a remote Ollama instance instead of local
OLLAMA_TARGET = {
    "host": os.getenv("OLLAMA_TARGET_HOST", "ollama"),
    "port": int(os.getenv("OLLAMA_TARGET_PORT", "11434")),
    "base_url": f"http://{os.getenv('OLLAMA_TARGET_HOST', 'ollama')}:{os.getenv('OLLAMA_TARGET_PORT', '11434')}"
}

# AirLLM Service Configuration
# AirLLM provides KV cache compression for large models (70B+)
AIRLLM_CONFIG = {
    "enabled": os.getenv("AIRLLM_ENABLED", "false").lower() == "true",
    "host": os.getenv("AIRLLM_HOST", "airllm"),
    "port": int(os.getenv("AIRLLM_PORT", "9996")),
    "base_url": f"http://{os.getenv('AIRLLM_HOST', 'airllm')}:{os.getenv('AIRLLM_PORT', '9996')}"
}

# Model-specific AirLLM configuration
# Set to True to route requests for this model through AirLLM
MODEL_AIRLLM_CONFIG = {}

# ============================================================================
# PERSISTENT CONFIGURATION STORAGE
# ============================================================================

import os
from pathlib import Path

# Config directory - use /app/data for Docker, local directory otherwise
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
CONFIG_FILE = DATA_DIR / "router_config.json"
MODELS_FILE = DATA_DIR / "models.json"

def ensure_data_dir():
    """Ensure data directory exists"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Could not create data directory: {e}")

def load_router_config() -> dict:
    """Load router configuration from JSON file"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                print(f"✅ Loaded router config from {CONFIG_FILE}")
                return config
    except json.JSONDecodeError as e:
        print(f"⚠️ Config file corrupted ({e}), will rebuild from defaults")
    except Exception as e:
        print(f"⚠️ Error loading config: {e}")
    
    return {}

def save_router_config(config: dict):
    """Save router configuration to JSON file"""
    try:
        ensure_data_dir()
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved router config to {CONFIG_FILE}")
    except Exception as e:
        print(f"⚠️ Error saving config: {e}")

def load_models_config() -> dict:
    """Load models configuration from JSON file"""
    try:
        if MODELS_FILE.exists():
            with open(MODELS_FILE, 'r') as f:
                models_config = json.load(f)
                print(f"✅ Loaded models config from {MODELS_FILE}")
                return models_config
    except json.JSONDecodeError as e:
        print(f"⚠️ Models file corrupted ({e}), will rebuild from Ollama")
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
    
    return {"models": {}, "model_attributes": {}}

def save_models_config(models_config: dict):
    """Save models configuration to JSON file"""
    try:
        ensure_data_dir()
        with open(MODELS_FILE, 'w') as f:
            json.dump(models_config, f, indent=2)
        print(f"✅ Saved models config to {MODELS_FILE}")
    except Exception as e:
        print(f"⚠️ Error saving models config: {e}")

def initialize_configs():
    """Initialize configurations from files or defaults"""
    global OLLAMA_TARGET, AIRLLM_CONFIG, MODEL_AIRLLM_CONFIG, MODEL_FALLBACKS, MODEL_ATTRIBUTES
    
    # Load router config
    router_config = load_router_config()
    
    # Apply saved Ollama target
    if "ollama_target" in router_config:
        OLLAMA_TARGET = router_config["ollama_target"]
    
    # Apply saved AirLLM config
    if "airllm_config" in router_config:
        AIRLLM_CONFIG = router_config["airllm_config"]
    
    # Apply saved model AirLLM config
    if "model_airllm" in router_config:
        MODEL_AIRLLM_CONFIG = router_config["model_airllm"]
    
    # Apply saved fallbacks
    if "fallbacks" in router_config:
        MODEL_FALLBACKS.update(router_config["fallbacks"])
    
    # Load models config
    models_config = load_models_config()
    
    # Apply saved model attributes
    if "model_attributes" in models_config:
        MODEL_ATTRIBUTES.update(models_config["model_attributes"])
    
    print(f"📋 Loaded {len(MODEL_ATTRIBUTES)} model attributes")

def update_ollama_target_config(host: str, port: int):
    """Update and persist Ollama target configuration"""
    global OLLAMA_TARGET
    OLLAMA_TARGET["host"] = host
    OLLAMA_TARGET["port"] = port
    OLLAMA_TARGET["base_url"] = f"http://{host}:{port}"
    
    # Save to config file
    router_config = load_router_config()
    router_config["ollama_target"] = OLLAMA_TARGET
    save_router_config(router_config)

def update_airllm_config(enabled: bool, host: str, port: int):
    """Update and persist AirLLM service configuration"""
    global AIRLLM_CONFIG
    AIRLLM_CONFIG["enabled"] = enabled
    AIRLLM_CONFIG["host"] = host
    AIRLLM_CONFIG["port"] = port
    AIRLLM_CONFIG["base_url"] = f"http://{host}:{port}"
    
    # Save to config file
    router_config = load_router_config()
    router_config["airllm_config"] = AIRLLM_CONFIG
    save_router_config(router_config)

def persist_model_airllm_config(model_name: str, enabled: bool):
    """Update and persist model AirLLM configuration"""
    global MODEL_AIRLLM_CONFIG
    MODEL_AIRLLM_CONFIG[model_name] = enabled
    
    # Save to config file
    router_config = load_router_config()
    router_config["model_airllm"] = MODEL_AIRLLM_CONFIG
    save_router_config(router_config)

def update_model_attribute(model_name: str, attributes: dict):
    """Update and persist model attributes"""
    global MODEL_ATTRIBUTES
    MODEL_ATTRIBUTES[model_name] = attributes
    
    # Save to models config file
    models_config = load_models_config()
    models_config["model_attributes"] = MODEL_ATTRIBUTES
    save_models_config(models_config)

def update_fallback_config(fallbacks: dict):
    """Update and persist fallback configuration"""
    global MODEL_FALLBACKS
    MODEL_FALLBACKS.update(fallbacks)
    
    # Save to config file
    router_config = load_router_config()
    router_config["fallbacks"] = MODEL_FALLBACKS
    save_router_config(router_config)

def get_model_endpoint(model_name: str) -> str:
    """
    Determine which endpoint to use based on model configuration.
    If AirLLM is enabled for the model, use AirLLM service.
    Otherwise, use the configured Ollama target.
    """
    if AIRLLM_CONFIG["enabled"] and MODEL_AIRLLM_CONFIG.get(model_name, False):
        return AIRLLM_CONFIG["base_url"]
    else:
        return OLLAMA_TARGET["base_url"]

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

# Ollama-compatible request/response models
class GenerateRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    images: Optional[List[str]] = None
    stream: bool = False
    options: Optional[Dict] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None

class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool = True

class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    options: Optional[Dict] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None

class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool = True

class ModelInfo(BaseModel):
    name: str
    modified_at: str
    size: int
    description: Optional[str] = None

# AirLLM Configuration Models
class OllamaTargetConfig(BaseModel):
    host: str
    port: int

class AirLLMServiceConfig(BaseModel):
    enabled: bool
    host: str
    port: int

class ModelAirLLMConfig(BaseModel):
    model_name: str
    enabled: bool

class AirLLMConfigResponse(BaseModel):
    ollama_host: str
    ollama_port: int
    airllm_enabled: bool
    airllm_host: str
    airllm_port: int
    model_airllm_settings: Dict[str, bool]

# Statistics tracking
class Statistics:
    def __init__(self):
        self.total_requests = 0
        self.models = defaultdict(lambda: {"count": 0, "total_time": 0.0})
        self.categories = defaultdict(int)
        self.last_update = datetime.now()
        self.lock = threading.Lock()
        self.recent_requests = []  # Last 50 requests for debug
        self.max_recent = 50
    
    def record_request(self, model: str, category: str, execution_time: float = 0.0, prompt: str = ""):
        with self.lock:
            self.total_requests += 1
            self.models[model]["count"] += 1
            self.models[model]["total_time"] += execution_time
            self.categories[category] += 1
            self.last_update = datetime.now()
            
            # Add to recent requests (with prompt logged)
            self.recent_requests.append({
                "model_used": model,
                "task_classification": category,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,  # Store truncated prompt
                "execution_time": round(execution_time, 2),
                "timestamp": self.last_update.isoformat()
            })
            
            # Keep only last 50 requests
            if len(self.recent_requests) > self.max_recent:
                self.recent_requests = self.recent_requests[-self.max_recent:]
    
    def to_dict(self):
        with self.lock:
            # Calculate average response time per model
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
    
    def get_recent(self):
        with self.lock:
            return list(self.recent_requests)
    
    def clear_recent(self):
        with self.lock:
            self.recent_requests = []

# Initialize statistics
stats = Statistics()

# Initialize FastAPI apps
api_app = FastAPI(
    title="Ollama Intelligent Router",
    description="Intelligent task routing proxy for Ollama models",
    version="1.0"
)

# Add CORS middleware to allow dashboard on different port to access API
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
                    
                    # Auto-discover and save attributes for new models
                    if name not in MODEL_ATTRIBUTES:
                        print(f"🔍 New model detected: {name}")
                        if ENABLE_AUTO_DISCOVERY:
                            try:
                                attrs = await self.discover_model_attributes(name)
                                MODEL_ATTRIBUTES[name] = attrs
                                # Persist the new model attributes
                                update_model_attribute(name, attrs)
                            except Exception as e:
                                print(f"⚠️ Failed to discover attributes for {name}: {e}")
                                default_attrs = {"speed": 5, "complexity": 5, "preferred_for": ["general"]}
                                MODEL_ATTRIBUTES[name] = default_attrs
                                update_model_attribute(name, default_attrs)
                
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
            "image": [],
            "vision": [],
            "reasoning": [],
            "general": [],
            "uncensored": [],
        }
        
        keywords = {
            "code": ["coder", "code"],
            "image": ["goonsai", "image", "generate"],
            "vision": ["llava", "vision"],
            "reasoning": ["deepseek", "r1"],
            "uncensored": ["uncensored", "nsfw"],
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

    async def discover_model_attributes(self, model_name: str) -> Dict:
        """
        Automatically discover attributes for an unknown model by asking an LLM.
        Uses the classifier model to analyze and return speed, complexity, and preferred_for.
        """
        if not ENABLE_AUTO_DISCOVERY:
            return {"speed": 5, "complexity": 5, "preferred_for": ["general"]}
        
        analysis_prompt = f"""Analyze this Ollama model name and respond with a JSON object containing its attributes.

Model name: {model_name}

Respond ONLY with valid JSON in this exact format, nothing else:
{{"speed": <1-10>, "complexity": <1-10>, "preferred_for": ["category1", "category2"]}}

Rules:
- speed: 1=slowest, 10=fastest (estimate based on model size and name)
- complexity: 1=simple tasks only, 10=can handle very complex tasks
- preferred_for: array of categories this model is good for (code, reasoning, vision, general, image, uncensored, conversation, writing, analysis)

Consider:
- Smaller models (1-4B params) are faster but less capable
- Larger models (7B+) are slower but more capable
- Code-specific models have "code" or "coder" in name
- Vision models have "vision", "llava", or "vision" in name
- Reasoning models have "r1", "reasoning", or "math" in name
- Uncensored models often have "uncensored" or similar in name

JSON:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": CLASSIFIER_MODEL,
                    "prompt": analysis_prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "").strip()
                
                # Try to parse JSON from response
                try:
                    # Find JSON in response (might have extra text)
                    import re
                    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                    if json_match:
                        attrs = json.loads(json_match.group())
                        
                        # Validate and normalize
                        speed = max(1, min(10, int(attrs.get("speed", 5))))
                        complexity = max(1, min(10, int(attrs.get("complexity", 5))))
                        preferred_for = attrs.get("preferred_for", ["general"])
                        
                        print(f"🔍 Auto-discovered attributes for {model_name}: speed={speed}, complexity={complexity}")
                        
                        return {
                            "speed": speed,
                            "complexity": complexity,
                            "preferred_for": preferred_for
                        }
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"⚠️ Failed to parse model attributes for {model_name}: {e}")
                    
        except Exception as e:
            print(f"⚠️ Error discovering model attributes for {model_name}: {e}")
        
        # Default attributes if discovery fails
        return {"speed": 5, "complexity": 5, "preferred_for": ["general"]}
    
    async def classify_task(self, prompt: str) -> str:
        """Use lightweight LLM to classify the task"""
        
        # First, check for image-related keywords directly
        image_keywords = ["image", "picture", "photo", "describe", "analyze", "what is in", 
                        "visual", "see", "look at", "generate image", "create image"]
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in image_keywords):
            return "image"
        
        classification_prompt = f"""Analyze this task and respond with ONLY one category:

Task: "{prompt}"

Categories:
- code: Writing, debugging, or analyzing code
- image: Generating, describing, or processing images (NOT vision tasks with existing images)
- vision: Analyzing, describing existing images or visual content
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
                valid_categories = ["code", "image", "vision", "reasoning", "uncensored", "general"]
                if classification in valid_categories:
                    return classification
                else:
                    return "general"
            else:
                return "general"
        
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return "general"
    
    def _select_best_model(self, category: str, prompt: str = "") -> Optional[str]:
        """Select best model for category based on prompt complexity"""
        
        # Analyze prompt complexity
        prompt_complexity = self._analyze_prompt_complexity(prompt)
        
        # Get models in category
        category_models = []
        if category in self.model_categories and self.model_categories[category]:
            category_models = self.model_categories[category]
        elif self.model_categories["general"]:
            category_models = self.model_categories["general"]
        else:
            # Fallback to any available model
            for models in self.model_categories.values():
                if models:
                    category_models = models
                    break
        
        if not category_models:
            return None
        
        # Score each model based on attributes and prompt complexity
        best_model = None
        best_score = -1
        
        for model in category_models:
            # Get model attributes, auto-discover if unknown
            if model not in MODEL_ATTRIBUTES:
                if ENABLE_AUTO_DISCOVERY and model in self.available_models:
                    # Trigger async discovery (will be cached in MODEL_ATTRIBUTES)
                    import asyncio
                    try:
                        attrs = asyncio.run(self.discover_model_attributes(model))
                        MODEL_ATTRIBUTES[model] = attrs
                    except:
                        attrs = {"speed": 5, "complexity": 5, "preferred_for": ["general"]}
                else:
                    attrs = {"speed": 5, "complexity": 5, "preferred_for": ["general"]}
            else:
                attrs = MODEL_ATTRIBUTES.get(model, {"speed": 5, "complexity": 5, "preferred_for": []})
            
            # Score: prefer model where complexity >= prompt_complexity (can handle it)
            # and speed is high if prompt is simple
            complexity_match = 1 if attrs["complexity"] >= prompt_complexity else 0
            speed_score = attrs["speed"] / 10.0
            
            # If model can handle the complexity, prefer faster models for simple prompts
            if complexity_match:
                if prompt_complexity <= 3:
                    score = speed_score * 2 + 1  # Favor speed for simple tasks
                elif prompt_complexity <= 6:
                    score = speed_score + complexity_match  # Balanced
                else:
                    score = complexity_match * 2 - speed_score  # Favor capability for complex
            else:
                score = -1  # Cannot handle
            
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model
    
    def _analyze_prompt_complexity(self, prompt: str) -> int:
        """Analyze prompt complexity (1-10)"""
        prompt_lower = prompt.lower()
        
        # Simple indicators
        simple_keywords = ["hello", "hi", "hey", "simple", "quick", "what is", "who is"]
        medium_keywords = ["explain", "compare", "describe", "write", "create", "make"]
        complex_keywords = ["analyze", "debug", "optimize", "architecture", "design", "implement", 
                          "research", "complex", "detailed", "comprehensive", "math", "proof"]
        
        simple_count = sum(1 for kw in simple_keywords if kw in prompt_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in prompt_lower)
        complex_count = sum(1 for kw in complex_keywords if kw in prompt_lower)
        
        # Length factor
        word_count = len(prompt.split())
        length_factor = min(word_count / 100, 3)  # Up to 3 points for length
        
        # Calculate base complexity
        base = 3  # Default middle complexity
        base += medium_count * 1.5
        base += complex_count * 2.5
        base += length_factor
        
        return min(max(int(base), 1), 10)  # Clamp to 1-10
    
    async def route_and_execute(self, prompt: str, task_type: Optional[str] = None, stream: bool = False) -> Dict:
        """Classify task, select model, and execute with fallback on timeout"""
        import time
        start_time = time.time()
        
        if task_type:
            classification = task_type
        else:
            classification = await self.classify_task(prompt)
        
        selected_model = self._select_best_model(classification, prompt)
        
        if not selected_model:
            raise HTTPException(status_code=503, detail="No models available")
        
        print(f"\n🎯 Task Classification: {classification}")
        print(f"📍 Selected Model: {selected_model}")
        
        # Try to execute with fallback on timeout
        result = await self._execute_with_fallback(prompt, selected_model, stream, start_time, classification)
        
        return result
    
    async def _execute_with_fallback(self, prompt: str, model: str, stream: bool, start_time: float, classification: str) -> Dict:
        """Execute request with fallback to alternative model on timeout or error"""
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
                stats.record_request(model, classification, execution_time, prompt)
                
                return {
                    "result": result,
                    "model_used": model,
                    "task_classification": classification,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Non-200 status - check for error message
                error_detail = "Model execution failed"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_detail = error_data["error"]
                except:
                    pass
                
                # Check if error is due to memory or model loading issues - try fallback
                memory_keywords = ["memory", "requires more", "not available", "failed to load"]
                if any(keyword in error_detail.lower() for keyword in memory_keywords):
                    fallback_model = MODEL_FALLBACKS.get(model)
                    if fallback_model and fallback_model in self.available_models:
                        print(f"⚠️ Model {model} failed ({error_detail}), trying fallback: {fallback_model}")
                        return await self._execute_with_fallback(prompt, fallback_model, stream, start_time, classification)
                    else:
                        print(f"❌ Model {model} failed and no fallback available: {error_detail}")
                
                raise HTTPException(status_code=response.status_code, detail=error_detail)
        
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
            error_msg = str(e)
            # Check if error is due to memory or model loading issues - try fallback
            memory_keywords = ["memory", "requires more", "not available", "failed to load"]
            if any(keyword in error_msg.lower() for keyword in memory_keywords):
                fallback_model = MODEL_FALLBACKS.get(model)
                if fallback_model and fallback_model in self.available_models:
                    print(f"⚠️ Model {model} failed ({error_msg}), trying fallback: {fallback_model}")
                    return await self._execute_with_fallback(prompt, fallback_model, stream, start_time, classification)
                else:
                    print(f"❌ Model {model} failed and no fallback available: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

# Initialize router
router = OllamaRouter(OLLAMA_TARGET["base_url"], CLASSIFIER_MODEL)

# ============================================================================
# API ENDPOINTS (Port 9998)
# ============================================================================

@api_app.on_event("startup")
async def startup_event():
    """Discover models on startup"""
    print("\n" + "="*70)
    print("🚀 Ollama Intelligent Router Starting...")
    print("="*70)
    
    # Load persistent configurations
    print("📂 Loading configurations...")
    initialize_configs()
    
    print(f"Connecting to Ollama at: {OLLAMA_TARGET['base_url']}")
    await router.discover_models()
    
    print(f"\n✅ API running at http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"✅ Dashboard at http://{WEB_HOST}:{WEB_PORT}")
    print(f"✅ Ollama at {OLLAMA_TARGET['base_url']}")
    print(f"✅ AirLLM: {'Enabled' if AIRLLM_CONFIG['enabled'] else 'Disabled'} at {AIRLLM_CONFIG['base_url']}")
    print("="*70 + "\n")

@api_app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "available_models": len(router.available_models),
        "classifier_model": CLASSIFIER_MODEL,
        "ollama_url": OLLAMA_TARGET["base_url"]
    }

@api_app.get("/models")
async def list_models():
    """List all discovered models"""
    # Add attributes to each model
    models_with_attrs = {}
    for name, info in router.available_models.items():
        models_with_attrs[name] = {
            **info,
            **MODEL_ATTRIBUTES.get(name, {"speed": 5, "complexity": 5, "preferred_for": []})
        }
    
    return {
        "total": len(router.available_models),
        "models": models_with_attrs,
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
    selected_model = router._select_best_model(classification, request.prompt)
    
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
        "ollama_url": OLLAMA_TARGET["base_url"],
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
    if "fallbacks" in request:
        update_fallback_config(request["fallbacks"])
    if "timeout" in request:
        global REQUEST_TIMEOUT
        REQUEST_TIMEOUT = int(request["timeout"])
    return {
        "status": "updated",
        "fallbacks": MODEL_FALLBACKS,
        "timeout": REQUEST_TIMEOUT
    }

# ============================================================================
# AIRLLM CONFIGURATION ENDPOINTS
# ============================================================================

@api_app.get("/config/airllm")
async def get_airllm_config():
    """Get AirLLM and Ollama target configuration"""
    return {
        "ollama_host": OLLAMA_TARGET["host"],
        "ollama_port": OLLAMA_TARGET["port"],
        "airllm_enabled": AIRLLM_CONFIG["enabled"],
        "airllm_host": AIRLLM_CONFIG["host"],
        "airllm_port": AIRLLM_CONFIG["port"],
        "model_airllm_settings": MODEL_AIRLLM_CONFIG
    }

@api_app.post("/config/ollama")
async def update_ollama_target(config: OllamaTargetConfig):
    """Update the target Ollama IP and port for decentralized connections"""
    update_ollama_target_config(config.host, config.port)
    print(f"📍 Ollama target updated to: {OLLAMA_TARGET['base_url']}")
    return {
        "status": "updated",
        "ollama_host": OLLAMA_TARGET["host"],
        "ollama_port": OLLAMA_TARGET["port"],
        "base_url": OLLAMA_TARGET["base_url"]
    }

@api_app.post("/config/airllm/service")
async def update_airllm_service(config: AirLLMServiceConfig):
    """Update AirLLM service configuration"""
    update_airllm_config(config.enabled, config.host, config.port)
    print(f"🚀 AirLLM service {'enabled' if config.enabled else 'disabled'}: {AIRLLM_CONFIG['base_url']}")
    return {
        "status": "updated",
        "airllm_enabled": AIRLLM_CONFIG["enabled"],
        "airllm_host": AIRLLM_CONFIG["host"],
        "airllm_port": AIRLLM_CONFIG["port"],
        "base_url": AIRLLM_CONFIG["base_url"]
    }

@api_app.post("/config/model/airllm")
async def update_model_airllm(config: ModelAirLLMConfig):
    """Enable or disable AirLLM for a specific model"""
    persist_model_airllm_config(config.model_name, config.enabled)
    status = "enabled" if config.enabled else "disabled"
    print(f"✈️ AirLLM {status} for model: {config.model_name}")
    return {
        "status": "updated",
        "model_name": config.model_name,
        "airllm_enabled": config.enabled,
        "model_airllm_settings": MODEL_AIRLLM_CONFIG
    }

@api_app.post("/models/refresh")
async def refresh_models():
    """Force refresh model list from Ollama and save to models.json"""
    global MODEL_ATTRIBUTES
    
    # Discover models from Ollama
    discovered = await router.discover_models()
    
    # Get list of available models
    available_models = router.available_models
    
    # Check for new models not in models.json
    new_models = []
    for model_name in available_models.keys():
        if model_name not in MODEL_ATTRIBUTES:
            new_models.append(model_name)
    
    # Save to models.json
    models_config = load_models_config()
    models_config["model_attributes"] = MODEL_ATTRIBUTES
    save_models_config(models_config)
    
    print(f"🔄 Refreshed models from Ollama. Found {len(available_models)} models, {len(new_models)} new")
    
    return {
        "status": "success",
        "total_models": len(available_models),
        "new_models": new_models,
        "all_models": list(available_models.keys())
    }

@api_app.get("/requests")
async def get_requests():
    """Get recent request log (last 50)"""
    return {
        "recent": stats.get_recent()
    }

@api_app.post("/requests/clear")
async def clear_requests():
    """Clear the request log"""
    stats.clear_recent()
    return {"status": "cleared"}

# ============================================================================
# Performance Test Endpoint
# ============================================================================

import time

def get_model_endpoint_test(model_name: str, use_airllm: bool = False, use_proxy: bool = False) -> str:
    """
    Get the endpoint for a performance test.
    
    Parameters:
    - model_name: The model to use
    - use_airllm: Whether to use AirLLM compression
    - use_proxy: Whether to route through the IntelliProxy (port 9998)
    
    Returns the appropriate endpoint URL.
    """
    if use_proxy:
        # Route through IntelliProxy API (port 9998)
        return f"http://{PROXY_HOST}:{PROXY_PORT}"
    elif use_airllm and AIRLLM_CONFIG.get("enabled", False):
        # Go directly to AirLLM service (not through proxy)
        return AIRLLM_CONFIG.get("base_url", "http://airllm:9996")
    else:
        # Direct to configured Ollama target
        return OLLAMA_TARGET["base_url"]

async def run_performance_test(prompt: str, mode: str, test_number: int = 1, total_tests: int = 4) -> dict:
    """
    Run a performance test for a specific mode.
    Modes:
    - "direct": Direct to configured Ollama (ignores AirLLM, goes to configured target)
    - "direct_airllm": Through proxy with AirLLM enabled (port 9998)
    - "llm": Use IntelliProxy for routing (port 9998, model=IntelliProxyLLM)
    - "llm_airllm": IntelliProxy + AirLLM (port 9998)
    
    Parameters:
    - test_number: Current test number (1-based)
    - total_tests: Total number of tests
    """
    start_time = time.time()
    
    try:
        # Determine routing mode
        if mode == "direct":
            # Direct to configured Ollama, ignore AirLLM
            use_airllm = False
            use_proxy = False
            model_to_use = DEFAULT_MODEL  # Use default model
        elif mode == "direct_airllm":
            # Through proxy with AirLLM enabled
            use_airllm = True
            use_proxy = True
            model_to_use = DEFAULT_MODEL  # Use default model
        elif mode == "llm":
            # Use IntelliProxy LLM for routing, no AirLLM
            use_airllm = False
            use_proxy = True
            classification = await router.classify_task(prompt)
            model_to_use = router._select_best_model(classification, prompt)
            if not model_to_use:
                model_to_use = DEFAULT_MODEL
        elif mode == "llm_airllm":
            # IntelliProxy with AirLLM enabled
            use_airllm = True
            use_proxy = True
            classification = await router.classify_task(prompt)
            model_to_use = router._select_best_model(classification, prompt)
            if not model_to_use:
                model_to_use = DEFAULT_MODEL
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        endpoint = get_model_endpoint_test(model_to_use, use_airllm, use_proxy)
        
        ollama_request = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            f"{endpoint}/api/generate",
            json=ollama_request,
            timeout=120
        )
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "mode": mode,
                "test_number": test_number,
                "total_tests": total_tests,
                "model": model_to_use,
                "endpoint": endpoint,
                "duration": duration,
                "tokens": result.get("eval_count", 0),
                "response": result.get("response", "")[:200] + "..." if len(result.get("response", "")) > 200 else result.get("response", ""),
                "success": True
            }
        else:
            return {
                "mode": mode,
                "test_number": test_number,
                "total_tests": total_tests,
                "model": model_to_use,
                "endpoint": endpoint,
                "duration": duration,
                "tokens": 0,
                "response": f"Error: {response.status_code}",
                "success": False,
                "error": response.text[:100]
            }
            
    except Exception as e:
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        return {
            "mode": mode,
            "test_number": test_number,
            "total_tests": total_tests,
            "model": "N/A",
            "endpoint": "N/A",
            "duration": duration,
            "tokens": 0,
            "response": f"Error: {str(e)[:100]}",
            "success": False,
            "error": str(e)[:100]
        }

@api_app.post("/performance-test")
async def run_full_performance_test(request: dict):
    """
    Run performance test across all 4 routing paths or a single mode.
    
    Request body:
    - prompt: The prompt to test with
    - mode: (optional) Specific mode to test: "direct", "direct_airllm", "llm", "llm_airllm"
    
    If mode is specified, only that single test is run.
    """
    prompt = request.get("prompt", "What is a transparent proxy?")
    mode = request.get("mode", None)  # Optional: run single mode
    
    # If a specific mode is requested, run only that one
    if mode:
        modes = [mode]
    else:
        modes = [ "llm", "llm_airllm","direct", "direct_airllm"]
    
    results = []
    
    for i, mode_item in enumerate(modes):
        try:
            result = await run_performance_test(prompt, mode_item, i + 1, len(modes))
            results.append(result)
        except Exception as e:
            results.append({
                "mode": mode_item,
                "test_number": i + 1,
                "total_tests": len(modes),
                "model": "N/A",
                "endpoint": "N/A",
                "duration": 0,
                "tokens": 0,
                "response": f"Error: {str(e)[:100]}",
                "success": False,
                "error": str(e)[:100]
            })
    
    return {
        "prompt": prompt,
        "results": results
    }

# ============================================================================
# OLLAMA-COMPATIBLE ENDPOINTS (Transparent Proxy)
# ============================================================================

INTELLI_PROXY_MODEL = "IntelliProxyLLM"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:7b")

@api_app.get("/api/tags")
async def list_models_ollama():
    """List available models including IntelliProxyLLM with auto-discovered attributes"""
    models = []
    
    # Add real models from Ollama with attributes
    for name, info in router.available_models.items():
        # Get or discover attributes for unknown models
        if name not in MODEL_ATTRIBUTES and ENABLE_AUTO_DISCOVERY:
            try:
                import asyncio
                attrs = asyncio.run(router.discover_model_attributes(name))
                MODEL_ATTRIBUTES[name] = attrs
            except:
                attrs = {"speed": 5, "complexity": 5, "preferred_for": ["general"]}
        
        model_data = {
            "name": name,
            "modified_at": info.get("modified", datetime.now().isoformat()),
            "size": info.get("size", 0)
        }
        
        # Add attributes if available
        if name in MODEL_ATTRIBUTES:
            model_data.update(MODEL_ATTRIBUTES[name])
        
        models.append(model_data)
    
    # Add IntelliProxyLLM for intelligent routing
    models.append({
        "name": INTELLI_PROXY_MODEL,
        "modified_at": datetime.now().isoformat(),
        "size": 0,
        "description": "Intelligent routing - proxy selects best model based on task",
        "speed": 0,
        "complexity": 0,
        "preferred_for": ["auto"]
    })
    
    return {"models": models}

# ============================================================================
# STREAMING ENDPOINTS
# ============================================================================

@api_app.post("/api/generate")
async def generate_with_stream(request: GenerateRequest):
    """
    Ollama-compatible generate endpoint with intelligent routing.
    Supports both streaming and non-streaming requests.
    """
    from fastapi.responses import StreamingResponse
    import time
    
    # Determine which model to use
    model_to_use = request.model
    classification = "general"
    
    if model_to_use == INTELLI_PROXY_MODEL or not model_to_use:
        prompt = request.prompt or ""
        if prompt:
            classification = await router.classify_task(prompt)
            model_to_use = router._select_best_model(classification, prompt)
        else:
            model_to_use = DEFAULT_MODEL
    
    if not model_to_use:
        model_to_use = DEFAULT_MODEL
    
    # Handle streaming
    if request.stream:
        ollama_request = {
            "model": model_to_use,
            "prompt": request.prompt or "",
            "stream": True
        }
        if request.options:
            ollama_request["options"] = request.options
        if request.system:
            ollama_request["system"] = request.system
        if request.images:
            ollama_request["images"] = request.images
        
        def generate_stream():
            model_endpoint = get_model_endpoint(model_to_use)
            try:
                response = requests.post(
                    f"{model_endpoint}/api/generate",
                    json=ollama_request,
                    stream=True,
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            yield f"data: {line.decode('utf-8')}\n\n"
                else:
                    yield f"data: {json.dumps({'error': response.text})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    # Non-streaming request
    start_time = time.time()
    ollama_request = {
        "model": model_to_use,
        "prompt": request.prompt or "",
        "stream": False
    }
    if request.options:
        ollama_request["options"] = request.options
    if request.system:
        ollama_request["system"] = request.system
    if request.images:
        ollama_request["images"] = request.images
    
    try:
        model_endpoint = get_model_endpoint(model_to_use)
        response = requests.post(
            f"{model_endpoint}/api/generate",
            json=ollama_request,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            execution_time = time.time() - start_time
            stats.record_request(model_to_use, classification, execution_time, request.prompt or "")
            
            return {
                "model": model_to_use,
                "response": result.get("response", ""),
                "done": True
            }
        else:
            # Check for memory errors - try fallback
            error_detail = response.text
            memory_keywords = ["memory", "requires more", "not available", "failed to load", "does not have enough"]
            if any(keyword in error_detail.lower() for keyword in memory_keywords):
                fallback = MODEL_FALLBACKS.get(model_to_use)
                if fallback and fallback != model_to_use:
                    print(f"⚠️ Memory error with {model_to_use}, trying fallback: {fallback}")
                    ollama_request["model"] = fallback
                    try:
                        fallback_endpoint = get_model_endpoint(fallback)
                        response = requests.post(
                            f"{fallback_endpoint}/api/generate",
                            json=ollama_request,
                            timeout=REQUEST_TIMEOUT
                        )
                        if response.status_code == 200:
                            result = response.json()
                            execution_time = time.time() - start_time
                            stats.record_request(fallback, classification, execution_time, request.prompt or "")
                            return {
                                "model": fallback,
                                "response": result.get("response", ""),
                                "done": True
                            }
                    except:
                        pass
            raise HTTPException(status_code=response.status_code, detail=error_detail)
            
    except requests.exceptions.Timeout:
        fallback = MODEL_FALLBACKS.get(model_to_use)
        if fallback:
            ollama_request["model"] = fallback
            fallback_endpoint = get_model_endpoint(fallback)
            response = requests.post(
                f"{fallback_endpoint}/api/generate",
                json=ollama_request,
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                result = response.json()
                execution_time = time.time() - start_time
                stats.record_request(fallback, classification, execution_time, request.prompt or "")
                return {
                    "model": fallback,
                    "response": result.get("response", ""),
                    "done": True
                }
        raise HTTPException(status_code=504, detail="Model execution timeout")

@api_app.post("/api/chat")
async def chat_with_stream(request: ChatRequest):
    """
    Ollama-compatible chat endpoint with intelligent routing.
    Supports both streaming and non-streaming requests.
    """
    from fastapi.responses import StreamingResponse
    import time
    
    # Determine which model to use
    model_to_use = request.model
    classification = "general"
    
    if model_to_use == INTELLI_PROXY_MODEL or not model_to_use:
        prompt = " ".join([msg.content for msg in request.messages])
        if prompt:
            classification = await router.classify_task(prompt)
            model_to_use = router._select_best_model(classification, prompt)
        else:
            model_to_use = DEFAULT_MODEL
    
    if not model_to_use:
        model_to_use = DEFAULT_MODEL
    
    # Handle streaming
    if request.stream:
        ollama_request = {
            "model": model_to_use,
            "messages": [msg.dict() for msg in request.messages],
            "stream": True
        }
        if request.options:
            ollama_request["options"] = request.options
        if request.system:
            ollama_request["system"] = request.system
        
        def chat_stream():
            model_endpoint = get_model_endpoint(model_to_use)
            try:
                response = requests.post(
                    f"{model_endpoint}/api/chat",
                    json=ollama_request,
                    stream=True,
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            yield f"data: {line.decode('utf-8')}\n\n"
                else:
                    yield f"data: {json.dumps({'error': response.text})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(chat_stream(), media_type="text/event-stream")
    
    # Non-streaming request
    start_time = time.time()
    ollama_request = {
        "model": model_to_use,
        "messages": [msg.dict() for msg in request.messages],
        "stream": False
    }
    if request.options:
        ollama_request["options"] = request.options
    if request.system:
        ollama_request["system"] = request.system
    
    try:
        model_endpoint = get_model_endpoint(model_to_use)
        response = requests.post(
            f"{model_endpoint}/api/chat",
            json=ollama_request,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            execution_time = time.time() - start_time
            prompt = " ".join([msg.content for msg in request.messages])
            stats.record_request(model_to_use, classification, execution_time, prompt)
            
            return {
                "model": model_to_use,
                "message": result.get("message", {"role": "assistant", "content": ""}),
                "done": True
            }
        else:
            # Check for memory errors - try fallback
            error_detail = response.text
            memory_keywords = ["memory", "requires more", "not available", "failed to load", "does not have enough"]
            if any(keyword in error_detail.lower() for keyword in memory_keywords):
                fallback = MODEL_FALLBACKS.get(model_to_use)
                if fallback and fallback != model_to_use:
                    print(f"⚠️ Memory error with {model_to_use}, trying fallback: {fallback}")
                    ollama_request["model"] = fallback
                    try:
                        fallback_endpoint = get_model_endpoint(fallback)
                        response = requests.post(
                            f"{fallback_endpoint}/api/chat",
                            json=ollama_request,
                            timeout=REQUEST_TIMEOUT
                        )
                        if response.status_code == 200:
                            result = response.json()
                            execution_time = time.time() - start_time
                            prompt = " ".join([msg.content for msg in request.messages])
                            stats.record_request(fallback, classification, execution_time, prompt)
                            return {
                                "model": fallback,
                                "message": result.get("message", {"role": "assistant", "content": ""}),
                                "done": True
                            }
                    except:
                        pass
            raise HTTPException(status_code=response.status_code, detail=error_detail)
            
    except requests.exceptions.Timeout:
        fallback = MODEL_FALLBACKS.get(model_to_use)
        if fallback:
            ollama_request["model"] = fallback
            fallback_endpoint = get_model_endpoint(fallback)
            response = requests.post(
                f"{fallback_endpoint}/api/chat",
                json=ollama_request,
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                result = response.json()
                execution_time = time.time() - start_time
                prompt = " ".join([msg.content for msg in request.messages])
                stats.record_request(fallback, classification, execution_time, prompt)
                return {
                    "model": fallback,
                    "message": result.get("message", {"role": "assistant", "content": ""}),
                    "done": True
                }
        raise HTTPException(status_code=504, detail="Model execution timeout")

# ============================================================================
# ADDITIONAL OLLAMA COMPATIBLE ENDPOINTS (Transparent Forwarding)
# ============================================================================

@api_app.post("/api/pull")
async def pull_model(request: dict):
    """Forward pull model request to Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_TARGET['base_url']}/api/pull",
            json=request,
            timeout=REQUEST_TIMEOUT
        )
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.delete("/api/delete")
async def delete_model(request: dict):
    """Forward delete model request to Ollama"""
    try:
        response = requests.delete(
            f"{OLLAMA_TARGET['base_url']}/api/delete",
            json=request,
            timeout=REQUEST_TIMEOUT
        )
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.post("/api/embeddings")
async def embeddings(request: dict):
    """Forward embeddings request to Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_TARGET['base_url']}/api/embeddings",
            json=request,
            timeout=REQUEST_TIMEOUT
        )
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        "ollama_url": OLLAMA_TARGET["base_url"]
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
