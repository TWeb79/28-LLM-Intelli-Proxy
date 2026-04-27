"""
Updated API routes for LLM IntelliProxy with NVIDIA integration.

This file contains the updated API endpoints including NVIDIA support and registry refresh functionality.
"""
import os
import io
import json
import time
import csv
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import Request, Response, HTTPException
from fastapi.responses import StreamingResponse

from services import registry as model_registry
from services import router_service
from services.scheduler import get_scheduler
from services.router import IntelligentRouter, get_http_client
from services.decision_engine import DecisionEngine
from services.config_loader import load_config
from providers.ollama_provider import OllamaProvider

# ============================================================================
# GLOBAL STATE (injected at startup)
# ============================================================================

_router: Optional[IntelligentRouter] = None
_config: Optional[Dict[str, Any]] = None
_classifier_model: Optional[str] = None
_ollama_target: Dict[str, str] = {"base_url": "http://localhost:11434"}
_decision_engine: Optional[DecisionEngine] = None
_ollama_provider: Optional[OllamaProvider] = None
_proxy_port: int = 8130
_db_path: str = "/data/llmproxy.db"

# Core dependencies (also injected)
_ollama_target: Dict[str, str] = {"base_url": "http://localhost:11434"}
_decision_engine: Optional[DecisionEngine] = None
_ollama_provider: Optional[OllamaProvider] = None
_proxy_port: int = 8130
_db_path: str = "/data/llmproxy.db"


# ============================================================================
# DEPENDENCY INJECTION SETTERS
# ============================================================================

def set_router(router: IntelligentRouter) -> None:
    """Set the global router instance."""
    global _router
    _router = router


def set_config(config: Dict[str, Any]) -> None:
    """Set global configuration."""
    global _config
    _config = config


def set_classifier_model(model: str) -> None:
    """Set classifier model name."""
    global _classifier_model
    _classifier_model = model


def set_ollama_target(target: Dict[str, str]) -> None:
    """Set Ollama target configuration."""
    global _ollama_target
    _ollama_target = target


def set_decision_engine(engine: DecisionEngine) -> None:
    """Set the decision engine."""
    global _decision_engine
    _decision_engine = engine


def set_ollama_provider(provider: OllamaProvider) -> None:
    """Set the Ollama provider."""
    global _ollama_provider
    _ollama_provider = provider


def set_proxy_port(port: int) -> None:
    """Set the proxy port."""
    global _proxy_port
    _proxy_port = port


def set_db_path(path: str) -> None:
    """Set the database path."""
    global _db_path
    _db_path = path


# ============================================================================
# REGISTRY ENDPOINTS
# ============================================================================

async def refresh_registry() -> Dict[str, Any]:
    """Refresh models from all configured providers."""
    from routes.registry import refresh_models
    return await refresh_models()


async def get_nvidia_status() -> Dict[str, Any]:
    """Get NVIDIA provider status."""
    from routes.registry import get_nvidia_status
    return await get_nvidia_status()


async def set_nvidia_config(request: dict) -> Dict[str, Any]:
    """Set NVIDIA configuration."""
    from routes.registry import set_nvidia_config
    return await set_nvidia_config(request)


# ============================================================================
# CORE ENDPOINTS
# ============================================================================

async def list_models():
    models = dict(_router.available_models) if _router else {}
    decision_model = _config.get("decision", {}).get("model") if _config else None
    if decision_model and decision_model in models:
        models.pop(decision_model, None)
    return {
        "total": len(models),
        "models": models,
        "categories": _router.model_categories if _router else {}
    }


async def v1_models():
    registry_models = model_registry.list_models()
    data = [{
        "id": "intelliproxy-auto",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "intelliproxy",
        "description": "Automatically routes your request to the most suitable model based on task analysis."
    }]
    for m in registry_models:
        if not m.get("enabled", True):
            continue
        data.append({
            "id": m.get("id"),
            "object": "model",
            "created": int(time.mktime(datetime.fromisoformat(m["last_seen"]).timetuple())) if m.get("last_seen") else int(time.time()),
            "owned_by": m.get("provider"),
            "description": m.get("description") or ""
        })
    return {"object": "list", "data": data}


async def process_task(request):
    return await router_service.route_and_execute(request.prompt, request.stream)


async def generate(request):
    model = request.model or "intelliproxy-auto"
    prompt = request.prompt or ""
    registry = {m['id']: m for m in model_registry.list_models()}

    async def forward_to_provider(provider_name: str, model_id: str, payload: dict, endpoint: str = "/api/generate"):
        if provider_name == 'ollama':
            prov = _ollama_provider or OllamaProvider(name="ollama", base_url=_ollama_target.get("base_url"))
            return await prov.forward_request(model_id, payload, stream=request.stream, endpoint=endpoint)
        raise HTTPException(status_code=502, detail=f"Provider adapter for '{provider_name}' not implemented")

    if model == 'intelliproxy-auto':
        decision = await _decision_engine.select_model(prompt) if _decision_engine else {"selected_model": None}
        selected = decision.get('selected_model')
        provider_name = decision.get('provider') or 'ollama'
        reason = decision.get('reason')
        latency_ms = decision.get('latency_ms', 0)

        if not selected:
            selected = _config.get('proxy', {}).get('fallback_model') if _config else "qwen2.5:8b"

        resp = await forward_to_provider(provider_name, selected, {"prompt": prompt}, endpoint="/api/generate")

        try:
            if _decision_engine:
                _decision_engine.persist_decision(prompt, selected, provider_name, reason or '', latency_ms, routing_mode='auto')
        except Exception:
            pass

        headers = {"X-IntelliProxy-Model": selected, "X-IntelliProxy-Provider": provider_name}
        return Response(content=json.dumps(resp), media_type="application/json", headers=headers)

    if model in registry:
        entry = registry[model]
        if not entry.get('enabled', True):
            raise HTTPException(status_code=404, detail="Model not enabled")

        provider_name = entry.get('provider')
        resp = await forward_to_provider(provider_name, model, {"prompt": prompt}, endpoint="/api/generate")

        try:
            if _decision_engine:
                _decision_engine.persist_decision(prompt, model, provider_name, 'passthrough', 0, routing_mode='passthrough')
        except Exception:
            pass

        return resp

    raise HTTPException(status_code=404, detail="Model not found in IntelliProxy registry")


async def chat(request):
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    model = request.model or "intelliproxy-auto"
    registry = {m['id']: m for m in model_registry.list_models()}

    async def forward_to_provider(provider_name: str, model_id: str, payload: dict, endpoint: str = "/api/chat"):
        if provider_name == 'ollama':
            prov = _ollama_provider or OllamaProvider(name="ollama", base_url=_ollama_target.get("base_url"))
            return await prov.forward_request(model_id, payload, stream=request.stream, endpoint=endpoint)
        raise HTTPException(status_code=502, detail=f"Provider adapter for '{provider_name}' not implemented")

    if model == 'intelliproxy-auto':
        decision = await _decision_engine.select_model(prompt) if _decision_engine else {"selected_model": None}
        selected = decision.get('selected_model')
        provider_name = decision.get('provider') or 'ollama'
        reason = decision.get('reason')
        latency_ms = decision.get('latency_ms', 0)

        if not selected:
            selected = _config.get('proxy', {}).get('fallback_model') if _config else "qwen2.5:8b"

        resp = await forward_to_provider(provider_name, selected, {"messages": [m.dict() for m in request.messages]}, endpoint="/api/chat")

        try:
            if _decision_engine:
                _decision_engine.persist_decision(prompt, selected, provider_name, reason or '', latency_ms, routing_mode='auto')
        except Exception:
            pass

        headers = {"X-IntelliProxy-Model": selected, "X-IntelliProxy-Provider": provider_name}
        return Response(content=json.dumps(resp), media_type="application/json", headers=headers)

    if model in registry:
        entry = registry[model]
        if not entry.get('enabled', True):
            raise HTTPException(status_code=404, detail="Model not enabled")

        provider_name = entry.get('provider')
        resp = await forward_to_provider(provider_name, model, {"messages": [m.dict() for m in request.messages]}, endpoint="/api/chat")

        try:
            if _decision_engine:
                _decision_engine.persist_decision(prompt, model, provider_name, 'passthrough', 0, routing_mode='passthrough')
        except Exception:
            pass

        return resp

    raise HTTPException(status_code=404, detail="Model not found in IntelliProxy registry")


async def get_stats():
    stats_dict = {"total_requests": 0, "models": {}, "model_avg_times": {}, "categories": {}}
    cache_stats = {"hits": 0, "misses": 0, "hit_rate": "0.0%"}
    if _router:
        stats_dict = _router.stats.to_dict()
        cache_stats = _router.classification_cache.stats()
    return {
        "requests": stats_dict,
        "cache": cache_stats,
        "models": {
            model: {
                "count": count["count"],
                "total_time": count["total_time"],
                "avg_time": round(count["total_time"] / count["count"], 2) if count["count"] > 0 else 0
            }
            for model, count in _router.stats.models.items()
        } if _router else {}
    }


async def performance_test(request):
    results = []
    available_models = list(_router.available_models.keys()) if _router else []
    if not available_models:
        return {"error": "No models available", "results": []}

    if request.mode in ["direct", "all"]:
        start = time.time()
        try:
            client = await get_http_client()
            test_model = available_models[0]
            response = await client.post(
                f"{_ollama_target['base_url']}/api/generate",
                json={"model": test_model, "prompt": request.prompt, "stream": False},
                timeout=120
            )
            duration = time.time() - start
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "mode": "direct", "label": "Ollama Direct", "model": test_model,
                    "duration": round(duration, 2), "tokens": data.get("eval_count", 0),
                    "response": data.get("response", "")[:100], "success": True
                })
            else:
                results.append({"mode": "direct", "label": "Ollama Direct", "model": test_model,
                                "duration": round(duration, 2), "tokens": 0,
                                "response": f"Error: {response.status_code}", "success": False})
        except Exception as e:
            results.append({"mode": "direct", "label": "Ollama Direct",
                            "model": available_models[0], "duration": 0, "tokens": 0,
                            "response": str(e), "success": False})

    if request.mode in ["intelliproxy", "all"]:
        start = time.time()
        try:
            result = await router_service.route_and_execute(request.prompt, False)
            duration = time.time() - start
            results.append({
                "mode": "intelliproxy", "label": "IntelliProxy",
                "model": result.get("model_used", "unknown"),
                "duration": round(duration, 2), "tokens": 0,
                "response": result.get("result", "")[:100], "success": True
            })
        except Exception as e:
            results.append({
                "mode": "intelliproxy", "label": "IntelliProxy",
                "model": "error", "duration": round(time.time() - start, 2), "tokens": 0,
                "response": f"{type(e).__name__}: {str(e)}", "success": False
            })

    return {"results": results}


async def health_check():
    client = await get_http_client()
    ollama_ok = False
    ollama_count = 0
    try:
        response = await client.get(f"{_ollama_target['base_url']}/api/tags", timeout=5)
        ollama_ok = response.status_code == 200
        ollama_count = len(response.json().get("models", [])) if ollama_ok else 0
    except Exception:
        ollama_ok = False
        ollama_count = 0

    overall = "healthy" if ollama_ok else "degraded"

    return {
        "overall_status": overall,
        "proxy": {"status": "running", "port": _proxy_port},
        "ollama": {"status": "running" if ollama_ok else "unreachable", "models": ollama_count, "url": _ollama_target['base_url']},
        "nvidia": await get_nvidia_status(),
        "performance": {
            "classification_cache_hit_rate": _router.classification_cache.stats()["hit_rate"] if _router else "0.0%",
            "total_requests": _router.stats.total_requests if _router else 0
        }
    }


async def classify_only(prompt: str):
    if not _router:
        raise HTTPException(status_code=503, detail="Router not initialized")
    classification = await _router.classify_task(prompt)
    complexity = _router._analyze_prompt_complexity(prompt)
    model = _router._select_best_model(classification, complexity)
    return {
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "classification": classification,
        "complexity": complexity,
        "recommended_model": model,
        "models_in_category": _router.model_categories.get(classification, [])
    }


# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================

async def get_config():
    return {
        "ollama": _ollama_target,
        "nvidia": {
            "configured": bool(os.getenv("NVIDIA_API_KEY", "")),
            "api_key_set": bool(os.getenv("NVIDIA_API_KEY", ""))
        },
        "router": {
            "mode": "IntelliRouter",
            "classifier_model": _classifier_model,
            "timeout": int(os.getenv('REQUEST_TIMEOUT', '120'))
        }
    }


async def get_fallbacks():
    from services.fallbacks import get_fallbacks as get_fb
    return {
        "timeout": int(os.getenv('FALLBACK_TIMEOUT', '30')),
        "fallbacks": get_fb()
    }


async def set_ollama_target(request: dict):
    global _ollama_target
    if 'host' in request:
        _ollama_target['host'] = request['host']
    if 'port' in request:
        _ollama_target['port'] = request['port']
    _ollama_target['base_url'] = f"http://{_ollama_target['host']}:{_ollama_target['port']}"
    return {"status": "ok", "base_url": _ollama_target['base_url']}


async def set_fallbacks(request: dict):
    from services.fallbacks import set_fallbacks as set_fb, set_timeout as set_tmo
    fallbacks = request.get('fallbacks', {})
    if not isinstance(fallbacks, dict):
        fallbacks = {}
    if any(isinstance(v, list) for v in (fallbacks.values() if fallbacks else [])):
        fallbacks = {}
    set_fb(fallbacks)
    if 'timeout' in request:
        set_tmo(request['timeout'])
    return {"status": "ok", "fallbacks": get_fallbacks(), "timeout": int(os.getenv('FALLBACK_TIMEOUT', '30'))}


async def get_requests():
    return {"recent": [], "total": _router.stats.total_requests if _router else 0}


async def clear_requests():
    return {"status": "ok"}


# ============================================================================
# INITIALIZATION (called by ollama_router at startup)
# ============================================================================

def initialize(
    router: IntelligentRouter,
    config: Dict[str, Any],
    classifier_model: str,
    ollama_target: Dict[str, str],
    decision_engine: DecisionEngine,
    ollama_provider: OllamaProvider,
    proxy_port: int,
    db_path: str
) -> None:
    """Initialize module-level globals from the main application."""
    global _router, _config, _classifier_model, _ollama_target
    global _decision_engine, _ollama_provider, _proxy_port, _db_path

    _router = router
    _config = config
    _classifier_model = classifier_model
    _ollama_target = ollama_target
    _decision_engine = decision_engine
    _ollama_provider = ollama_provider
    _proxy_port = proxy_port
    _db_path = db_path