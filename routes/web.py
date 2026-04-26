"""
Web dashboard routes for LLM IntelliProxy.

This module defines endpoints for the web dashboard and static file serving.
"""
import os
from typing import Dict

from fastapi.responses import FileResponse, JSONResponse

from services.router import get_http_client

# Global state (set during app startup)
_router = None
_ollama_target: Dict[str, str] = {"base_url": "http://localhost:11434"}


def set_router(router):
    global _router
    _router = router


def set_ollama_target(target: Dict[str, str]):
    global _ollama_target
    _ollama_target = target


def initialize(router, ollama_target: Dict[str, str]) -> None:
    """Initialize module-level globals from main app."""
    global _router, _ollama_target
    _router = router
    _ollama_target = ollama_target


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

async def web_stats():
    """Usage statistics for dashboard."""
    if not _router:
        return {"total_requests": 0, "models": {}, "model_avg_times": {}, "categories": {}}
    return _router.stats.to_dict()


async def web_health():
    """Health check for dashboard."""
    client = await get_http_client()
    ollama_ok = False
    try:
        resp = await client.get(f"{_ollama_target.get('base_url', 'http://localhost:11434')}/api/tags", timeout=5)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False
    return {
        "status": "healthy" if ollama_ok else "degraded",
        "models": len(_router.available_models) if _router else 0,
        "cache_hit_rate": _router.classification_cache.stats()["hit_rate"] if _router else "0.0%"
    }


async def get_fallbacks_config():
    """Get fallback configuration."""
    from services.fallbacks import get_fallbacks
    return {
        "timeout": int(os.getenv("FALLBACK_TIMEOUT", "30")),
        "fallbacks": get_fallbacks()
    }


async def dashboard():
    """Serve dashboard HTML."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "Dashboard not found"}, status_code=404)
