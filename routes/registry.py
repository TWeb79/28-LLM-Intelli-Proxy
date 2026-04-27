"""
Registry refresh endpoints for LLM IntelliProxy.

Provides endpoints for refreshing model registries from different providers.
"""
import asyncio
from typing import Dict, Any, List
from fastapi import HTTPException
from services import registry as model_registry
from providers.nvidia_provider import NvidiaProvider
from providers.ollama_provider import OllamaProvider


async def refresh_models() -> Dict[str, Any]:
    """Refresh models from all configured providers."""
    results = {
        "ollama": {"status": "skipped", "models": 0},
        "nvidia": {"status": "skipped", "models": 0}
    }
    
    # Refresh Ollama models
    try:
        ollama_provider = OllamaProvider()
        ollama_models = await ollama_provider.list_models()
        
        for model in ollama_models:
            model_registry.upsert_model(
                provider="ollama",
                model_id=model["id"],
                source_url=model["source_url"],
                category=model.get("category"),
                description=model.get("description"),
                context_window=model.get("context_window"),
                enabled=True
            )
        
        results["ollama"] = {
            "status": "success",
            "models": len(ollama_models)
        }
    except Exception as e:
        results["ollama"] = {
            "status": "error",
            "error": str(e),
            "models": 0
        }
    
    # Refresh NVIDIA models
    try:
        nvidia_provider = NvidiaProvider()
        if nvidia_provider.api_key:  # Only if API key is configured
            nvidia_models = await nvidia_provider.list_models()
            
            for model in nvidia_models:
                model_registry.upsert_model(
                    provider="nvidia",
                    model_id=model["id"],
                    source_url=model["source_url"],
                    category=model.get("category"),
                    description=model.get("description"),
                    context_window=model.get("context_window"),
                    enabled=True
                )
            
            results["nvidia"] = {
                "status": "success",
                "models": len(nvidia_models)
            }
        else:
            results["nvidia"] = {
                "status": "skipped",
                "reason": "API key not configured",
                "models": 0
            }
    except Exception as e:
        results["nvidia"] = {
            "status": "error",
            "error": str(e),
            "models": 0
        }
    
    return {
        "status": "completed",
        "providers": results,
        "total_models": sum(r["models"] for r in results.values())
    }


async def get_nvidia_status() -> Dict[str, Any]:
    """Get NVIDIA provider status."""
    try:
        provider = NvidiaProvider()
        is_healthy = await provider.health_check()
        
        return {
            "status": "running" if is_healthy else "unreachable",
            "configured": bool(provider.api_key),
            "models": len(await provider.list_models()) if is_healthy else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "configured": False,
            "models": 0
        }


async def set_nvidia_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set NVIDIA configuration."""
    api_key = config.get("api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    
    # Test the configuration
    try:
        provider = NvidiaProvider(api_key=api_key)
        is_healthy = await provider.health_check()
        
        if not is_healthy:
            raise HTTPException(status_code=400, detail="Invalid API key or NVIDIA service unreachable")
        
        # Store configuration (in production, this would be saved to config)
        import os
        os.environ["NVIDIA_API_KEY"] = api_key
        
        return {
            "status": "ok",
            "message": "NVIDIA configuration saved",
            "models_available": len(await provider.list_models())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))