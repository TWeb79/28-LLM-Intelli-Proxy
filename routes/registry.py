"""
Registry refresh endpoints for LLM IntelliProxy.

Provides endpoints for refreshing model registries from different providers.
"""
import asyncio
import httpx
from typing import Dict, Any, List
from fastapi import HTTPException
from services import registry as model_registry
from providers.nvidia_provider import NvidiaProvider
from providers.ollama_provider import OllamaProvider


async def refresh_models() -> Dict[str, Any]:
    """Refresh models from all configured providers and return detailed model lists."""
    results = {
        "ollama": {"status": "skipped", "models": 0, "model_list": []},
        "nvidia": {"status": "skipped", "models": 0, "model_list": []}
    }

    all_models = []

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
            "models": len(ollama_models),
            "model_list": ollama_models
        }
        all_models.extend([{**m, "provider": "ollama"} for m in ollama_models])
    except Exception as e:
        results["ollama"] = {
            "status": "error",
            "error": str(e),
            "models": 0,
            "model_list": []
        }

    # Refresh NVIDIA models
    try:
        nvidia_provider = NvidiaProvider()
        if nvidia_provider.api_key: # Only if API key is configured
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
                "models": len(nvidia_models),
                "model_list": nvidia_models
            }
            all_models.extend([{**m, "provider": "nvidia"} for m in nvidia_models])
        else:
            results["nvidia"] = {
                "status": "skipped",
                "reason": "API key not configured",
                "models": 0,
                "model_list": []
            }
    except Exception as e:
        results["nvidia"] = {
            "status": "error",
            "error": str(e),
            "models": 0,
            "model_list": []
        }

    return {
        "status": "completed",
        "providers": results,
        "total_models": len(all_models),
        "all_models": all_models
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

    # Validate API key format
    if not api_key.startswith("nvapi-") and len(api_key) < 20:
        raise HTTPException(
            status_code=400,
            detail="Invalid API key format. NVIDIA API keys should start with 'nvapi-'"
        )

    # Test the configuration with detailed error handling
    try:
        provider = NvidiaProvider(api_key=api_key)

        # Test health check first
        is_healthy = await provider.health_check()
        if not is_healthy:
            # Try to get more specific error
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    test_resp = await client.get(
                        "https://integrate.api.nvidia.com/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"}
                    )
                if test_resp.status_code == 401:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid API key. Please check your NVIDIA API key."
                    )
                elif test_resp.status_code == 403:
                    raise HTTPException(
                        status_code=400,
                        detail="API key not authorized for NVIDIA NIM service."
                    )
                elif test_resp.status_code >= 500:
                    raise HTTPException(
                        status_code=400,
                        detail="NVIDIA service temporarily unavailable. Please try again later."
                    )
            except httpx.ConnectError:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot connect to NVIDIA services. Check your internet connection."
                )
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=400,
                    detail="Connection to NVIDIA services timed out. Please try again."
                )

        # Get available models
        models = await provider.list_models()

        # Store configuration persistently
        import os
        os.environ["NVIDIA_API_KEY"] = api_key

        # Save to config file for persistence
        try:
            import json
            config_path = "/data/nvidia_config.json"
            with open(config_path, "w") as f:
                json.dump({"api_key": api_key}, f)
        except Exception:
            pass # Don't fail if we can't write to file

        return {
            "status": "ok",
            "message": f"NVIDIA configuration saved successfully",
            "models_available": len(models),
            "models": [m["id"] for m in models[:5]] # Return first 5 model names
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = str(e)
        if "SSL" in error_msg or "certificate" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="SSL certificate error. This might be a temporary network issue."
            )
        elif "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail="Connection timeout. Please check your internet connection."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration failed: {error_msg}"
            )