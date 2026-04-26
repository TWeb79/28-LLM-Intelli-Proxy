"""
Intelligent Router - Fast model selection and execution.

This module contains the core routing logic for selecting the best
LLM model for a given prompt and executing the request.
"""
import os
import time
import asyncio
import logging
import httpx
from typing import Optional, Dict, List, Any
from datetime import datetime

from fastapi import HTTPException

from providers.ollama_provider import OllamaProvider
from services import registry as model_registry
from services.caches import ClassificationCache, ModelScoreCache, Statistics
from services.model_metadata import MODEL_ATTRIBUTES
from services.decision_engine import DecisionEngine
from services.fallbacks import get_fallback_for_model


# Global async HTTP client (initialized on first use)
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client():
    """Get or create global async HTTP client with connection pooling.

    Returns:
        httpx.AsyncClient instance
    """
    global _http_client
    if _http_client is None:
        import httpx
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            timeout=httpx.Timeout(int(os.getenv('REQUEST_TIMEOUT', '120'))),
            verify=False,
            http2=False
        )
    return _http_client


class IntelligentRouter:
    """Intelligent model router with caching and optimized selection."""

    def __init__(
        self,
        ollama_base_url: str,
        decision_engine: DecisionEngine,
        classifier_model: Optional[str] = None
    ):
        """Initialize the intelligent router.

        Args:
            ollama_base_url: Base URL for Ollama API
            decision_engine: Configured DecisionEngine instance
            classifier_model: Optional classifier model name (for legacy compatibility)
        """
        self.ollama_base_url = ollama_base_url
        self.decision_engine = decision_engine
        self.classifier_model = classifier_model
        self.available_models: Dict[str, Dict[str, Any]] = {}
        self.model_categories: Dict[str, List[str]] = {}

        # Initialize caches
        self.classification_cache = ClassificationCache(max_size=2000)
        self.model_score_cache = ModelScoreCache()
        self.stats = Statistics()

    async def discover_models(self) -> Dict[str, Dict[str, Any]]:
        """Discover available models from Ollama.

        Returns:
            Dictionary mapping model names to metadata
        """
        try:
            client = await get_http_client()
            response = await client.get(f"{self.ollama_base_url}/api/tags", timeout=30)

            if response.status_code == 200:
                models = response.json().get("models", [])
                self.available_models.clear()

                for model in models:
                    name = model["name"]
                    self.available_models[name] = {
                        "name": name,
                        "size": model.get("size", 0),
                        "modified": model.get("modified_at", ""),
                    }

                # Pre-compute model scores for fast selection
                self.model_score_cache.compute_scores(MODEL_ATTRIBUTES)

                self._categorize_models()
                logging.info(f"Discovered {len(self.available_models)} models from Ollama")
                return self.available_models
            else:
                raise Exception(f"Ollama returned HTTP {response.status_code}")

        except Exception as e:
            logging.warning(f"Model discovery failed: {e}")
            return {}

    def _categorize_models(self) -> None:
        """Categorize discovered models by capability."""
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
        """Classify the task type from the prompt.

        Checks cache first, then uses simple heuristics,
        and optionally enhances with LLM-based classification.

        Args:
            prompt: User prompt

        Returns:
            Classification string (e.g., 'code', 'vision', 'reasoning', 'general')
        """
        # Check cache first
        cached = self.classification_cache.get(prompt)
        if cached:
            return cached

        # Quick heuristic classification
        prompt_lower = prompt.lower()

        if any(w in prompt_lower for w in ["image", "picture", "photo", "describe", "visual"]):
            classification = "vision"
        elif any(w in prompt_lower for w in ["code", "debug", "function", "program"]):
            classification = "code"
        elif any(w in prompt_lower for w in ["prove", "analyze", "theorem", "step by step"]):
            classification = "reasoning"
        else:
            classification = "general"

        # Cache the result
        self.classification_cache.put(prompt, classification)
        return classification

    def _select_best_model(self, category: str, prompt_complexity: int) -> Optional[str]:
        """Select the best model for a task using pre-computed scores.

        Args:
            category: Task category
            prompt_complexity: Complexity score (1-10)

        Returns:
            Model name or None if no suitable model found
        """
        category_models = self.model_categories.get(category, [])
        if not category_models:
            category_models = self.model_categories.get("general", [])
        if not category_models:
            return None

        best_model = None
        best_score = -999

        for model in category_models:
            attrs = MODEL_ATTRIBUTES.get(model, {"speed": 5, "complexity": 5})
            complexity_match = 1 if attrs["complexity"] >= prompt_complexity else 0.5
            speed_factor = attrs["speed"] / 10.0

            if prompt_complexity <= 3:
                score = speed_factor * 2 + complexity_match
            elif prompt_complexity <= 6:
                score = speed_factor + complexity_match * 2
            else:
                score = complexity_match * 3 - (10 - speed_factor) * 0.5

            if score > best_score:
                best_score = score
                best_model = model

        return best_model

    def _analyze_prompt_complexity(self, prompt: str) -> int:
        """Analyze prompt complexity on a scale of 1-10.

        Args:
            prompt: User prompt

        Returns:
            Complexity score (1-10)
        """
        complexity = 3  # Default
        prompt_lower = prompt.lower()

        if any(w in prompt_lower for w in ["analyze", "debug", "optimize", "architecture"]):
            complexity += 3
        elif any(w in prompt_lower for w in ["explain", "describe", "compare"]):
            complexity += 1

        word_count = len(prompt.split())
        complexity += min(word_count // 300, 3)

        return min(max(complexity, 1), 10)

    async def route_and_execute(
        self,
        prompt: str,
        stream: bool = False,
        requested_model: Optional[str] = None,
        override_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Route and execute a prompt.

        Decision flow:
        - If override_model provided, use it directly
        - Else if requested_model provided (and not 'IntelliProxyLLM'), use it
        - Else consult DecisionEngine to select from registry
        - Fall back to configured fallback model

        Args:
            prompt: User prompt
            stream: Whether to stream response (not yet supported)
            requested_model: Explicitly requested model
            override_model: Override from X-LLMProxy-Model header

        Returns:
            Execution result dict with response, model used, timing, etc.
        """
        start_time = time.time()

        if override_model:
            selected_model = override_model
            classification = await self.classify_task(prompt)
        elif requested_model and requested_model != "IntelliProxyLLM":
            selected_model = requested_model
            classification = await self.classify_task(prompt)
        else:
            # Use decision engine
            decision_info = await self.decision_engine.select_model(prompt)
            selected_model = decision_info.get("selected_model")
            reason = decision_info.get("reason")
            latency_ms = decision_info.get("latency_ms", 0)
            provider = decision_info.get("provider")

            # Validate selected_model against registry; use fallback if invalid
            registry_models = {m["id"]: m for m in model_registry.list_models()}
            if not selected_model or selected_model not in registry_models:
                fallback = os.getenv("FALLBACK_MODEL", "qwen2.5:8b")
                selected_model = fallback
                try:
                    self.decision_engine.persist_decision(
                        prompt, selected_model or "", provider,
                        reason or "fallback used", latency_ms
                    )
                except Exception:
                    pass

            classification = await self.classify_task(prompt)

        if not selected_model:
            raise HTTPException(status_code=503, detail="No models available")

        # Execute the request
        result = await self._execute_direct(selected_model, prompt, stream, start_time, classification)

        # Persist decision asynchronously (best-effort)
        try:
            self.decision_engine.persist_decision(
                prompt, selected_model, None, "",
                int((time.time() - start_time) * 1000)
            )
        except Exception:
            pass

        return result

    async def _execute_direct(
        self,
        model: str,
        prompt: str,
        stream: bool,
        start_time: float,
        classification: str
    ) -> Dict[str, Any]:
        """Execute request directly via Ollama provider.

        Args:
            model: Model name
            prompt: User prompt
            stream: Streaming flag (ignored, always False for now)
            start_time: Request start timestamp
            classification: Task classification

        Returns:
            Result dict with response, model_used, execution_time, etc.

        Raises:
            HTTPException: If all model attempts fail
        """
        from fastapi import HTTPException

        # Get fallback model from fallback store
        fallback_model = get_fallback_for_model(model)
        models_to_try = [model]
        if fallback_model and fallback_model != model:
            models_to_try.append(fallback_model)

        # Add category-based fallbacks
        category_fallbacks = self.model_categories.get(classification, [])
        for m in category_fallbacks:
            if m != model and m not in models_to_try:
                models_to_try.append(m)

        last_error = None
        for model_to_try in models_to_try:
            try:
                client = await get_http_client()
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={"model": model_to_try, "prompt": prompt, "stream": False},
                    timeout=int(os.getenv('REQUEST_TIMEOUT', '120'))
                )
                if response.status_code == 200:
                    result_text = response.json()["response"]
                    execution_time = time.time() - start_time
                    self.stats.record_request(model_to_try, classification, execution_time)
                    return {
                        "result": result_text,
                        "model_used": model_to_try,
                        "task_classification": classification,
                        "execution_time": round(execution_time, 2),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    last_error = f"Ollama HTTP {response.status_code}"
                    continue
            except Exception as e:
                last_error = str(e)
                continue

        logging.error(f"All model attempts failed: {last_error}")
        raise HTTPException(status_code=500, detail=f"All models failed: {last_error}")

    async def warm_up_models(self) -> None:
        """Pre-load common models into memory for faster first request."""
        warm_models = [
            m for m in self.available_models.keys()
            if m in MODEL_ATTRIBUTES
        ][:3]

        client = await get_http_client()
        for model in warm_models:
            try:
                await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={"model": model, "prompt": "warmup", "stream": False},
                    timeout=30
                )
                logging.info(f"Warmed up model: {model}")
            except Exception:
                logging.debug(f"Warmup failed for model: {model}", exc_info=True)
