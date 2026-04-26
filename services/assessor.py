"""
AI Model Assessor

When a new model appears in the registry (assessed==0), this module can be used
to generate a category and short description using the configured decision model
or the fallback model.

This assessor runs asynchronously and persists results to the registry.
"""
import os
import json
import time
from typing import Optional
from services import registry as model_registry
from services.decision_engine import DecisionEngine


class Assessor:
    def __init__(self, decision_model: Optional[str] = None):
        self.engine = DecisionEngine(decision_model)

    async def assess_model(self, provider: str, model_id: str) -> None:
        """Run assessment prompt for a single model and persist category+description."""
        prompt = (
            f"You are a model catalog assistant. Given the model name below, infer its most likely\n"
            f"primary capability category and write a 1-2 sentence description of its key strengths.\n\n"
            f"Model name: {model_id}\n"
            f"Provider: {provider}\n\n"
            f"Return ONLY valid JSON:\n{{\n  \"category\": \"<one of: reasoning|coding|chat|image|vision|math|tool-use|embedding>\",\n  \"description\": \"<1-2 sentences>\"\n}}\n"
        )

        # Use decision engine to call underlying LLM
        result = await self.engine.select_model(prompt)

        # select_model returns structure for routing; but here we assume engine will return a response
        # We try to parse any text in reason as JSON
        category = None
        description = None
        try:
            # Sometimes select_model returns selected_model as None; the raw LLM output parsing is in engine
            # We'll call the engine's provider directly for assessor use (reuse engine.default_provider)
            provider_adapter = self.engine.default_provider
            resp = await provider_adapter.forward_request(self.engine.model_id or model_id, {"prompt": prompt, "stream": False})
            text = None
            if isinstance(resp, dict):
                text = resp.get("response") or resp.get("text") or json.dumps(resp)
            else:
                text = str(resp)
            parsed = json.loads(text or "{}")
            category = parsed.get("category")
            description = parsed.get("description")
        except Exception:
            # best-effort fallback: set generic category/description
            category = "general"
            description = f"Auto-assessed placeholder for {model_id} (provider: {provider})"

        # Persist to registry
        try:
            if category or description:
                model_registry.mark_assessed(provider, model_id, category or "general", description or "")
        except Exception:
            pass
