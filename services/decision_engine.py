"""
Decision engine: uses a configured LLM to choose the best model for a prompt.

This module is intentionally conservative: if no decision model is configured
it returns None so the caller may fallback to heuristics.
"""
import os
import json
import time
import yaml
from typing import Optional, Dict, Any
from providers.ollama_provider import OllamaProvider
from services import registry as model_registry


class DecisionEngine:
    def __init__(self, decision_model: Optional[str] = None):
        # model id like 'IntelliProxyLLM' or 'nvidia/llama-3...'
        self.model_id = decision_model or os.getenv("DECISION_MODEL")
        # default provider adapter to use if decision model provider cannot be resolved
        self.default_provider = OllamaProvider()

    async def select_model(self, user_prompt: str) -> Dict[str, Any]:
        """Return a dict with keys: selected_model, reason, provider, latency_ms, token_count

        If no decision model configured, returns selected_model=None and reason set.
        """
        if not self.model_id:
            return {"selected_model": None, "reason": "no decision model configured", "provider": None, "latency_ms": 0, "token_count": 0}

        # Build dynamic system prompt from registry
        models = model_registry.list_models()
        models_yaml = yaml.safe_dump(models, sort_keys=False)

        system_prompt = (
            "You are an intelligent LLM routing agent. Based on the user's prompt, select the\n"
            "single most appropriate model from the registry below.\n\n"
            "Return ONLY valid JSON: {\"selected_model\": \"<model_id>\", \"reason\": \"<one sentence>\"}\n\n"
            "Available models:\n"
            f"{models_yaml}\n"
            "Selection criteria:\n"
            "- coding / debugging → prefer category: coding\n"
            "- multi-step logic, math, planning → prefer category: reasoning\n"
            "- creative writing, general chat → prefer category: chat\n"
            "- image understanding or generation → prefer category: image or vision\n"
            "- prefer smaller/faster models for simple tasks\n"
            "- prefer larger models for complex or multi-step tasks\n"
            "- if uncertain, use the configured fallback model\n"
        )

        prompt = system_prompt + "\nUser prompt:\n" + user_prompt

        # Attempt to find provider for decision model in registry
        provider_name = None
        for m in models:
            if m.get("id") == self.model_id:
                provider_name = m.get("provider")
                break

        provider = self.default_provider

        # Currently we only have an Ollama provider adapter available; use default
        # In future this can select other provider adapters based on provider_name

        start = time.time()
        try:
            resp = await provider.forward_request(self.model_id, {"prompt": prompt, "stream": False}, stream=False)
            latency_ms = int((time.time() - start) * 1000)

            # response may be a dict with 'response' text
            text = None
            if isinstance(resp, dict):
                # Ollama-style
                text = resp.get("response") or resp.get("text") or json.dumps(resp)
            else:
                text = str(resp)

            # try to parse returned text as JSON
            selected = None
            reason = ""
            try:
                parsed = json.loads(text)
                selected = parsed.get("selected_model")
                reason = parsed.get("reason")
            except Exception:
                # best-effort: attempt to find JSON blob within the text
                try:
                    start_idx = text.find("{")
                    end_idx = text.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        parsed = json.loads(text[start_idx:end_idx+1])
                        selected = parsed.get("selected_model")
                        reason = parsed.get("reason")
                except Exception:
                    selected = None

            if not selected:
                return {"selected_model": None, "reason": "decision model did not return a valid selection", "provider": provider_name, "latency_ms": latency_ms, "token_count": 0}

            return {"selected_model": selected, "reason": reason or "", "provider": provider_name or "ollama", "latency_ms": latency_ms, "token_count": 0}

        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            return {"selected_model": None, "reason": f"decision model call failed: {e}", "provider": provider_name, "latency_ms": latency_ms, "token_count": 0}

    def persist_decision(self, prompt: str, selected_model: str, provider: Optional[str], reason: str, latency_ms: int, token_count: int = 0, routing_mode: str = "auto") -> None:
        """Persist a routing decision to the decision_backlog table.

        routing_mode: 'auto' | 'passthrough' etc.
        """
        db = model_registry.get_db()
        with db() as conn:
            cur = conn.cursor()
            prompt_hash = json.dumps(prompt)[:200]
            preview = prompt[:200]
            try:
                cur.execute(
                    "INSERT INTO decision_backlog (prompt_hash, prompt_preview, selected_model, provider, reason, latency_ms, token_count, request_data, routing_mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (prompt_hash, preview, selected_model, provider, reason, latency_ms, token_count, None, routing_mode)
                )
            except Exception:
                # If column doesn't exist (older DB), fall back to insert without routing_mode
                cur.execute(
                    "INSERT INTO decision_backlog (prompt_hash, prompt_preview, selected_model, provider, reason, latency_ms, token_count, request_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (prompt_hash, preview, selected_model, provider, reason, latency_ms, token_count, None)
                )
