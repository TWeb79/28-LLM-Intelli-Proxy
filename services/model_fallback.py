"""
Model Fallback Engine for LLM IntelliProxy.

Provides intelligent model fallback with similarity-based ranking and retry logic.
"""
import asyncio
import logging
import httpx
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json

from services.database import get_db
from services.model_availability import get_availability_monitor


class FallbackAttempt:
    """Represents a single fallback attempt."""
    
    def __init__(
        self,
        model_id: str,
        provider: str,
        attempt_number: int,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        latency_ms: int = 0,
        timestamp: Optional[str] = None
    ):
        self.model_id = model_id
        self.provider = provider
        self.attempt_number = attempt_number
        self.error_code = error_code
        self.error_message = error_message
        self.latency_ms = latency_ms
        self.timestamp = timestamp or datetime.utcnow().isoformat()


class FallbackResult:
    """Result of fallback execution."""
    
    def __init__(
        self,
        response: Dict[str, Any],
        model_used: str,
        provider_used: str,
        attempts: List[FallbackAttempt],
        fallback_used: bool
    ):
        self.response = response
        self.model_used = model_used
        self.provider_used = provider_used
        self.attempts = attempts
        self.fallback_used = fallback_used


class ModelFallbackEngine:
    """Intelligent model fallback with similarity-based ranking."""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = settings.get("enabled", True)
        self.max_attempts = settings.get("max_attempts", 3)
        self.retry_on = set(settings.get("retry_on", [429, 500, 502, 503, 504, "timeout", "connection_error"]))
        self.similarity_strategy = settings.get("similarity_strategy", "category_then_size")
        self.excluded_models = set(settings.get("excluded_models", []))
    
    def rank_fallback_candidates(
        self,
        failed_model_id: str,
        registry: List[Dict[str, Any]],
        settings: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank fallback candidates based on similarity strategy.
        
        Args:
            failed_model_id: The model that just failed
            registry: List of all available models
            settings: Fallback settings
            
        Returns:
            Ordered list of candidate models
        """
        if not self.enabled:
            return []
        
        # Get current settings or use defaults
        current_settings = settings or self.settings
        strategy = current_settings.get("similarity_strategy", self.similarity_strategy)
        
        # Filter eligible candidates
        candidates = []
        for model in registry:
            model_id = model.get("id", "")
            provider = model.get("provider", "")
            
            # Skip ineligible models
            if (model_id == failed_model_id or 
                model_id in self.excluded_models or
                not model.get("enabled", True)):
                continue
            
            # Check availability status
            availability_monitor = get_availability_monitor()
            status = availability_monitor.get_model_status(model_id, provider)
            if status and status.get("current_status") == "unavailable":
                continue
            
            candidates.append(model)
        
        if not candidates:
            return []
        
        # Rank candidates based on strategy
        if strategy == "category_then_size":
            return self._rank_by_category_then_size(failed_model_id, candidates)
        elif strategy == "provider_first":
            return self._rank_by_provider_first(failed_model_id, candidates)
        elif strategy == "any":
            return self._rank_by_any(failed_model_id, candidates)
        else:
            return candidates
    
    def _rank_by_category_then_size(
        self, 
        failed_model_id: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank candidates: same category, then closest context window."""
        # Get failed model info (simplified - would need actual model registry)
        failed_category = self._get_model_category(failed_model_id)
        failed_context = self._get_model_context(failed_model_id)
        
        # Score candidates
        scored = []
        for candidate in candidates:
            score = 0
            candidate_id = candidate.get("id", "")
            candidate_provider = candidate.get("provider", "")
            
            # Same category bonus
            candidate_category = self._get_model_category(candidate_id)
            if candidate_category == failed_category:
                score += 100
            
            # Context window similarity
            candidate_context = self._get_model_context(candidate_id)
            if failed_context and candidate_context:
                context_diff = abs(failed_context - candidate_context)
                score += max(0, 50 - context_diff // 1000)  # Closer is better
            
            # Same provider bonus
            if candidate_provider == self._get_model_provider(failed_model_id):
                score += 25
            
            scored.append((score, candidate))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored]
    
    def _rank_by_provider_first(
        self, 
        failed_model_id: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank candidates: same provider, then category, then any."""
        failed_provider = self._get_model_provider(failed_model_id)
        failed_category = self._get_model_category(failed_model_id)
        
        # Group by priority
        same_provider = []
        same_category = []
        others = []
        
        for candidate in candidates:
            candidate_provider = candidate.get("provider", "")
            candidate_category = self._get_model_category(candidate.get("id", ""))
            
            if candidate_provider == failed_provider:
                same_provider.append(candidate)
            elif candidate_category == failed_category:
                same_category.append(candidate)
            else:
                others.append(candidate)
        
        return same_provider + same_category + others
    
    def _rank_by_any(
        self, 
        failed_model_id: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank candidates: any enabled model, ordered by category match."""
        failed_category = self._get_model_category(failed_model_id)
        
        # Simple category-based ranking
        same_category = []
        others = []
        
        for candidate in candidates:
            candidate_category = self._get_model_category(candidate.get("id", ""))
            if candidate_category == failed_category:
                same_category.append(candidate)
            else:
                others.append(candidate)
        
        return same_category + others
    
    def _get_model_category(self, model_id: str) -> str:
        """Get model category from model ID."""
        model_id_lower = model_id.lower()
        
        if any(word in model_id_lower for word in ["coder", "code", "program"]):
            return "code"
        elif any(word in model_id_lower for word in ["llava", "vision", "image"]):
            return "vision"
        elif any(word in model_id_lower for word in ["deepseek", "r1", "reason"]):
            return "reasoning"
        else:
            return "general"
    
    def _get_model_context(self, model_id: str) -> Optional[int]:
        """Get model context window size."""
        # This would need actual model metadata lookup
        # For now, return estimated values
        if "7b" in model_id.lower():
            return 8192
        elif "13b" in model_id.lower():
            return 16384
        elif "70b" in model_id.lower():
            return 32768
        else:
            return 4096
    
    def _get_model_provider(self, model_id: str) -> str:
        """Get model provider."""
        # This would need actual model registry lookup
        # For now, assume Ollama
        return "ollama"
    
    async def execute_with_fallback(
        self,
        request: Dict[str, Any],
        initial_model: Dict[str, Any],
        registry: List[Dict[str, Any]],
        provider_call: Callable[[Dict[str, Any], Dict[str, Any]], Any]
    ) -> FallbackResult:
        """
        Execute request with fallback on failure.
        
        Args:
            request: The original request
            initial_model: The initially selected model
            registry: All available models
            provider_call: Function to call provider with model and request
            
        Returns:
            FallbackResult with response and attempt log
        """
        if not self.enabled:
            # Direct execution without fallback
            response = await provider_call(initial_model, request)
            return FallbackResult(
                response=response,
                model_used=initial_model.get("id", ""),
                provider_used=initial_model.get("provider", ""),
                attempts=[],
                fallback_used=False
            )
        
        attempts = []
        models_to_try = [initial_model]
        
        # Add fallback candidates
        candidates = self.rank_fallback_candidates(
            initial_model.get("id", ""), 
            registry, 
            self.settings
        )
        models_to_try.extend(candidates)
        
        # Limit to max attempts
        models_to_try = models_to_try[:self.max_attempts]
        
        last_error = None
        
        for attempt_num, model in enumerate(models_to_try, 1):
            model_id = model.get("id", "")
            provider = model.get("provider", "")
            
            start_time = datetime.utcnow()
            
            try:
                response = await provider_call(model, request)
                
                # Success - return result
                return FallbackResult(
                    response=response,
                    model_used=model_id,
                    provider_used=provider,
                    attempts=attempts,
                    fallback_used=attempt_num > 1
                )
                
            except Exception as e:
                # Failure - log attempt and continue
                error_code = self._extract_error_code(e)
                error_message = str(e)
                latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                attempt = FallbackAttempt(
                    model_id=model_id,
                    provider=provider,
                    attempt_number=attempt_num,
                    error_code=error_code,
                    error_message=error_message,
                    latency_ms=latency_ms
                )
                attempts.append(attempt)
                
                # Report failure to availability monitor
                try:
                    availability_monitor = get_availability_monitor()
                    await availability_monitor.report_external_failure(
                        model_id, provider, error_code, error_message
                    )
                except Exception:
                    pass  # Best effort
                
                # Log to database
                await self._log_fallback_attempt(attempt, request)
                
                last_error = e
                
                # Check if we should retry based on error type
                if not self._should_retry(error_code):
                    break
        
        # All attempts failed
        error_response = {
            "error": {
                "message": f"All fallback models exhausted. Attempted: {[a.model_id for a in attempts]}",
                "type": "fallback_exhausted",
                "attempts": [
                    {
                        "model_id": a.model_id,
                        "provider": a.provider,
                        "error_code": a.error_code,
                        "error_message": a.error_message,
                        "latency_ms": a.latency_ms
                    }
                    for a in attempts
                ]
            }
        }
        
        raise Exception(json.dumps(error_response))
    
    def _extract_error_code(self, error: Exception) -> str:
        """Extract error code from exception."""
        error_str = str(error).lower()
        
        if "429" in error_str or "rate limit" in error_str:
            return "429"
        elif "500" in error_str or "internal server" in error_str:
            return "500"
        elif "502" in error_str or "bad gateway" in error_str:
            return "502"
        elif "503" in error_str or "service unavailable" in error_str:
            return "503"
        elif "504" in error_str or "gateway timeout" in error_str:
            return "504"
        elif "timeout" in error_str:
            return "timeout"
        elif "connection" in error_str:
            return "connection_error"
        else:
            return "unknown"
    
    def _should_retry(self, error_code: str) -> bool:
        """Check if error should trigger retry."""
        return error_code in self.retry_on
    
    async def _log_fallback_attempt(self, attempt: FallbackAttempt, request: Dict[str, Any]):
        """Log fallback attempt to database."""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # Generate request ID from request content
                request_id = hash(json.dumps(request, sort_keys=True))
                
                cursor.execute("""
                    INSERT INTO fallback_log (
                        request_id, attempt_number, model_id, provider,
                        error_code, error_message, latency_ms, fallback_used, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(request_id),
                    attempt.attempt_number,
                    attempt.model_id,
                    attempt.provider,
                    attempt.error_code,
                    attempt.error_message,
                    attempt.latency_ms,
                    attempt.attempt_number > 1,
                    attempt.timestamp
                ))
        except Exception as e:
            logging.error(f"Failed to log fallback attempt: {e}")


# Global instance
_fallback_engine = None


def get_fallback_engine(settings: Dict[str, Any] = None) -> ModelFallbackEngine:
    """Get the global fallback engine instance."""
    global _fallback_engine
    if _fallback_engine is None:
        _fallback_engine = ModelFallbackEngine(settings or {})
    return _fallback_engine