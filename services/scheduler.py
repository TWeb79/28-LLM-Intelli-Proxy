"""
Background scheduler for LLM IntelliProxy.

Handles periodic model list refresh from all configured providers.
Runs as a background thread to avoid blocking the main application.
"""
import asyncio
import threading
import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from providers.base_provider import LLMProvider, ProviderRegistry
from services import registry as model_registry
from services.assessor import Assessor


class ModelRefreshScheduler:
    """Background scheduler for periodic model list refresh.
    
    Periodically fetches model lists from all registered providers
    and updates the unified model registry.
    """
    
    def __init__(
        self,
        refresh_interval_minutes: int = 15,
        assessor: Optional[Assessor] = None
    ):
        """Initialize the scheduler.
        
        Args:
            refresh_interval_minutes: Default refresh interval (can be overridden per provider)
            assessor: Optional AI assessor for new models
        """
        self.refresh_interval_minutes = refresh_interval_minutes
        self.assessor = assessor
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_refresh: Optional[datetime] = None
        self._refresh_count = 0
        self._error_count = 0
        self._on_refresh_callbacks: List[Callable] = []
    
    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._running:
            return
        
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[Scheduler] Started model refresh scheduler (interval: {self.refresh_interval_minutes} min)")
    
    def stop(self) -> None:
        """Stop the background scheduler thread."""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        print("[Scheduler] Stopped model refresh scheduler")
    
    def _run_loop(self) -> None:
        """Main scheduler loop running in background thread."""
        # Initial refresh on startup
        self._do_refresh()
        
        while not self._stop_event.is_set():
            # Wait for the refresh interval
            interval_seconds = self.refresh_interval_minutes * 60
            self._stop_event.wait(timeout=interval_seconds)
            
            if not self._stop_event.is_set():
                self._do_refresh()
    
    def _do_refresh(self) -> None:
        """Perform model refresh from all providers."""
        print(f"[Scheduler] Starting model refresh (attempt #{self._refresh_count + 1})")
        
        try:
            # Get all registered providers
            providers = ProviderRegistry.get_all()
            
            if not providers:
                print("[Scheduler] No providers registered, skipping refresh")
                return
            
            new_models = []
            
            for provider in providers:
                try:
                    # Get provider-specific refresh interval
                    interval = provider.refresh_interval_minutes or self.refresh_interval_minutes
                    
                    # Check if this provider should be refreshed
                    # For simplicity, refresh all providers on each cycle
                    # In production, track last refresh per provider
                    
                    models = asyncio.run(self._fetch_provider_models(provider))
                    
                    # Upsert each model to the registry
                    for model in models:
                        model_registry.upsert_model(
                            provider=model.get("provider", provider.name),
                            model_id=model.get("id", ""),
                            source_url=model.get("source_url", provider.base_url),
                            category=model.get("category"),
                            description=model.get("description"),
                            context_window=model.get("context_window"),
                            enabled=True,
                            assessed=False  # Will be set to True if already assessed
                        )
                        new_models.append(model)
                    
                    print(f"[Scheduler] Refreshed {len(models)} models from {provider.name}")
                    
                except Exception as e:
                    print(f"[Scheduler] Error refreshing {provider.name}: {e}")
                    self._error_count += 1
            
            self._last_refresh = datetime.utcnow()
            self._refresh_count += 1
            
            # Trigger AI assessment for new models
            if self.assessor and new_models:
                self._trigger_assessments(new_models)
            
            # Notify callbacks
            for callback in self._on_refresh_callbacks:
                try:
                    callback(new_models)
                except Exception as e:
                    print(f"[Scheduler] Callback error: {e}")
            
            print(f"[Scheduler] Refresh completed. Total models: {len(new_models)}")
            
        except Exception as e:
            print(f"[Scheduler] Refresh failed: {e}")
            self._error_count += 1
    
    async def _fetch_provider_models(self, provider: LLMProvider) -> List[Dict[str, Any]]:
        """Fetch models from a single provider asynchronously.
        
        Args:
            provider: Provider instance
            
        Returns:
            List of model dictionaries
        """
        return await provider.list_models()
    
    def _trigger_assessments(self, models: List[Dict[str, Any]]) -> None:
        """Trigger AI assessment for newly discovered models.
        
        Args:
            models: List of model dictionaries
        """
        if not self.assessor:
            return
        
        # Find models that haven't been assessed
        for model in models:
            model_id = model.get("id", "")
            provider = model.get("provider", "")
            
            # Check if already assessed
            existing = model_registry.list_models()
            for ex in existing:
                if ex.get("id") == model_id and ex.get("provider") == provider:
                    if ex.get("assessed"):
                        continue
                    break
            
            # Trigger async assessment
            try:
                asyncio.run(self._assess_model_async(provider, model_id))
            except Exception as e:
                print(f"[Scheduler] Assessment trigger failed for {model_id}: {e}")
    
    async def _assess_model_async(self, provider: str, model_id: str) -> None:
        """Run model assessment asynchronously.
        
        Args:
            provider: Provider name
            model_id: Model identifier
        """
        if self.assessor:
            await self.assessor.assess_model(provider, model_id)
    
    def refresh_now(self) -> None:
        """Trigger an immediate refresh (bypasses the schedule)."""
        print("[Scheduler] Triggering immediate refresh")
        thread = threading.Thread(target=self._do_refresh, daemon=True)
        thread.start()
    
    def on_refresh(self, callback: Callable) -> None:
        """Register a callback to be called after each refresh.
        
        Args:
            callback: Function that receives the list of models
        """
        self._on_refresh_callbacks.append(callback)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get scheduler statistics.
        
        Returns:
            Dictionary with refresh count, error count, last refresh time
        """
        return {
            "running": self._running,
            "refresh_count": self._refresh_count,
            "error_count": self._error_count,
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "interval_minutes": self.refresh_interval_minutes,
        }


# Global scheduler instance
_scheduler: Optional[ModelRefreshScheduler] = None


def get_scheduler() -> ModelRefreshScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ModelRefreshScheduler()
    return _scheduler


def start_scheduler(
    refresh_interval_minutes: int = 15,
    assessor: Optional[Assessor] = None
) -> ModelRefreshScheduler:
    """Start the global scheduler.
    
    Args:
        refresh_interval_minutes: Refresh interval in minutes
        assessor: Optional AI assessor for new models
        
    Returns:
        The scheduler instance
    """
    scheduler = get_scheduler()
    scheduler.refresh_interval_minutes = refresh_interval_minutes
    scheduler.assessor = assessor
    scheduler.start()
    return scheduler


def stop_scheduler() -> None:
    """Stop the global scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
        _scheduler = None