"""
Model Availability Monitor for LLM IntelliProxy.

Continuously monitors model health and availability with background probes,
ETA prediction, and real-time status updates.
"""
import asyncio
import logging
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sqlite3
import json

from services.database import get_db


class ModelAvailabilityState:
    """Represents the current availability state of a model."""
    
    def __init__(self, model_id: str, provider: str):
        self.model_id = model_id
        self.provider = provider
        self.current_status = "unknown"
        self.unavailable_since = None
        self.last_available_at = None
        self.last_checked_at = None
        self.last_error_code = None
        self.last_error_message = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.estimated_recovery_at = None
        self.eta_confidence = None


class ModelAvailabilityMonitor:
    """Background monitor for model availability and health."""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.check_interval = settings.get("check_interval_seconds", 60)
        self.probe_timeout = settings.get("probe_timeout_seconds", 10)
        self.recovery_window = settings.get("recovery_window_hours", 168)
        self.min_samples = settings.get("min_samples_for_eta", 3)
        self.mark_unavailable_after = settings.get("mark_unavailable_after", 2)
        self.mark_available_after = settings.get("mark_available_after", 1)
        
        self._running = False
        self._task = None
        self._http_client = None
        
    async def start(self):
        """Start the availability monitoring background task."""
        if self._running:
            return
            
        self._running = True
        self._http_client = httpx.AsyncClient(timeout=self.probe_timeout)
        
        # Load initial state from database
        await self._load_initial_state()
        
        # Start background task
        self._task = asyncio.create_task(self._monitor_loop())
        logging.info("Model availability monitor started")
    
    async def stop(self):
        """Stop the availability monitoring background task."""
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        if self._http_client:
            await self._http_client.aclose()
        
        logging.info("Model availability monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop running in background."""
        while self._running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Error in availability monitor: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_check(self):
        """Perform health checks on all enabled models."""
        from services.registry import list_models
        
        models = list_models()
        tasks = []
        
        for model in models:
            if not model.get("enabled", True):
                continue
                
            model_id = model.get("id", "")
            provider = model.get("provider", "")
            
            if not model_id or not provider:
                continue
                
            if provider == "ollama":
                tasks.append(self._probe_ollama_model(model_id, provider))
            elif provider == "nvidia":
                tasks.append(self._probe_nvidia_model(model_id, provider))
            else:
                # Skip image models and unknown providers
                model_id_lower = str(model_id).lower()
                if "llava" not in model_id_lower and "vision" not in model_id_lower:
                    tasks.append(self._probe_generic_model(model_id, provider))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _probe_ollama_model(self, model_id: str, provider: str) -> None:
        """Probe an Ollama model for availability."""
        try:
            from services.router import get_http_client
            client = await get_http_client()
            
            # Use a simple ping prompt
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_id,
                    "prompt": "ping",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=self.probe_timeout
            )
            
            success = response.status_code == 200
            response_time = response.elapsed.total_seconds() * 1000
            
            await self._update_model_status(
                model_id, provider, success, response_time,
                None if success else str(response.status_code),
                None if success else response.text[:200]
            )
            
        except Exception as e:
            await self._update_model_status(
                model_id, provider, False, 0, "connection_error", str(e)
            )
    
    async def _probe_nvidia_model(self, model_id: str, provider: str) -> None:
        """Probe a NVIDIA NIM model for availability."""
        try:
            # This would need NVIDIA API key and endpoint configuration
            # For now, mark as unknown to avoid probing cloud services
            await self._update_model_status(
                model_id, provider, None, 0, "unknown", "Cloud provider probing not implemented"
            )
        except Exception as e:
            await self._update_model_status(
                model_id, provider, False, 0, "probe_error", str(e)
            )
    
    async def _probe_generic_model(self, model_id: str, provider: str) -> None:
        """Probe a generic model (fallback)."""
        try:
            # For unknown providers, mark as unknown
            await self._update_model_status(
                model_id, provider, None, 0, "unknown", f"Provider {provider} not supported for probing"
            )
        except Exception as e:
            await self._update_model_status(
                model_id, provider, False, 0, "probe_error", str(e)
            )
    
    async def _update_model_status(
        self, 
        model_id: str, 
        provider: str, 
        success: Optional[bool], 
        response_time: float,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Update model availability status in database."""
        now = datetime.utcnow().isoformat()
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get current state
            cursor.execute("""
                SELECT * FROM model_availability_state 
                WHERE model_id = ? AND provider = ?
            """, (model_id, provider))
            
            row = cursor.fetchone()
            if row:
                current_status = row["current_status"]
                consecutive_failures = row["consecutive_failures"]
                consecutive_successes = row["consecutive_successes"]
            else:
                current_status = "unknown"
                consecutive_failures = 0
                consecutive_successes = 0
            
            # Update counters based on probe result
            if success is True:
                consecutive_successes += 1
                consecutive_failures = 0
            elif success is False:
                consecutive_failures += 1
                consecutive_successes = 0
            else:
                # Unknown result, don't change counters
                pass
            
            # Determine new status with hysteresis
            new_status = current_status
            if success is True and consecutive_successes >= self.mark_available_after:
                new_status = "available"
            elif success is False and consecutive_failures >= self.mark_unavailable_after:
                new_status = "unavailable"
            elif success is None:
                new_status = "unknown"
            
            # Handle status transitions
            unavailable_since = None
            last_available_at = None
            
            if new_status == "unavailable" and current_status != "unavailable":
                unavailable_since = now
                # Set last_available_at from previous state
                if current_status == "available":
                    last_available_at = row.get("last_checked_at") if row else None
            elif new_status == "available" and current_status == "unavailable":
                last_available_at = now
                unavailable_since = None
            else:
                # Keep existing values
                unavailable_since = row.get("unavailable_since") if row else None
                last_available_at = row.get("last_available_at") if row else None
            
            # Calculate ETA if becoming unavailable
            estimated_recovery_at = None
            eta_confidence = None
            
            if new_status == "unavailable" and current_status != "unavailable":
                eta_result = await self._compute_recovery_eta(model_id, provider)
                if eta_result:
                    estimated_recovery_at = eta_result.get("estimated_recovery_at")
                    eta_confidence = eta_result.get("confidence")
            
            # Update availability state
            cursor.execute("""
                INSERT OR REPLACE INTO model_availability_state (
                    model_id, provider, current_status, unavailable_since,
                    last_available_at, last_checked_at, last_error_code,
                    last_error_message, consecutive_failures, consecutive_successes,
                    estimated_recovery_at, eta_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, provider, new_status, unavailable_since,
                last_available_at, now, error_code, error_message,
                consecutive_failures, consecutive_successes,
                estimated_recovery_at, eta_confidence
            ))
            
            # Log the probe result
            cursor.execute("""
                INSERT INTO model_availability_log (
                    model_id, provider, status, checked_at, response_time_ms,
                    error_code, error_message, consecutive_failures, consecutive_successes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, provider, "available" if success else "unavailable",
                now, response_time, error_code, error_message,
                consecutive_failures, consecutive_successes
            ))
    
    async def _compute_recovery_eta(self, model_id: str, provider: str) -> Optional[Dict[str, Any]]:
        """Compute estimated recovery time based on historical data."""
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get downtime events from the last recovery_window hours
            cutoff_time = (datetime.utcnow() - timedelta(hours=self.recovery_window)).isoformat()
            
            cursor.execute("""
                SELECT unavailable_since, last_available_at, 
                       (julianday(last_available_at) - julianday(unavailable_since)) * 24 * 60 as downtime_minutes
                FROM model_availability_log l
                JOIN model_availability_state s ON l.model_id = s.model_id AND l.provider = s.provider
                WHERE l.model_id = ? AND l.provider = ? 
                AND l.status = 'unavailable' 
                AND s.unavailable_since IS NOT NULL 
                AND s.last_available_at IS NOT NULL
                AND l.checked_at > ?
                ORDER BY l.checked_at DESC
            """, (model_id, provider, cutoff_time))
            
            events = cursor.fetchall()
            
            if len(events) < self.min_samples:
                return None
            
            # Calculate average downtime
            downtimes = [row["downtime_minutes"] for row in events if row["downtime_minutes"] > 0]
            if not downtimes:
                return None
            
            avg_downtime = sum(downtimes) / len(downtimes)
            
            # Calculate standard deviation for confidence
            if len(downtimes) > 1:
                variance = sum((d - avg_downtime) ** 2 for d in downtimes) / len(downtimes)
                stddev = variance ** 0.5
                
                # Determine confidence based on coefficient of variation
                cv = stddev / avg_downtime if avg_downtime > 0 else 0
                if cv < 0.2:
                    confidence = "high"
                elif cv < 0.5:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                confidence = "low"
            
            estimated_recovery = datetime.utcnow() + timedelta(minutes=avg_downtime)
            
            return {
                "estimated_recovery_at": estimated_recovery.isoformat(),
                "confidence": confidence,
                "based_on_samples": len(downtimes),
                "avg_downtime_minutes": avg_downtime
            }
    
    async def _load_initial_state(self):
        """Load existing availability state from database."""
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Ensure tables exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_availability_state (
                    model_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    current_status TEXT NOT NULL CHECK (current_status IN ('available', 'unavailable', 'unknown')),
                    unavailable_since TEXT,
                    last_available_at TEXT,
                    last_checked_at TEXT NOT NULL,
                    last_error_code TEXT,
                    last_error_message TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    consecutive_successes INTEGER DEFAULT 0,
                    estimated_recovery_at TEXT,
                    eta_confidence TEXT CHECK (eta_confidence IN ('low', 'medium', 'high')),
                    PRIMARY KEY (model_id, provider)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_availability_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('available', 'unavailable')),
                    checked_at TEXT NOT NULL,
                    response_time_ms INTEGER,
                    error_code TEXT,
                    error_message TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    consecutive_successes INTEGER DEFAULT 0
                )
            """)
    
    def get_model_status(self, model_id: str, provider: str) -> Optional[Dict[str, Any]]:
        """Get current availability status for a model."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM model_availability_state 
                WHERE model_id = ? AND provider = ?
            """, (model_id, provider))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_all_model_statuses(self) -> List[Dict[str, Any]]:
        """Get availability status for all models."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_availability_state")
            return [dict(row) for row in cursor.fetchall()]
    
    async def report_external_failure(
        self, 
        model_id: str, 
        provider: str, 
        error_code: str, 
        error_message: str,
        timestamp: datetime = None
    ):
        """Report a failure from external source (e.g., fallback engine)."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        await self._update_model_status(
            model_id, provider, False, 0, error_code, error_message
        )


# Global instance
_availability_monitor = None


def get_availability_monitor(settings: Dict[str, Any] = None) -> ModelAvailabilityMonitor:
    """Get the global availability monitor instance."""
    global _availability_monitor
    if _availability_monitor is None:
        _availability_monitor = ModelAvailabilityMonitor(settings or {})
    return _availability_monitor