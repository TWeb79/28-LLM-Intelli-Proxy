"""
Caching utilities for LLM IntelliProxy.

Provides LRU cache for prompt classifications, pre-computed model scores,
and request statistics tracking.
"""
import hashlib
from collections import OrderedDict, defaultdict
from datetime import datetime
from threading import Lock
from typing import Dict, Optional, Any, List


class ClassificationCache:
    """LRU cache for prompt classifications.

    Caches classification results (e.g., 'code', 'vision', 'reasoning')
    based on a hash of the first 200 characters of the prompt.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the classification cache.

        Args:
            max_size: Maximum number of entries to store (default 1000)
        """
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.max_size = max_size
        self.hits: int = 0
        self.misses: int = 0

    def _get_key(self, prompt: str) -> str:
        """Generate MD5 hash of prompt prefix.

        Args:
            prompt: The user prompt

        Returns:
            32-character hex MD5 hash of first 200 chars
        """
        return hashlib.md5(prompt[:200].encode()).hexdigest()

    def get(self, prompt: str) -> Optional[str]:
        """Get cached classification for a prompt.

        Args:
            prompt: The user prompt

        Returns:
            Cached classification string or None if not found
        """
        key = self._get_key(prompt)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, prompt: str, classification: str) -> None:
        """Store a classification in the cache.

        Evicts least-recently-used entry if cache is full.

        Args:
            prompt: The user prompt
            classification: Classification result to cache
        """
        key = self._get_key(prompt)
        self.cache[key] = classification
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, total, hit_rate, and size
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self.cache)
        }


class ModelScoreCache:
    """Thread-safe cache for pre-computed model attributes/scores."""

    def __init__(self):
        """Initialize the score cache."""
        self.scores: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()

    def compute_scores(self, model_attrs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Store pre-computed model attributes/scores.

        Args:
            model_attrs: Dictionary mapping model names to attribute dicts

        Returns:
            The stored scores dictionary
        """
        with self.lock:
            self.scores = model_attrs
            return self.scores

    def get_score(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get score/attributes for a model.

        Args:
            model_name: Model identifier

        Returns:
            Attribute dict or None if not found
        """
        with self.lock:
            return self.scores.get(model_name)


class Statistics:
    """Thread-safe usage statistics tracker."""

    def __init__(self):
        """Initialize statistics."""
        self.total_requests: int = 0
        self.models: defaultdict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "total_time": 0.0})
        self.categories: defaultdict[str, int] = defaultdict(int)
        self.last_update: datetime = datetime.now()
        self.lock = Lock()

    def record_request(self, model: str, category: str, execution_time: float) -> None:
        """Record a request in statistics.

        Args:
            model: Model used
            category: Task classification
            execution_time: Time taken in seconds
        """
        with self.lock:
            self.total_requests += 1
            self.models[model]["count"] += 1
            self.models[model]["total_time"] += execution_time
            self.categories[category] += 1
            self.last_update = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary.

        Returns:
            Dictionary with total_requests, models, model_avg_times, categories, last_update
        """
        with self.lock:
            model_avg_times = {}
            for model, data in self.models.items():
                if data["count"] > 0:
                    model_avg_times[model] = round(data["total_time"] / data["count"], 2)

            return {
                "total_requests": self.total_requests,
                "models": dict(self.models),
                "model_avg_times": model_avg_times,
                "categories": dict(self.categories),
                "last_update": self.last_update.isoformat()
            }
