"""
Model attributes and catalog used by the routing logic.
Separated from the main `ollama_router.py` to keep that file smaller.
"""
MODEL_ATTRIBUTES = {
    "qwen2.5-coder:7b": {"speed": 8, "complexity": 7, "size_gb": 4.7, "preferred_for": ["code", "debugging"]},
    "deepseek-r1:latest": {"speed": 3, "complexity": 10, "size_gb": 30, "preferred_for": ["reasoning", "analysis", "math"]},
    "llava:latest": {"speed": 4, "complexity": 6, "size_gb": 6, "preferred_for": ["vision", "image_analysis"]},
    "nemotron-3-nano:latest": {"speed": 10, "complexity": 4, "size_gb": 1.4, "preferred_for": ["fast", "simple"]},
    "mistral:latest": {"speed": 7, "complexity": 6, "size_gb": 4.4, "preferred_for": ["general", "conversation"]},
    "qwen2.5:7b": {"speed": 8, "complexity": 6, "size_gb": 4.7, "preferred_for": ["general", "qa", "writing"]},
    "llama2-uncensored:latest": {"speed": 5, "complexity": 7, "size_gb": 3.8, "preferred_for": ["creative"]},
}
