

# ============================================================================
# DECISION ENGINE & ROUTING LOGIC
# ============================================================================

class ClassificationCache:
    """LRU cache for prompt classifications."""
    
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, prompt: str) -> str:
        return hashlib.md5(prompt[:200].encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        key = self._get_key(prompt)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, prompt: str, classification: str):
        key = self._get_key(prompt)
        self.cache[key] = classification
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def stats(self) -> Dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": f"{hit_rate:.1f}%"}

class DecisionEngine:
    """Intelligent routing decision engine."""
    
    def __init__(self, registry: ProviderRegistry):
        self.registry = registry
        self.classification_cache = ClassificationCache()
        self.fallback_model = CONFIG["proxy"]["fallback_model"]
    
    async def classify_task(self, prompt: str, use_llm: bool = True) -> str:
        """Classify task type (cached + LLM-assisted)."""
        # Check cache first
        cached = self.classification_cache.get(prompt)
        if cached:
            return cached
        
        # Fast heuristic classification
        classification = self._heuristic_classify(prompt)
        
        # Enhance with LLM if available and use_llm is True
        if use_llm:
            llm_classification = await self._llm_classify(prompt)
            if llm_classification:
                classification = llm_classification
        
        self.classification_cache.put(prompt, classification)
        return classification
    
    def _heuristic_classify(self, prompt: str) -> str:
        """Fast keyword-based classification."""
        prompt_lower = prompt.lower()
        
        # Code-related
        if any(w in prompt_lower for w in ["code", "function", "class", "debug", "fix", "program", "script", "api", "algorithm"]):
            return "code"
        
        # Vision/Image
        if any(w in prompt_lower for w in ["image", "picture", "photo", "visual", "describe this", "see "]):
            return "vision"
        
        # Reasoning/Analysis
        if any(w in prompt_lower for w in ["prove", "analyze", "theorem", "step by step", "reason", "explain in detail", "breakdown"]):
            return "reasoning"
        
        # Math
        if any(w in prompt_lower for w in ["calculate", "math", "equation", "formula", "solve", "integral", "derivative"]):
            return "math"
        
        # Creative writing
        if any(w in prompt_lower for w in ["write", "story", "poem", "essay", "article", "creative"]):
            return "chat"
        
        return "general"
    
    async def _llm_classify(self, prompt: str) -> Optional[str]:
        """Use decision model for classification."""
        decision_model = CONFIG["decision"]["model"]
        if not decision_model:
            return None
        
        provider = self.registry.get_provider_for_model(decision_model)
        if not provider:
            return None
        
        system_prompt = "Classify the user's prompt into one of: reasoning, coding, chat, vision, image, math, tool-use, embedding. Return only the category name."
        
        try:
            result = await provider.forward_chat(decision_model, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
            
            response = ""
            if "message" in result:
                response = result["message"].get("content", "").lower().strip()
            elif "response" in result:
                response = result["response"].lower().strip()
            
            valid_categories = ["reasoning", "coding", "chat", "vision", "image", "math", "tool-use", "embedding"]
            for cat in valid_categories:
                if cat in response:
                    return cat
        except Exception as e:
            logging.debug(f"LLM classification failed: {e}")
        
        return None
    
    def analyze_complexity(self, prompt: str) -> int:
        """Analyze prompt complexity (1-10)."""
        complexity = 3  # Default
        prompt_lower = prompt.lower()
        
        # Keyword complexity
        if any(w in prompt_lower for w in ["analyze", "debug", "optimize", "architect", "theorem"]):
            complexity += 3
        elif any(w in prompt_lower for w in ["explain", "describe", "compare"]):
            complexity += 1
        elif any(w in prompt_lower for w in ["implement", "write code", "create"]):
            complexity += 2
        
        # Length factor
        word_count = len(prompt.split())
        complexity += min(word_count // 200, 4)
        
        return min(max(complexity, 1), 10)
    
    async def select_model(self, category: str, complexity: int, 
                          exclude_models: List[str] = None) -> Optional[str]:
        """Select best model for task."""
        exclude_models = exclude_models or []
        
        # Get enabled models from registry
        enabled_models = await self.registry.get_enabled_models()
        
        if not enabled_models:
            return None
        
        # Filter by category preference
        scored_models = []
        for model in enabled_models:
            if model["id"] in exclude_models:
                continue
            
            provider = self.registry.providers.get(model["provider"])
            if not provider or not provider.enabled:
                continue
            
            # Score based on category match and complexity
            score = self._score_model(model, category, complexity)
            scored_models.append((score, model))
        
        if not scored_models:
            return None
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[0], reverse=True)
        return scored_models[0][1]["id"]
    
    def _score_model(self, model: Dict, category: str, complexity: int) -> float:
        """Score a model for a given task."""
        score = 0.0
        
        # Category match bonus
        model_category = model.get("category", "")
        if model_category == category:
            score += 10
        elif category in str(model.get("description", "")).lower():
            score += 5
        
        # Complexity matching
        # Models in 'reasoning' category get bonus for high complexity
        if category == "reasoning" and complexity > 6:
            if model_category in ("reasoning", "coding"):
                score += 8
        
        # Prefer faster models for simple tasks
        if complexity <= 3:
            score += 5
        
        # Recent models (by last_seen) get small bonus
        if model.get("last_seen"):
            score += 1
        
        return score
    
    async def generate_routing_prompt(self) -> str:
        """Generate dynamic system prompt from live model registry."""
        enabled_models = await self.registry.get_enabled_models()
        
        yaml_list = ""
        for model in enabled_models[:20]:  # Limit to 20 for prompt size
            yaml_list += f"""
  - id: {model['id']}
    provider: {model['provider']}
    category: {model.get('category', 'unknown') or 'unknown'}
    description: {model.get('description', 'No description') or 'No description'}
    enabled: {model.get('enabled', True)}
"""
        
        prompt = f"""You are an intelligent LLM routing agent. Based on the user's prompt, 
select the single most appropriate model from the registry below.

Return ONLY valid JSON:
{{
  "selected_model": "<model_id>",
  "reason": "<one sentence explanation>"
}}

Available models:
{yaml_list}

Selection criteria:
- coding / debugging → prefer category: coding
- multi-step logic, math, planning → prefer category: reasoning  
- creative writing, general chat → prefer category: chat
- image understanding or generation → prefer category: image or vision
- prefer smaller/faster models for simple tasks
- prefer larger models for complex or multi-step tasks
- if uncertain, use the configured fallback model
"""
        return prompt
    
    async def make_decision(self, prompt: str, complexity: int = None) -> Dict:
        """Make routing decision using LLM or heuristics."""
        if complexity is None:
            complexity = self.analyze_complexity(prompt)
        
        # Check if decision model is configured
        decision_model = CONFIG["decision"]["model"]
        
        if decision_model:
            # Try LLM-based decision with dynamic prompt
            try:
                decision_provider = self.registry.get_provider_for_model(decision_model)
                if decision_provider:
                    system_prompt = await self.generate_routing_prompt()
                    
                    result = await decision_provider.forward_chat(decision_model, [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ])
                    
                    # Parse JSON response
                    response_text = ""
                    if "message" in result:
                        response_text = result["message"].get("content", "")
                    elif "response" in result:
                        response_text = result["response"]
                    
                    if response_text:
                        import re
                        json_match = re.search(r'\{[^{}]*"selected_model"[^{}]*"reason"[^{}]*\}', response_text)
                        if json_match:
                            decision = json.loads(json_match.group())
                            return {
                                "selected_model": decision["selected_model"],
                                "reason": decision["reason"],
                                "method": "llm",
                                "complexity": complexity
                            }
            except Exception as e:
                logging.warning(f"LLM-based decision failed: {e}")
        
        # Fallback to heuristic-based decision
        classification = await self.classify_task(prompt, use_llm=False)
        selected_model = await self.select_model(classification, complexity)
        
        if not selected_model:
            selected_model = self.fallback_model
        
        return {
            "selected_model": selected_model,
            "reason": f"Heuristic routing: {classification} task (complexity: {complexity})",
            "method": "heuristic",
            "classification": classification,
            "complexity": complexity
        }