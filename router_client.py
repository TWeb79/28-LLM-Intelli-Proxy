#!/usr/bin/env python3
"""
Ollama Router Client
Simple client to interact with the Intelligent Router Proxy
"""

import requests
import json
from typing import Optional, Dict, List
from tabulate import tabulate

class OllamaRouterClient:
    """Client for the Intelligent Router Proxy"""
    
    def __init__(self, router_url: str = "http://localhost:8000"):
        self.router_url = router_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check router health"""
        response = self.session.get(f"{self.router_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict:
        """List all available models"""
        response = self.session.get(f"{self.router_url}/models")
        response.raise_for_status()
        return response.json()
    
    def list_models_by_category(self, category: str) -> List[str]:
        """List models in a specific category"""
        response = self.session.get(f"{self.router_url}/models/{category}")
        response.raise_for_status()
        return response.json()["models"]
    
    def classify(self, prompt: str) -> Dict:
        """Classify a task (without executing)"""
        response = self.session.post(
            f"{self.router_url}/classify",
            json={"prompt": prompt}
        )
        response.raise_for_status()
        return response.json()
    
    def process(self, prompt: str, task_type: Optional[str] = None, stream: bool = False) -> Dict:
        """
        Process a task with intelligent routing
        
        Args:
            prompt: The task/question to process
            task_type: Optional category override (code, vision, reasoning, uncensored, general)
            stream: Whether to stream response (not yet implemented)
        
        Returns:
            Dict with result, model_used, classification, timestamp
        """
        response = self.session.post(
            f"{self.router_url}/task",
            json={
                "prompt": prompt,
                "task_type": task_type,
                "stream": stream
            }
        )
        response.raise_for_status()
        return response.json()
    
    def print_models(self):
        """Pretty print available models"""
        data = self.list_models()
        
        print("\n" + "="*70)
        print("📊 AVAILABLE MODELS")
        print("="*70)
        
        for category, models in data["categories"].items():
            if models:
                print(f"\n🏷️  {category.upper()}")
                for model in models:
                    size_gb = data["models"][model]["size"] / (1024**3)
                    print(f"   • {model} ({size_gb:.1f} GB)")
        
        print("\n" + "="*70 + "\n")
    
    def print_classification(self, prompt: str):
        """Pretty print classification results"""
        result = self.classify(prompt)
        
        print("\n" + "="*70)
        print("🎯 TASK CLASSIFICATION")
        print("="*70)
        print(f"\n📝 Prompt: {result['prompt'][:100]}...")
        print(f"🏷️  Classification: {result['classification'].upper()}")
        print(f"📍 Recommended Model: {result['recommended_model']}")
        print(f"🔀 Available in Category: {', '.join(result['available_models_in_category'])}")
        print("\n" + "="*70 + "\n")
    
    def print_result(self, result: Dict):
        """Pretty print execution result"""
        print("\n" + "="*70)
        print("✅ TASK RESULT")
        print("="*70)
        print(f"\n🏷️  Classification: {result['task_classification'].upper()}")
        print(f"📍 Model Used: {result['model_used']}")
        print(f"⏰ Timestamp: {result['timestamp']}")
        print(f"\n📄 Result:\n{result['result']}")
        print("\n" + "="*70 + "\n")

# Example usage functions
def example_usage():
    """Example: Using the router client"""
    
    print("\n🚀 Ollama Intelligent Router - Client Examples\n")
    
    client = OllamaRouterClient()
    
    # 1. Check health
    print("1️⃣  Checking router health...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Models available: {health['models_available']}\n")
    
    # 2. List models
    print("2️⃣  Listing all models...")
    client.print_models()
    
    # 3. Classify a task
    print("3️⃣  Classifying tasks...")
    test_prompts = [
        "Write a Python function to calculate fibonacci",
        "Analyze this image for me",
        "Explain quantum computing step by step",
    ]
    
    for prompt in test_prompts:
        client.print_classification(prompt)
    
    # 4. Process tasks
    print("4️⃣  Processing tasks with intelligent routing...\n")
    
    tasks = [
        {
            "prompt": "Write a Python function to find the longest common subsequence",
            "task_type": None  # Auto-classify
        },
        {
            "prompt": "Solve this math problem: If x + 5 = 12, what is x?",
            "task_type": None
        },
        {
            "prompt": "Write a Python async server that handles websocket connections",
            "task_type": "code"  # Force code category
        },
    ]
    
    for task in tasks:
        print(f"Processing: {task['prompt'][:60]}...")
        result = client.process(task["prompt"], task_type=task.get("task_type"))
        client.print_result(result)
        print()

def quick_task(prompt: str, task_type: Optional[str] = None):
    """Quick task processing"""
    client = OllamaRouterClient()
    try:
        result = client.process(prompt, task_type=task_type)
        client.print_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick mode: python client.py "Your prompt here"
        prompt = " ".join(sys.argv[1:])
        quick_task(prompt)
    else:
        # Demo mode
        example_usage()
