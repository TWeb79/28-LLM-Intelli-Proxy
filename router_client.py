#!/usr/bin/env python3
"""
Ollama Compatible Client for IntelliProxy
Standard Ollama client implementation with IntelliProxy as the default model.
Connects to proxy on port 9998 instead of standard Ollama port.
"""

import requests
import json
from typing import Optional, Dict, List, Iterator, Union, overload, Literal
from datetime import datetime

class OllamaClient:
    """
    Ollama-compatible client for IntelliProxy.
    Works exactly like standard Ollama client, but uses IntelliProxyLLM by default.
    """
    
    def __init__(self, proxy_url: str = "http://localhost:9998", timeout: int = 600):
        """
        Initialize the Ollama client.
        
        Args:
            proxy_url: URL of the IntelliProxy (default: http://localhost:9998)
            timeout: Request timeout in seconds (default: 600)
        """
        self.proxy_url = proxy_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout
    
    # =========================================================================
    # Standard Ollama API Methods
    # =========================================================================
    
    def list_models(self) -> Dict:
        """
        List all available models (including IntelliProxyLLM).
        Equivalent to: curl http://localhost:9998/api/tags
        
        Returns:
            Dict with 'models' key containing list of models
        """
        response = self.session.get(
            f"{self.proxy_url}/api/tags",
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    @overload
    def generate(
        self,
        prompt: str,
        model: str = "IntelliProxyLLM",
        *,
        stream: Literal[False] = False,
        **options
    ) -> Dict:
        ...
    
    @overload
    def generate(
        self,
        prompt: str,
        model: str = "IntelliProxyLLM",
        *,
        stream: Literal[True],
        **options
    ) -> Iterator[str]:
        ...
    
    def generate(
        self,
        prompt: str,
        model: str = "IntelliProxyLLM",
        stream: bool = False,
        **options
    ) -> Union[Dict, Iterator[str]]:
        """
        Generate text completion.
        Equivalent to: curl -X POST http://localhost:9998/api/generate
        
        Args:
            prompt: The prompt to generate completion for
            model: Model to use (default: IntelliProxyLLM for intelligent routing)
            stream: Whether to stream the response
            **options: Additional Ollama options (temperature, top_p, etc.)
        
        Returns:
            Dict response if stream=False, Iterator if stream=True
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **options
        }
        
        if stream:
            return self._stream_request("/api/generate", payload)
        
        response = self.session.post(
            f"{self.proxy_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "IntelliProxyLLM",
        stream: bool = False,
        **options
    ) -> Union[Dict, Iterator[str]]:
        """
        Generate chat completion.
        Equivalent to: curl -X POST http://localhost:9998/api/chat
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model to use (default: IntelliProxyLLM for intelligent routing)
            stream: Whether to stream the response
            **options: Additional Ollama options (temperature, top_p, etc.)
        
        Returns:
            Dict response or streaming iterator if stream=True
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **options
        }
        
        if stream:
            return self._stream_request("/api/chat", payload)
        
        response = self.session.post(
            f"{self.proxy_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def embeddings(self, prompt: str, model: str = "nomic-embed-text") -> Dict:
        """
        Generate embeddings for a prompt.
        
        Args:
            prompt: The prompt to generate embeddings for
            model: Model to use for embeddings
        
        Returns:
            Dict with 'embedding' key
        """
        response = self.session.post(
            f"{self.proxy_url}/api/embeddings",
            json={"prompt": prompt, "model": model},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _stream_request(self, endpoint: str, payload: Dict) -> Iterator[str]:
        """Handle streaming requests."""
        response = self.session.post(
            f"{self.proxy_url}{endpoint}",
            json=payload,
            stream=True,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded == "data: [DONE]":
                    break
                if decoded.startswith("data: "):
                    yield decoded[6:]  # Remove "data: " prefix
    
    def pull(self, model: str, stream: bool = False) -> Union[Dict, Iterator[str]]:
        """Pull a model from Ollama."""
        payload = {"name": model, "stream": stream}
        
        if stream:
            return self._stream_request("/api/pull", payload)
        
        response = self.session.post(
            f"{self.proxy_url}/api/pull",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def delete(self, model: str) -> Dict:
        """Delete a model."""
        response = self.session.delete(
            f"{self.proxy_url}/api/delete",
            json={"name": model},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def ask(self, prompt: str, model: str = "IntelliProxyLLM") -> str:
        """
        Simple Q&A interface - returns just the text response.
        
        Args:
            prompt: Question to ask
            model: Model to use
        
        Returns:
            Generated text response
        """
        result = self.generate(prompt, model=model, stream=False)  # type: ignore
        return result.get("response", "")
    
    def chat_ask(
        self,
        message: str,
        model: str = "IntelliProxyLLM",
        history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[str, List[Dict[str, str]]]:
        """
        Simple chat interface - returns response and updated history.
        
        Args:
            message: User message
            model: Model to use
            history: Previous message history
        
        Returns:
            Tuple of (response_text, updated_history)
        """
        if history is None:
            history = []
        
        history.append({"role": "user", "content": message})
        
        result: Dict = self.chat(history, model=model, stream=False)  # type: ignore
        response = result.get("message", {}).get("content", "")
        
        history.append({"role": "assistant", "content": response})
        
        return response, history
    
    def print_models(self):
        """Pretty print available models."""
        data = self.list_models()
        
        print("\n" + "="*70)
        print("📦 AVAILABLE MODELS")
        print("="*70)
        
        for model in data.get("models", []):
            name = model.get("name", "unknown")
            size_gb = model.get("size", 0) / (1024**3)
            desc = model.get("description", "")
            
            if name == "IntelliProxyLLM":
                print(f"\n🤖 {name}")
                print(f"   → {desc}")
            else:
                print(f"\n📄 {name} ({size_gb:.1f} GB)")
                if model.get("speed"):
                    print(f"   → Speed: {model['speed']}/10, Complexity: {model.get('complexity', '?')}/10")
        
        print("\n" + "="*70 + "\n")


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example: Using the Ollama-compatible client."""
    
    print("\n🚀 Ollama Compatible Client for IntelliProxy\n")
    
    # Create client (connects to proxy on port 9998)
    client = OllamaClient()
    
    # 1. List models
    print("1️⃣  Listing models...")
    client.print_models()
    
    # 2. Simple text generation with IntelliProxyLLM
    print("2️⃣  Text generation with IntelliProxyLLM (intelligent routing)...\n")
    response = client.ask("What is Python?")
    print(f"Response: {response}\n")
    
    # 3. Chat with IntelliProxyLLM
    print("3️⃣  Chat with IntelliProxyLLM...\n")
    history = []
    response, history = client.chat_ask("Hello! How are you?", history=history)
    print(f"User: Hello! How are you?")
    print(f"Assistant: {response}\n")
    
    response, history = client.chat_ask("What is 2+2?", history=history)
    print(f"User: What is 2+2?")
    print(f"Assistant: {response}\n")
    
    # 4. Using specific model directly
    print("4️⃣  Using specific model directly...\n")
    result = client.generate("Explain AI in one sentence", model="qwen2.5:7b")
    print(f"Direct model (qwen2.5:7b): {result.get('response')}\n")
    
    # 5. Streaming response
    print("5️⃣  Streaming response...\n")
    print("Streaming: ", end="", flush=True)
    for chunk in client.generate("Count to 5", stream=True):
        data = json.loads(chunk)
        print(data.get("response", ""), end="", flush=True)
    print("\n")


def quick_ask(prompt: str):
    """Quick CLI usage: python router_client.py "Your question" """
    client = OllamaClient()
    response = client.ask(prompt)
    print(response)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick mode: python router_client.py "Your prompt here"
        prompt = " ".join(sys.argv[1:])
        quick_ask(prompt)
    else:
        # Demo mode
        example_usage()
