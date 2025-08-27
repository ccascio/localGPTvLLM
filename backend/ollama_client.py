import requests
import json
import os
from typing import List, Dict, Optional

class VLLMClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        if base_url is None:
            base_url = os.getenv("VLLM_HOST", "http://localhost:8000")
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def is_server_running(self) -> bool:
        """Check if vLLM server is running"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", headers=self.headers, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
            if response.status_code == 200:
                models_data = response.json().get("data", [])
                return [model["id"] for model in models_data]
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Models are pre-loaded in vLLM, so this is a no-op"""
        available_models = self.list_models()
        if model_name in available_models:
            print(f"Model {model_name} is available")
            return True
        else:
            print(f"Model {model_name} not found in available models: {available_models}")
            return False
    
    def chat(self, message: str, model: str = "llama3.2", conversation_history: List[Dict] = None, enable_thinking: bool = True) -> str:
        """Send a chat message to vLLM server"""
        if conversation_history is None:
            conversation_history = []
        
        # Add user message to conversation
        messages = conversation_history + [{"role": "user", "content": message}]
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                # Additional cleanup: remove any thinking tokens that might slip through
                if not enable_thinking:
                    import re
                    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
                    response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
                    response_text = response_text.strip()
                
                return response_text
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Connection error: {e}"
    
    def chat_stream(self, message: str, model: str = "llama3.2", conversation_history: List[Dict] = None, enable_thinking: bool = True):
        """Stream chat response from vLLM server"""
        if conversation_history is None:
            conversation_history = []
        
        messages = conversation_history + [{"role": "user", "content": message}]
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=180
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            line_text = line_text[6:]  # Remove 'data: ' prefix
                        
                        if line_text.strip() == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(line_text)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    # Filter out thinking tokens in streaming mode
                                    if not enable_thinking:
                                        if '<think>' in content.lower() or '<thinking>' in content.lower():
                                            continue
                                    
                                    yield content
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            yield f"Connection error: {e}"

def main():
    """Test the vLLM client"""
    client = VLLMClient()
    
    # Check if vLLM server is running
    if not client.is_server_running():
        print("❌ vLLM server is not running. Please start vLLM server first.")
        print("Example: python -m vllm.entrypoints.openai.api_server --model your-model")
        return
    
    print("✅ vLLM server is running!")
    
    # List available models
    models = client.list_models()
    print(f"Available models: {models}")
    
    if not models:
        print("❌ No models available")
        return
    
    # Use first available model
    model_name = models[0]
    print(f"Using model: {model_name}")
    
    # Test chat
    print("\n🤖 Testing chat...")
    response = client.chat("Hello! Can you tell me a short joke?", model_name)
    print(f"AI: {response}")

# For backward compatibility, create an alias
OllamaClient = VLLMClient

if __name__ == "__main__":
    main()    