import requests
import json
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image
import httpx, asyncio
import os

class VLLMClient:
    """
    An enhanced client for vLLM that handles both text and image data for VLM models.
    """
    def __init__(self, host: str = None, api_key: Optional[str] = None):
        if host is None:
            host = os.getenv("VLLM_HOST", "http://localhost:8000")
        self.host = host.rstrip('/')
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _image_to_base64(self, image: Image.Image) -> str:
        """Converts a Pillow Image to a base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_embedding(self, model: str, text: str) -> List[float]:
        try:
            response = requests.post(
                f"{self.host}/v1/embeddings",
                headers=self.headers,
                json={"model": model, "input": text}
            )
            response.raise_for_status()
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error generating embedding: {e}")
            return []

    def generate_completion(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
    ) -> Dict[str, Any]:
        """
        Generates a completion, now with optional support for images.

        Args:
            model: The name of the generation model.
            prompt: The text prompt for the model.
            format: The format for the response, e.g., "json".
            images: A list of Pillow Image objects to send to the VLM.
            enable_thinking: Optional flag to disable chain-of-thought (ignored for vLLM).
        """
        try:
            # For vLLM, we use the completions endpoint for non-chat format
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "max_tokens": 1024,
                "temperature": 0.7
            }
            
            # Note: vLLM with vision models typically uses chat format for images
            # For now, we'll handle images through chat completions if needed
            if images:
                # Convert to chat format for vision models
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                for img in images:
                    img_b64 = self._image_to_base64(img)
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
                
                chat_payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": 1024,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    f"{self.host}/v1/chat/completions",
                    headers=self.headers,
                    json=chat_payload
                )
                response.raise_for_status()
                result = response.json()
                return {
                    "response": result["choices"][0]["message"]["content"],
                    "done": True
                }
            else:
                response = requests.post(
                    f"{self.host}/v1/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return {
                    "response": result["choices"][0]["text"],
                    "done": True
                }

        except requests.exceptions.RequestException as e:
            print(f"Error generating completion: {e}")
            return {}

    # -------------------------------------------------------------
    # Async variant – uses httpx so the caller can await multiple
    # LLM calls concurrently (triage, verification, etc.).
    # -------------------------------------------------------------
    async def generate_completion_async(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
        timeout: int = 180,
    ) -> Dict[str, Any]:
        """Asynchronous version of generate_completion using httpx."""

        payload = {
            "model": model, 
            "prompt": prompt, 
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if images:
                    # Use chat completions for images
                    messages = [
                        {
                            "role": "user", 
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                    for img in images:
                        img_b64 = self._image_to_base64(img)
                        messages[0]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })
                    
                    chat_payload = {
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "max_tokens": 1024,
                        "temperature": 0.7
                    }
                    
                    resp = await client.post(
                        f"{self.host}/v1/chat/completions", 
                        headers=self.headers,
                        json=chat_payload
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    return {
                        "response": result["choices"][0]["message"]["content"],
                        "done": True
                    }
                else:
                    resp = await client.post(
                        f"{self.host}/v1/completions", 
                        headers=self.headers,
                        json=payload
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    return {
                        "response": result["choices"][0]["text"],
                        "done": True
                    }
        except (httpx.HTTPError, asyncio.CancelledError) as e:
            print(f"Async vLLM completion error: {e}")
            return {}

    # -------------------------------------------------------------
    # Streaming variant – yields token chunks in real time
    # -------------------------------------------------------------
    def stream_completion(
        self,
        model: str,
        prompt: str,
        *,
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
    ):
        """Generator that yields partial *response* strings as they arrive.

        Example:

            for tok in client.stream_completion("model", "Hello"):
                print(tok, end="", flush=True)
        """
        payload: Dict[str, Any] = {
            "model": model, 
            "prompt": prompt, 
            "stream": True,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        try:
            if images:
                # Use chat completions for images
                messages = [
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
                for img in images:
                    img_b64 = self._image_to_base64(img)
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
                
                chat_payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 1024,
                    "temperature": 0.7
                }
                
                with requests.post(f"{self.host}/v1/chat/completions", headers=self.headers, json=chat_payload, stream=True) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line_text = raw_line.decode('utf-8')
                        if line_text.startswith('data: '):
                            line_text = line_text[6:]
                        if line_text.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(line_text)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                chunk = delta.get("content", "")
                                if chunk:
                                    yield chunk
                        except json.JSONDecodeError:
                            continue
            else:
                with requests.post(f"{self.host}/v1/completions", headers=self.headers, json=payload, stream=True) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line_text = raw_line.decode('utf-8')
                        if line_text.startswith('data: '):
                            line_text = line_text[6:]
                        if line_text.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(line_text)
                            if "choices" in data and len(data["choices"]) > 0:
                                chunk = data["choices"][0].get("text", "")
                                if chunk:
                                    yield chunk
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.RequestException as e:
            yield f"Connection error: {e}"

# For backward compatibility, create an alias
OllamaClient = VLLMClient

if __name__ == '__main__':
    print("vLLM client with multimodal (VLM) support.")
    try:
        client = VLLMClient()
        
        # Test basic completion
        print("\n--- Testing basic completion ---")
        response = client.generate_completion(
            model="your-model-name",
            prompt="Hello, how are you?"
        )
        
        if response and 'response' in response:
            print("Response:", response['response'])
        else:
            print("Failed to get response. Is vLLM server running?")
            
        # Test with image (if you have a vision model)
        print("\n--- Testing VLM completion ---")
        dummy_image = Image.new('RGB', (100, 100), 'black')
        
        vlm_response = client.generate_completion(
            model="your-vision-model-name",
            prompt="What color is this image?",
            images=[dummy_image]
        )
        
        if vlm_response and 'response' in vlm_response:
            print("VLM Response:", vlm_response['response'])
        else:
            print("VLM test skipped or failed")

    except Exception as e:
        print(f"An error occurred: {e}")