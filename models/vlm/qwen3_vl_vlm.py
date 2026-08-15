import requests
from typing import List, Optional, Dict, Any

class Qwen3VLClient:
    """
    Client for Qwen3-VL model deployed via vLLM with OpenAI-compatible API.
    Supports pure text (call_llm) and multimodal vision-language (call_vlm) inference.
    """

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Args:
            base_url: vLLM server base URL, e.g., "http://localhost:8000"
            model: model name; if None, will query /v1/models to fetch the first available
            timeout: request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}

        if self.model is None:
            self.model = self._get_default_model()

    def _get_default_model(self) -> str:
        """Fetch the first model name from /v1/models endpoint."""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers,
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0]["id"]
            else:
                raise RuntimeError("No model found in /v1/models response.")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch model list: {e}")

    def _call_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Send chat completion request to vLLM. Returns the assistant's content."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **kwargs,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Chat completion failed: {e}")

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Pure text chat."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_chat_completion(messages, **kwargs)

    def call_vlm(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64_list: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Multimodal vision-language chat."""
        content_parts = [{"type": "text", "text": user_prompt}]
        if image_base64_list:
            for img_b64 in image_base64_list:
                if not img_b64.startswith("data:image/"):
                    img_b64 = f"data:image/jpeg;base64,{img_b64}"
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": img_b64}}
                )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]
        return self._call_chat_completion(messages, **kwargs)