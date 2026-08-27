"""Ollama client — local LLM inference and embedding."""
from __future__ import annotations

import json
import httpx

from brain.utils.logger import get_logger

log = get_logger(__name__)


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", api_key: str = ""):
        self.host = host.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(timeout=60, headers=headers)

    async def generate(
        self,
        prompt: str,
        model: str = "phi3:mini",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_thread": 2,  # Limit threads on Pi
            },
        }
        try:
            resp = await self._http.post(f"{self.host}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            log.error(f"Ollama generate error ({model}): {e}")
            return ""

    async def chat(
        self,
        messages: list[dict],
        model: str = "phi3:mini",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        images: list[str] | None = None,
    ) -> str:
        # Attach image to last user message if provided (Ollama multimodal format)
        msgs = list(messages)
        if images:
            # Find or create the last user message and attach images
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    msgs[i] = dict(msgs[i])
                    msgs[i]["images"] = images
                    break
            else:
                msgs.append({"role": "user", "content": "", "images": images})

        payload = {
            "model": model,
            "messages": msgs,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_thread": 2,
            },
        }
        if system:
            payload["system"] = system
        try:
            resp = await self._http.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            # Local Ollama format: {"message": {"content": "..."}}
            content = data.get("message", {}).get("content", "")
            # OpenAI-compatible format used by Ollama Cloud: {"choices": [{"message": {"content": "..."}}]}
            if not content:
                choices = data.get("choices") or []
                content = choices[0].get("message", {}).get("content", "") if choices else ""
            if not content:
                log.debug("Ollama chat empty response body: %s", str(data)[:200])
            return content
        except Exception as e:
            log.error(f"Ollama chat error ({model}): {e}")
            return ""

    async def chat_stream(
        self,
        messages: list[dict],
        model: str = "phi3:mini",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        images: list[str] | None = None,
    ):
        """Async generator yielding (content_chunk, thinking_chunk) tuples as they stream."""
        msgs = list(messages)
        if images:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    msgs[i] = dict(msgs[i])
                    msgs[i]["images"] = images
                    break

        payload = {
            "model": model,
            "messages": msgs,
            "stream": True,
            "think": True,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_thread": 2},
        }
        if system:
            payload["system"] = system

        try:
            async with self._http.stream("POST", f"{self.host}/api/chat", json=payload, timeout=120) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    thinking = msg.get("thinking", "")
                    if content or thinking:
                        yield content, thinking
                    if data.get("done", False):
                        break
        except Exception as e:
            log.error(f"Ollama stream error ({model}): {e}")

    async def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        try:
            resp = await self._http.post(
                f"{self.host}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            log.error(f"Ollama embed error: {e}")
            return []

    async def is_available(self) -> bool:
        try:
            resp = await self._http.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._http.aclose()
