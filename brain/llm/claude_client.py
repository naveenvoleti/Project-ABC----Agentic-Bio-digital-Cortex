"""Claude API client — cloud LLM with prompt caching for context efficiency."""
from __future__ import annotations

import os

from brain.utils.logger import get_logger

log = get_logger(__name__)


class ClaudeClient:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except Exception as e:
                log.error(f"Anthropic SDK not available: {e}")
        return self._client

    async def chat(
        self,
        messages: list[dict],
        system: str = "",
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        client = self._get_client()
        if not client:
            return ""
        if not self.api_key:
            log.warning("No Anthropic API key configured")
            return ""

        # Use prompt caching on the system prompt (saves tokens on repeat calls)
        system_blocks = []
        if system:
            system_blocks = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        try:
            # Build kwargs without system key when no system blocks exist,
            # so we never reference the anthropic module in this scope.
            create_kwargs: dict = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
            }
            if system_blocks:
                create_kwargs["system"] = system_blocks
            resp = await client.messages.create(**create_kwargs)
            return resp.content[0].text if resp.content else ""
        except Exception as e:
            log.error(f"Claude API error: {e}")
            return ""

    async def is_available(self) -> bool:
        return bool(self.api_key) and self.api_key.startswith("sk-ant-")
