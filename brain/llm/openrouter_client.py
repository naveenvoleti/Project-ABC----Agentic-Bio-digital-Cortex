"""OpenRouter client — free cloud LLM via OpenAI-compatible API.

Strategy:
- On startup, calls GET /models to discover all currently live free models.
- Tracks per-model cooldowns: a 429 puts a model on ice for COOLDOWN_SECS (90s).
- Remembers the last working model and tries it first (sticky).
- On each request, skips models still in cooldown and walks the rest of the chain.

Get a free API key at https://openrouter.ai (no credit card needed)
"""
from __future__ import annotations

import asyncio
import os
import time

import httpx

from brain.utils.logger import get_logger

log = get_logger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_MODEL_CACHE_TTL = 3600   # refresh model list every hour
COOLDOWN_SECS = 90        # seconds to back off after a 429

# Bootstrap used only when /models fetch fails
_BOOTSTRAP_MODELS: list[str] = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "deepseek/deepseek-chat-v3-0324:free",
]

_PREFERRED_KEYWORDS = ["llama-3", "qwen", "gemma", "deepseek", "phi", "mistral"]


def _sort_free_models(ids: list[str]) -> list[str]:
    def _rank(mid: str) -> int:
        for i, kw in enumerate(_PREFERRED_KEYWORDS):
            if kw in mid.lower():
                return i
        return len(_PREFERRED_KEYWORDS)
    return sorted(ids, key=_rank)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str = "",
        default_model: str = "",
        site_url: str = "http://localhost:8080",
        site_name: str = "Project-ABC Brain",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._preferred_model = default_model
        self._http = httpx.AsyncClient(
            base_url=OPENROUTER_BASE,
            timeout=60,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": site_url,
                "X-Title": site_name,
                "Content-Type": "application/json",
            },
        )
        self._live_models: list[str] = []
        self._models_fetched_at: float = 0.0
        self._last_working_model: str = ""
        # model_id → monotonic time when cooldown expires
        self._cooldowns: dict[str, float] = {}

    @property
    def default_model(self) -> str:
        if self._preferred_model:
            return self._preferred_model
        return self._live_models[0] if self._live_models else _BOOTSTRAP_MODELS[0]

    def _refresh_auth(self) -> None:
        self._http.headers["Authorization"] = f"Bearer {self.api_key}"

    def _is_cooling(self, model: str) -> bool:
        expires = self._cooldowns.get(model, 0.0)
        return time.monotonic() < expires

    def _set_cooldown(self, model: str) -> None:
        expires_at = time.monotonic() + COOLDOWN_SECS
        self._cooldowns[model] = expires_at
        remaining = COOLDOWN_SECS
        log.warning(f"OpenRouter: [{model}] on cooldown for {remaining}s")

    def _clear_cooldown(self, model: str) -> None:
        self._cooldowns.pop(model, None)

    async def _ensure_models(self) -> list[str]:
        now = time.monotonic()
        if self._live_models and (now - self._models_fetched_at) < _MODEL_CACHE_TTL:
            return self._live_models
        try:
            self._refresh_auth()
            resp = await self._http.get("/models", timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            free_ids = [
                m["id"] for m in data
                if (
                    str(m.get("pricing", {}).get("prompt", "1")) == "0"
                    and str(m.get("pricing", {}).get("completion", "1")) == "0"
                )
            ]
            if free_ids:
                sorted_ids = _sort_free_models(free_ids)
                if self._preferred_model and self._preferred_model in sorted_ids:
                    sorted_ids.remove(self._preferred_model)
                    sorted_ids.insert(0, self._preferred_model)
                self._live_models = sorted_ids
                self._models_fetched_at = now
                log.info(
                    f"OpenRouter: {len(sorted_ids)} free models available. "
                    f"Top 3: {sorted_ids[:3]}"
                )
            else:
                log.warning("OpenRouter: no free models in live list, using bootstrap")
                self._live_models = _BOOTSTRAP_MODELS
        except Exception as e:
            log.warning(f"OpenRouter: model list fetch failed ({e}), using bootstrap")
            if not self._live_models:
                self._live_models = _BOOTSTRAP_MODELS
        return self._live_models

    async def _try_model(
        self,
        model: str,
        full_messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, bool]:
        """Returns (response_text, try_next). try_next=True means skip to next model."""
        payload = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = await self._http.post("/chat/completions", json=payload)
            if resp.status_code == 429:
                self._set_cooldown(model)
                return "", True
            if resp.status_code in (404, 400):
                log.warning(f"OpenRouter: [{model}] {resp.status_code} — skipping")
                return "", True
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            self._clear_cooldown(model)   # successful — remove any old cooldown
            return content, False
        except httpx.HTTPStatusError as e:
            log.error(f"OpenRouter [{model}]: {e.response.status_code} {e.response.text[:120]}")
            return "", True
        except Exception as e:
            # Network/timeout/JSON error on this model — try next candidate
            log.warning(f"OpenRouter [{model}]: {type(e).__name__}: {e}")
            return "", True

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if not self.api_key:
            log.warning("OpenRouter: no API key")
            return ""

        self._refresh_auth()

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # Specific model override
        if model:
            response, _ = await self._try_model(model, full_messages, max_tokens, temperature)
            return response

        # Build candidate list: sticky model first, then rest — but SKIP cooled-down models
        all_models = await self._ensure_models()

        # Put last working model at front
        if self._last_working_model and self._last_working_model in all_models:
            ordered = [self._last_working_model] + [
                m for m in all_models if m != self._last_working_model
            ]
        else:
            ordered = all_models

        # Filter out models still in cooldown
        available = [m for m in ordered if not self._is_cooling(m)]
        cooling   = [m for m in ordered if self._is_cooling(m)]

        if not available:
            soonest = min(self._cooldowns.get(m, 0) for m in cooling) - time.monotonic()
            log.warning(
                f"OpenRouter: all {len(cooling)} models on cooldown. "
                f"Soonest available in {soonest:.0f}s"
            )
            return ""

        skipped = len(ordered) - len(available)
        if skipped:
            log.debug(f"OpenRouter: skipping {skipped} cooled-down models, {len(available)} available")

        for i, candidate in enumerate(available[:8]):
            if i > 0:
                await asyncio.sleep(0.3)
            response, try_next = await self._try_model(candidate, full_messages, max_tokens, temperature)
            if response:
                if candidate != self._last_working_model:
                    log.info(f"OpenRouter: now using [{candidate}]")
                    self._last_working_model = candidate
                return response
            if not try_next:
                break

        log.error("OpenRouter: all available models failed")
        return ""

    async def is_available(self) -> bool:
        if not self.api_key:
            return False
        models = await self._ensure_models()
        return len(models) > 0

    async def list_free_models(self) -> list[str]:
        return await self._ensure_models()

    async def close(self) -> None:
        await self._http.aclose()
