"""
Google AI Studio client — Gemini + Gemma models via REST API.
No SDK needed — pure httpx calls to generativelanguage.googleapis.com.

Rate limits (free tier):
  Gemma 3 27B/12B/4B/1B:  30 RPM, 14,400/day  ← main workhorse
  Gemini 3.1 Flash-Lite:  15 RPM,    500/day   ← medium complexity
  Gemini 2.5 Flash:        5 RPM,     20/day   ← complex only, use sparingly
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import httpx

from brain.utils.logger import get_logger

log = get_logger(__name__)

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Model registry: id → (rpm_limit, daily_limit, tier)
_MODELS: dict[str, tuple[int, int, str]] = {
    "gemma-3-27b-it":                    (30, 14400, "standard"),
    "gemma-3-12b-it":                    (30, 14400, "standard"),
    "gemma-3-4b-it":                     (30, 14400, "fast"),
    "gemma-3-1b-it":                     (30, 14400, "fast"),
    "gemini-2.0-flash":                  (15,  1500, "complex"),
    "gemini-1.5-flash":                  (15,  1500, "complex"),
    "gemini-robotics-er-1.6-preview":    (10,   500, "complex"),  # ER robotics preview
}

# Routing tiers: pick based on estimated token complexity
_TIER_ORDER = ["fast", "standard", "complex"]


@dataclass
class _ModelState:
    rpm_limit: int
    daily_limit: int
    tier: str
    daily_count: int = 0
    daily_reset: float = field(default_factory=lambda: time.time() + 86400)
    # Sliding window of request timestamps for RPM tracking
    rpm_window: deque = field(default_factory=lambda: deque(maxlen=30))

    def can_use(self) -> bool:
        now = time.time()
        if now > self.daily_reset:
            self.daily_count = 0
            self.daily_reset = now + 86400
        if self.daily_count >= self.daily_limit:
            return False
        # Clean old RPM timestamps
        while self.rpm_window and now - self.rpm_window[0] > 60:
            self.rpm_window.popleft()
        return len(self.rpm_window) < self.rpm_limit

    def record_use(self) -> None:
        self.daily_count += 1
        self.rpm_window.append(time.time())


class GoogleAIClient:
    def __init__(self, api_key: str, default_model: str = "gemma-3-27b-it"):
        self._key = api_key
        self._default_model = default_model
        self._states: dict[str, _ModelState] = {
            mid: _ModelState(rpm, daily, tier)
            for mid, (rpm, daily, tier) in _MODELS.items()
        }
        self._http = httpx.AsyncClient(timeout=30)

    async def is_available(self) -> bool:
        return bool(self._key)

    def _pick_model(self, complexity: str = "standard") -> str | None:
        """Pick the best available model for the given complexity tier."""
        # Preferred order for each tier — always prefer 27B for conversation quality
        tier_prefs: dict[str, list[str]] = {
            "fast":    ["gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it", "gemma-3-1b-it"],
            "standard": ["gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it"],
            "complex": ["gemini-2.0-flash", "gemini-1.5-flash", "gemma-3-27b-it", "gemma-3-12b-it"],
        }
        candidates = tier_prefs.get(complexity, tier_prefs["standard"])
        for mid in candidates:
            state = self._states.get(mid)
            if state and state.can_use():
                return mid
        # Last resort: any available model
        for mid, state in self._states.items():
            if state.can_use():
                return mid
        return None

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        context_messages: list[dict] | None = None,
        max_tokens: int = 512,
        complexity: str = "standard",
        model_override: str | None = None,
        frame_b64: str = "",
        frames: list[str] | None = None,           # legacy multi-frame
        extra_frames: list[str] | None = None,     # Visual RAG: additional historical frames
        tools: list[dict] | None = None,           # MCP: function-calling schemas
    ) -> str:
        if model_override:
            # Register unknown override models with conservative limits so rate tracking works
            if model_override not in self._states:
                self._states[model_override] = _ModelState(rpm_limit=10, daily_limit=500, tier="complex")
            model = model_override if self._states[model_override].can_use() else None
        else:
            model = self._pick_model(complexity)
        if not model:
            log.warning("GoogleAI: all models rate-limited")
            return ""

        # Build contents array (Gemini format)
        contents = []
        is_gemma = model.startswith("gemma")

        # Cap context for Gemma — small models echo history when given too many turns
        ctx = context_messages or []
        if is_gemma and len(ctx) > 8:
            ctx = ctx[-8:]

        # Add conversation history (text-only — images are only in the current turn)
        for msg in ctx:
            role = msg.get("role", "user")
            if role == "system":
                continue  # skip compaction summary entries — handled via system_prompt
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": msg.get("content", "")}],
            })

        # Gemma models don't support systemInstruction — prepend to first user message instead
        if system_prompt and is_gemma:
            if contents:
                first = contents[0]
                if first.get("role") == "user":
                    orig = first["parts"][0]["text"]
                    first["parts"][0]["text"] = f"[System context: {system_prompt}]\n\n{orig}"
                else:
                    contents.insert(0, {"role": "user", "parts": [{"text": f"[System context: {system_prompt}]"}]})
                    contents.insert(1, {"role": "model", "parts": [{"text": "Understood."}]})
            else:
                prompt = f"[System context: {system_prompt}]\n\n{prompt}"

        # Build current user turn — images come BEFORE the text prompt.
        # Visual RAG: if multiple frames are provided (live + historical snapshots),
        # inject all of them so Gemini can compare past vs present visually.
        # Gemini Flash/Pro support up to 16 images per request.
        current_parts: list[dict] = []
        # Merge all image sources: explicit frames list > extra_frames > single frame_b64
        if frames:
            image_sources = frames
        else:
            image_sources = ([frame_b64] if frame_b64 else []) + (extra_frames or [])
        # Cap at 3 images to stay within reasonable token budgets (live + up to 2 historical)
        for img_b64 in image_sources[:3]:
            if img_b64:
                current_parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64,
                    }
                })
        if len(image_sources) > 1:
            log.debug("GoogleAI: grounded generation with %d images (live + historical)", min(3, len(image_sources)))
        current_parts.append({"text": prompt})
        contents.append({"role": "user", "parts": current_parts})

        body: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
            },
        }

        # systemInstruction only works for Gemini models (not Gemma)
        if system_prompt and not is_gemma:
            body["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        # MCP Tool Injection — inject hardware capabilities as Gemini functionDeclarations.
        # Only Gemini models support function calling (not Gemma).
        if tools and not is_gemma:
            body["tools"] = [{
                "functionDeclarations": [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    }
                    for t in tools
                ]
            }]

        url = f"{_BASE}/{model}:generateContent"
        try:
            resp = await self._http.post(
                url,
                headers={"x-goog-api-key": self._key, "Content-Type": "application/json"},
                json=body,
            )

            if resp.status_code == 429:
                log.warning(f"GoogleAI: rate limited on {model}")
                self._states[model].daily_count = self._states[model].daily_limit  # mark exhausted
                return ""

            if resp.status_code == 400:
                err_msg = resp.json().get("error", {}).get("message", "")
                if "Developer instruction is not enabled" in err_msg:
                    # This model doesn't support systemInstruction — shouldn't happen
                    # now that we guard by is_gemma, but skip it if it does
                    log.warning(f"GoogleAI: {model} doesn't support systemInstruction, skipping")
                    self._states[model].daily_count = self._states[model].daily_limit
                    return ""
                log.warning(f"GoogleAI: {model} returned 400: {resp.text[:200]}")
                return ""

            if resp.status_code != 200:
                log.warning(f"GoogleAI: {model} returned {resp.status_code}: {resp.text[:200]}")
                return ""

            data = resp.json()
            self._states[model].record_use()

            # Extract text from response — also handle functionCall parts (MCP tool calls)
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            text_parts = []
            tool_calls = []
            for p in parts:
                if "text" in p:
                    text_parts.append(p["text"])
                elif "functionCall" in p:
                    tool_calls.append(p["functionCall"])
            text = "".join(text_parts).strip()
            # Append tool calls as a parseable JSON suffix so LLMRouter can detect them
            if tool_calls:
                import json as _json
                text = (text + "\n[TOOL_CALLS]" + _json.dumps(tool_calls)).strip()
            log.info(f"GoogleAI [{model}]: {len(text)} chars tool_calls={len(tool_calls)}")
            return text

        except Exception as e:
            log.error(f"GoogleAI request error: {e}")
            return ""

    async def close(self) -> None:
        await self._http.aclose()
