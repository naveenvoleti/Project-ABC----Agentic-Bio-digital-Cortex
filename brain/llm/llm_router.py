"""
LLM Router — routes inference across local (Ollama) → Google AI → OpenRouter → Claude.

Routing tiers:
  1. Ollama phi3:mini           — simple queries   (< 150 tokens, offline)
  2. Ollama llama3.2:3b         — balanced queries  (< 800 tokens, offline)
  3. Google AI Gemma 3 27B      — primary cloud     (14,400/day, 30 RPM, free)
  4. Google AI Gemini 2.5 Flash — complex queries   (20/day, 5 RPM — used sparingly)
  5. OpenRouter free            — secondary cloud   (fallback)
  6. Claude haiku               — paid fallback     (requires ANTHROPIC_API_KEY)
  7. Ollama fallback            — when all cloud unavailable

Includes semantic response cache to avoid redundant inference on similar queries.
"""
from __future__ import annotations

import collections
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

from brain.utils.logger import get_logger
from .ollama_client import OllamaClient
from .claude_client import ClaudeClient
from .openrouter_client import OpenRouterClient
from .google_ai_client import GoogleAIClient
from .er_client import ERClient

log = get_logger(__name__)

SIMPLE_THRESHOLD = 150     # tokens
COMPLEX_THRESHOLD = 800    # tokens
CACHE_SIM_THRESHOLD = 0.97  # stricter — only near-identical queries hit cache
CACHE_TTL = 300            # 5 minutes — prevents stale responses in conversation


@dataclass
class _CacheEntry:
    query_hash: str
    response: str
    embedding: list[float]
    created_at: float = field(default_factory=time.time)
    hits: int = 0


def _token_estimate(text: str) -> int:
    return math.ceil(len(text.split()) * 1.3)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class LLMRouter:
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        simple_model: str = "phi3:mini",
        balanced_model: str = "llama3.2:3b",
        cloud_model: str = "claude-haiku-4-5-20251001",
        embed_model: str = "nomic-embed-text",
        anthropic_api_key: str = "",
        openrouter_api_key: str = "",
        openrouter_model: str = "meta-llama/llama-3.2-3b-instruct:free",
        google_ai_api_key: str = "",
        network_available: bool = True,
        ollama_enabled: bool = True,
        openrouter_enabled: bool = True,
        ollama_cloud_api_key: str = "",
        ollama_cloud_model: str = "llama3.3:70b-cloud",
        ollama_cloud_enabled: bool = False,
        er_enabled: bool = True,
        google_ai_enabled: bool = True,
    ):
        self.simple_model = simple_model
        self.balanced_model = balanced_model
        self.cloud_model = cloud_model
        self.embed_model = embed_model
        self.network_available = network_available

        self._ollama_enabled = ollama_enabled
        self._openrouter_enabled = openrouter_enabled
        self._er_enabled = er_enabled
        self._google_ai_enabled = google_ai_enabled
        self._ollama_cloud_enabled = ollama_cloud_enabled and bool(ollama_cloud_api_key)
        self._ollama_cloud_model = ollama_cloud_model
        self._ollama = OllamaClient(ollama_host)
        self._ollama_cloud = OllamaClient(
            host="https://ollama.com",
            api_key=ollama_cloud_api_key,
        )
        self._claude = ClaudeClient(anthropic_api_key)
        self._openrouter = OpenRouterClient(
            api_key=openrouter_api_key,
            default_model=openrouter_model,
        )
        self._google = GoogleAIClient(api_key=google_ai_api_key)
        self._er = ERClient(api_key=google_ai_api_key)
        # Use deque for O(1) append/popleft instead of O(n) list pop(0)
        self._cache: collections.deque[_CacheEntry] = collections.deque(maxlen=200)
        self._ollama_available: bool | None = None
        self._ollama_checked_at: float = 0.0           # TTL-based re-check
        self._openrouter_available: bool | None = None
        self._openrouter_checked_at: float = 0.0
        self._eco_mode: bool = False          # set by BehaviorAgent via cognition_agent
        _AVAILABILITY_TTL = 60.0              # re-check provider health every 60s

    def set_eco_mode(self, enabled: bool) -> None:
        """Switch to minimal compute profile in ECO mode.
        Prefers lightest local model; falls through to cloud if Ollama unavailable."""
        self._eco_mode = enabled
        if enabled:
            status = (f"ON — using {self.simple_model}" if self._ollama_enabled
                      else "ON — Ollama disabled, routing to cloud")
        else:
            status = "OFF — full routing restored"
        log.info("LLMRouter: ECO mode %s", status)

    async def _check_ollama(self) -> bool:
        if not self._ollama_enabled:
            return False
        now = time.time()
        if self._ollama_available is None or (now - self._ollama_checked_at) > 60.0:
            self._ollama_available = await self._ollama.is_available()
            self._ollama_checked_at = now
        return self._ollama_available

    async def _check_openrouter(self) -> bool:
        if not self._openrouter_enabled:
            return False
        now = time.time()
        if self._openrouter_available is None or (now - self._openrouter_checked_at) > 60.0:
            self._openrouter_available = await self._openrouter.is_available()
            self._openrouter_checked_at = now
        return self._openrouter_available

    async def embed(self, text: str) -> list[float]:
        # Primary: Ollama local model
        if await self._check_ollama():
            result = await self._ollama.embed(text, model=self.embed_model)
            if result:
                return result

        # Fallback: sentence-transformers (offline, ~90MB, no API key)
        try:
            import os
            import asyncio as _asyncio
            from sentence_transformers import SentenceTransformer
            loop = _asyncio.get_event_loop()

            if not hasattr(self, "_st_model"):
                # Suppress HF hub network checks — use cached model only if available
                def _load_model():
                    # Try fully offline first (no network, no rate limits)
                    try:
                        os.environ.setdefault("HF_HUB_OFFLINE", "1")
                        return SentenceTransformer(
                            "sentence-transformers/all-MiniLM-L6-v2",
                            local_files_only=True,
                        )
                    except Exception:
                        # Not cached yet — download once (clears offline flag temporarily)
                        os.environ.pop("HF_HUB_OFFLINE", None)
                        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                        os.environ["HF_HUB_OFFLINE"] = "1"
                        return model

                self._st_model = await loop.run_in_executor(None, _load_model)
                log.info("sentence-transformers model loaded (offline mode)")

            vec = await loop.run_in_executor(
                None, lambda: self._st_model.encode(text).tolist()
            )
            return vec
        except ImportError:
            log.debug("sentence-transformers not installed — embeddings unavailable")
        except Exception as e:
            log.debug(f"sentence-transformers embed error: {e}")
        return []

    async def _cache_lookup(self, query: str) -> str | None:
        embedding = await self.embed(query)
        now = time.time()
        # Expire stale entries in-place (deque supports iteration but not slice-assign)
        expired = [e for e in self._cache if now - e.created_at >= CACHE_TTL]
        for e in expired:
            try:
                self._cache.remove(e)
            except ValueError:
                pass
        for entry in self._cache:
            sim = _cosine_sim(embedding, entry.embedding)
            if sim >= CACHE_SIM_THRESHOLD:
                entry.hits += 1
                log.debug(f"Cache hit (sim={sim:.3f})")
                return entry.response
        return None

    async def _cache_store(self, query: str, response: str) -> None:
        embedding = await self.embed(query)
        h = hashlib.md5(query.encode()).hexdigest()
        # Deque with maxlen=200 automatically discards oldest entry when full
        self._cache.append(_CacheEntry(query_hash=h, response=response, embedding=embedding))

    async def infer(
        self,
        user_message: str,
        system_prompt: str = "",
        context_messages: list[dict] | None = None,
        max_tokens: int = 512,
        frame_b64: str = "",
        frames: list[str] | None = None,
        skip_cache: bool = False,
        tools: list[dict] | None = None,   # MCP: function-calling schemas (Gemini only)
    ) -> str:
        # Merge legacy frame_b64 + new multi-frame list.
        # Live frame (frame_b64) always goes first so the LLM sees the present
        # before the historical memories retrieved by Visual RAG.
        all_frames: list[str] = []
        if frame_b64:
            all_frames.append(frame_b64)
        if frames:
            all_frames.extend(f for f in frames if f and f != frame_b64)
        # Keep backward-compat: single-frame callers still work via frame_b64
        _primary_frame = all_frames[0] if all_frames else ""

        # Synthesis override — code generation cannot be truncated by fatigue mode.
        if (
            "synthesis" in system_prompt.lower()
            or "python expert" in system_prompt.lower()
            or "html" in system_prompt.lower()
            or "write a python script" in user_message.lower()
            or "code" in user_message.lower()
            or "tool" in user_message.lower()
            or "script" in user_message.lower()
        ):
            max_tokens = max(max_tokens, 2048)
            log.info("LLMRouter: synthesis request — overriding fatigue cap to %d tokens", max_tokens)
        else:
            # Hard minimum: never pass fewer than 32 tokens even under extreme fatigue.
            # Also log when affective feedback has significantly reduced the cap.
            max_tokens = max(32, max_tokens)
            if max_tokens <= 120:
                log.info("LLMRouter: affective fatigue mode — max_tokens=%d (body state constrained)", max_tokens)

        # Cache check — skipped for think-pass calls to prevent thought from being
        # returned as the main response (cache key is user_message only, not system_prompt)
        if not skip_cache:
            cached = await self._cache_lookup(user_message)
            if cached:
                return cached

        log.debug(
            "LLMRouter.infer | user=%r | system=%r | ctx_turns=%d | images=%d | max_tokens=%d",
            user_message[:120],
            system_prompt[:200] if system_prompt else "",
            len(context_messages or []),
            len(all_frames),
            max_tokens,
        )

        messages = list(context_messages or [])
        messages.append({"role": "user", "content": user_message})

        # ECO mode — prefer lightest local model, but fall through to cloud if Ollama is down.
        # Use chat() (not generate()) so conversation history is forwarded correctly.
        if self._eco_mode:
            if await self._check_ollama():
                response = await self._ollama.chat(
                    messages=messages,
                    model=self.simple_model,
                    system=system_prompt,
                    max_tokens=max_tokens,
                )
                if response:
                    if not skip_cache:
                        await self._cache_store(user_message, response)
                    return response
                log.warning("LLMRouter: ECO mode — Ollama returned empty, falling through to cloud")
            else:
                log.warning("LLMRouter: ECO mode — Ollama unavailable, falling through to cloud")

        # Route by user message complexity only — system_prompt is always large
        tokens = _token_estimate(user_message)
        ollama_ok = await self._check_ollama()

        response = ""

        google_ok = self._google_ai_enabled and self.network_available and await self._google.is_available()
        or_ok = self.network_available and await self._check_openrouter()

        # Determine complexity tier for Google AI model selection
        if tokens >= COMPLEX_THRESHOLD:
            complexity = "complex"
        elif tokens >= SIMPLE_THRESHOLD:
            complexity = "standard"
        else:
            complexity = "fast"

        # Ollama only supports a single image; pass primary frame only
        imgs = [_primary_frame] if _primary_frame else None

        if tokens < SIMPLE_THRESHOLD and ollama_ok:
            log.debug("Route: ollama/%s (%d tokens)%s", self.simple_model, tokens, " +image" if imgs else "")
            response = await self._ollama.chat(
                messages, model=self.simple_model, system=system_prompt, max_tokens=max_tokens, images=imgs
            )
        elif tokens < COMPLEX_THRESHOLD and ollama_ok:
            log.debug("Route: ollama/%s (%d tokens)%s", self.balanced_model, tokens, " +image" if imgs else "")
            response = await self._ollama.chat(
                messages, model=self.balanced_model, system=system_prompt, max_tokens=max_tokens, images=imgs
            )

        # Google AI — primary cloud tier; pass ALL frames for multi-image grounded reasoning
        google_override = (
            self.cloud_model
            if self.cloud_model and ("gemini" in self.cloud_model or "gemma" in self.cloud_model)
            else None
        )
        if not response and google_ok:
            is_er = self._er_enabled and self._er.is_available() and "robotics-er" in (self.cloud_model or "")
            if is_er:
                log.debug("Route: er-sdk/%s (%d tokens)%s", self.cloud_model, tokens,
                          f" +{len(all_frames)}img" if all_frames else "")
                response = await self._er.generate(
                    prompt=user_message,
                    system_prompt=system_prompt,
                    frame_b64=_primary_frame,
                    max_tokens=max_tokens,
                    thinking_budget=512,
                )
            else:
                log.debug("Route: google-ai/%s (%d tokens)%s", google_override or complexity, tokens,
                          f" +{len(all_frames)}img" if all_frames else "")
                # Pass all frames — GoogleAIClient accepts frame_b64 as the primary;
                # additional frames injected as extra_frames for multi-image reasoning
                response = await self._google.generate(
                    prompt=user_message,
                    system_prompt=system_prompt,
                    context_messages=context_messages,
                    max_tokens=max_tokens,
                    complexity=complexity,
                    model_override=google_override,
                    frame_b64=_primary_frame,
                    extra_frames=all_frames[1:] if len(all_frames) > 1 else [],
                    tools=tools,   # MCP: inject hardware tool schemas
                )

        # Ollama Cloud — paid cloud tier (same API as local, no GPU needed)
        if not response and self._ollama_cloud_enabled and self.network_available:
            log.debug("Route: ollama-cloud/%s (%d tokens)%s", self._ollama_cloud_model, tokens,
                      " +image" if imgs else "")
            response = await self._ollama_cloud.chat(
                messages, model=self._ollama_cloud_model, system=system_prompt,
                max_tokens=max_tokens, images=imgs,
            )

        # OpenRouter — secondary cloud fallback
        if not response and or_ok:
            log.debug(f"Route: openrouter fallback ({tokens} tokens)")
            response = await self._openrouter.chat(
                messages, system=system_prompt, max_tokens=max_tokens
            )

        # Claude — paid last resort
        if not response and self.network_available and await self._claude.is_available():
            log.debug(f"Route: claude/{self.cloud_model} ({tokens} tokens)")
            response = await self._claude.chat(
                messages, system=system_prompt, model=self.cloud_model, max_tokens=max_tokens
            )

        # Ollama offline fallback
        if not response and ollama_ok:
            log.debug(f"Route: ollama/{self.balanced_model} offline fallback")
            response = await self._ollama.chat(
                messages, model=self.balanced_model, system=system_prompt, max_tokens=max_tokens
            )

        if not response:
            log.warning(
                "LLMRouter: all providers exhausted — ollama=%s google=%s openrouter=%s network=%s",
                ollama_ok, google_ok, or_ok, self.network_available,
            )
            return ""

        if response:
            if not skip_cache:
                await self._cache_store(user_message, response)
        return response

    async def infer_stream(
        self,
        user_message: str,
        system_prompt: str = "",
        context_messages: list[dict] | None = None,
        max_tokens: int = 512,
        frame_b64: str = "",
    ):
        """Async generator yielding (content_chunk, thinking_chunk) tuples.

        Tries Ollama Cloud → Ollama local in order.
        Falls back to a single non-streaming infer() call if neither is available.
        """
        messages = list(context_messages or [])
        messages.append({"role": "user", "content": user_message})
        imgs = [frame_b64] if frame_b64 else None

        # Ollama Cloud — streaming with thinking support (qwen3, etc.)
        if self._ollama_cloud_enabled and self.network_available:
            try:
                got_any = False
                async for chunk in self._ollama_cloud.chat_stream(
                    messages, model=self._ollama_cloud_model,
                    system=system_prompt, max_tokens=max_tokens, images=imgs,
                ):
                    got_any = True
                    yield chunk
                if got_any:
                    return
            except Exception as e:
                log.warning("infer_stream: ollama-cloud failed: %s", e)

        # Local Ollama
        if await self._check_ollama():
            try:
                got_any = False
                async for chunk in self._ollama.chat_stream(
                    messages, model=self.balanced_model,
                    system=system_prompt, max_tokens=max_tokens, images=imgs,
                ):
                    got_any = True
                    yield chunk
                if got_any:
                    return
            except Exception as e:
                log.warning("infer_stream: ollama-local failed: %s", e)

        # Fallback: non-streaming, yield the whole response as one chunk
        response = await self.infer(
            user_message, system_prompt, context_messages, max_tokens, frame_b64,
            skip_cache=True,
        )
        if response:
            yield response, ""

    async def er_reason(
        self,
        task: str,
        context: str = "",
        frame_b64: str = "",
        max_tokens: int = 1024,
    ) -> dict:
        """V8.0 High-Level Brain — Gemini Robotics ER structured reasoning via official SDK.

        Returns dict with keys: plan (list[str|dict]), success_estimate (float), reasoning (str).
        Falls back to REST-based Gemini, then infer() if ER SDK unavailable.
        """
        # Primary: official google-genai SDK — supports ThinkingConfig and Part.from_bytes
        if self._er.is_available() and self.network_available:
            try:
                return await self._er.plan_task(
                    task=task,
                    context=context,
                    frame_b64=frame_b64,
                    thinking_budget=1024,
                )
            except Exception as e:
                log.warning("er_reason: ERClient.plan_task failed: %s", e)

        # Secondary: REST-based Gemini via GoogleAIClient
        system = (
            "You are the High-Level Brain of an embodied robot. "
            "Given a task and visual context, produce a step-by-step plan as a JSON object with keys: "
            "'plan' (list of action strings), 'success_estimate' (0.0–1.0 float), 'reasoning' (str). "
            "Be concise. Output valid JSON only."
        )
        user_msg = f"Task: {task}\nContext: {context}"

        google_ok = self.network_available and await self._google.is_available()
        if google_ok:
            try:
                raw = await self._google.generate(
                    prompt=user_msg,
                    system_prompt=system,
                    max_tokens=max_tokens,
                    complexity="complex",
                    model_override=self.cloud_model or "gemini-2.0-flash",
                    frame_b64=frame_b64,
                )
                if raw:
                    import json, re as _re
                    m = _re.search(r'\{.*\}', raw, _re.DOTALL)
                    if m:
                        return json.loads(m.group(0))
            except Exception as e:
                log.warning("er_reason: Gemini REST call failed: %s", e)

        # Tertiary: plain-text plan from infer()
        fallback = await self.infer(user_msg, system_prompt=system, max_tokens=max_tokens)
        steps = [l.strip("- ").strip() for l in fallback.splitlines() if l.strip()]
        return {"plan": steps, "success_estimate": 0.5, "reasoning": fallback}

    async def vla_act(
        self,
        command: str,
        frame_b64: str = "",
        dof_primitives: list[str] | None = None,
    ) -> dict:
        """V8.0 Low-Level Brain — VLA action token generation.

        Uses ER model's spatial point_objects() for object-targeting commands (thinking_budget=0,
        fastest). Falls back to REST Gemini, then regex parsing.
        Returns dict with keys: action_type (str), params (dict), confidence (float).
        """
        primitives = dof_primitives or [
            "move_pan(degrees)", "move_tilt(degrees)",
            "rotate_wheels(direction, degrees)", "speak(text)", "stop()",
        ]

        # ER SDK path: for spatial commands with a camera frame, use point_objects
        if self._er.is_available() and self.network_available and frame_b64:
            import re as _re
            target_match = _re.search(
                r'(?:look at|point at|track|find|locate|go to|face)\s+(.+)',
                command, _re.IGNORECASE
            )
            if target_match:
                target = target_match.group(1).strip()
                try:
                    points = await self._er.point_objects(frame_b64, [target], thinking_budget=0)
                    if points:
                        pt = points[0]
                        y_norm, x_norm = pt["point"]  # 0-1000 normalised
                        # Convert to pan/tilt degrees: centre=500 → 0°, range ±45°
                        pan = (x_norm - 500) / 1000 * 90
                        tilt = (500 - y_norm) / 1000 * 60
                        log.info("vla_act: ER point → pan=%.1f tilt=%.1f for '%s'", pan, tilt, target)
                        return {
                            "action_type": "move_pan_tilt",
                            "params": {"pan_degrees": round(pan, 1), "tilt_degrees": round(tilt, 1),
                                       "label": pt.get("label", target)},
                            "confidence": 0.85,
                        }
                except Exception as e:
                    log.warning("vla_act: ER point_objects failed: %s", e)

        system = (
            "You are the Low-Level Motor Cortex of an embodied robot. "
            "Given a motor command, output a JSON object with keys: "
            "'action_type' (str, one of the primitives), 'params' (dict), 'confidence' (0.0–1.0). "
            "Output valid JSON only. No explanation."
        )
        user_msg = f"Command: {command}\nAvailable primitives: {primitives}"

        google_ok = self.network_available and await self._google.is_available()
        if google_ok:
            try:
                raw = await self._google.generate(
                    prompt=user_msg,
                    system_prompt=system,
                    max_tokens=128,
                    complexity="fast",
                    model_override="gemini-2.0-flash",
                )
                if raw:
                    import json, re as _re
                    m = _re.search(r'\{.*\}', raw, _re.DOTALL)
                    if m:
                        return json.loads(m.group(0))
            except Exception as e:
                log.warning("vla_act: Gemini call failed: %s", e)

        # Fallback: regex parse common motor verbs
        import re as _re
        cmd_lower = command.lower()
        if _re.search(r'pan|left|right', cmd_lower):
            deg = float(_re.search(r'(-?\d+)', command).group(1)) if _re.search(r'-?\d+', command) else 30.0
            return {"action_type": "move_pan", "params": {"degrees": deg}, "confidence": 0.4}
        if _re.search(r'tilt|up|down', cmd_lower):
            deg = float(_re.search(r'(-?\d+)', command).group(1)) if _re.search(r'-?\d+', command) else 15.0
            return {"action_type": "move_tilt", "params": {"degrees": deg}, "confidence": 0.4}
        if _re.search(r'stop|halt', cmd_lower):
            return {"action_type": "stop", "params": {}, "confidence": 0.9}
        return {"action_type": "speak", "params": {"text": command}, "confidence": 0.3}

    @staticmethod
    def parse_tool_calls(response: str) -> tuple[str, list[dict]]:
        """Parse MCP tool call payloads from an LLM response.

        GoogleAIClient appends a '[TOOL_CALLS]...' JSON suffix when the model
        returns a functionCall part. This method splits text from tool calls.

        Returns:
            (clean_text, tool_calls) — tool_calls is [] when none present.
        """
        import json as _json
        marker = "[TOOL_CALLS]"
        if marker not in response:
            return response, []
        text_part, _, json_part = response.partition(marker)
        try:
            tool_calls = _json.loads(json_part.strip())
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
        except Exception:
            tool_calls = []
        return text_part.strip(), tool_calls

    async def close(self) -> None:
        await self._ollama.close()
        await self._ollama_cloud.close()
        await self._openrouter.close()
        await self._google.close()
