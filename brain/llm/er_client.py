"""
Gemini Robotics ER Client — google-genai SDK wrapper.

Uses the official SDK (not raw REST) so ThinkingConfig, Part.from_bytes,
and all ER-specific features work correctly.

Install: pip install google-genai

Thinking budget guide (from ER 1.6 docs):
  thinking_budget=0     — spatial tasks (object detection, pointing) — fastest
  thinking_budget=512   — general visual questions
  thinking_budget=1024  — complex reasoning, task planning
  thinking_budget=2048  — instrument reading, multi-step inference
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from brain.utils.logger import get_logger

log = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="er-client")


class ERClient:
    """Async wrapper around google-genai for Gemini Robotics ER 1.6."""

    def __init__(self, api_key: str, model: str = "gemini-robotics-er-1.6-preview"):
        self._api_key = api_key
        self._model = model
        self._client = None
        self._types = None

        if not api_key:
            log.warning("ERClient: no API key — ER reasoning disabled")
            return

        try:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=api_key)
            self._types = types
            log.info("ERClient: google-genai SDK ready (%s)", model)
        except ImportError:
            log.warning(
                "ERClient: google-genai not installed — run: pip install google-genai. "
                "ER reasoning will fall back to REST-based Gemini."
            )

    def is_available(self) -> bool:
        return self._client is not None

    # ── Core generation ───────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        frame_b64: str = "",
        max_tokens: int = 1024,
        thinking_budget: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Send prompt (+ optional camera frame) to ER model, return text response."""
        if not self._client:
            return ""

        types = self._types
        contents: list[Any] = []

        # Image must come BEFORE text (ER model convention)
        if frame_b64:
            try:
                img_bytes = base64.b64decode(frame_b64)
                contents.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                )
            except Exception as e:
                log.warning("ERClient: frame decode failed: %s", e)

        # ER model doesn't have a separate systemInstruction field in the SDK —
        # prepend system context into the text prompt
        full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
        contents.append(full_prompt)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                _executor,
                lambda: self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                ),
            )
            text = response.text or ""
            log.info("ERClient [%s]: %d chars (thinking_budget=%d)", self._model, len(text), thinking_budget)
            return text
        except Exception as e:
            log.error("ERClient: generate failed: %s", e)
            return ""

    # ── Specialised ER tasks ──────────────────────────────────────────────────

    async def point_objects(
        self,
        frame_b64: str,
        queries: list[str],
        thinking_budget: int = 0,
    ) -> list[dict]:
        """Locate objects in the frame, return list of {point:[y,x], label:str}.

        Coordinates are normalised to 0–1000. thinking_budget=0 is fastest.
        """
        prompt = (
            f"Get all points matching the following objects: {', '.join(queries)}. "
            "The label returned should be an identifying name for the object detected. "
            "The answer should follow the json format: "
            '[{"point": [y, x], "label": "<label>"}, ...]. '
            "The points are in [y, x] format normalised to 0-1000. "
            "If no objects are found, return an empty JSON list []."
        )
        raw = await self.generate(
            prompt=prompt,
            frame_b64=frame_b64,
            max_tokens=512,
            thinking_budget=thinking_budget,
            temperature=1.0,
        )
        return _parse_json_list(raw)

    async def describe_scene(
        self,
        frame_b64: str,
        focus: str = "",
        thinking_budget: int = 512,
    ) -> str:
        """High-quality scene description using ER spatial understanding."""
        focus_clause = f" Focus on: {focus}." if focus else ""
        prompt = (
            "Describe what you see in this image concisely and accurately."
            f"{focus_clause} Include relevant objects, people, and spatial relationships."
        )
        return await self.generate(
            prompt=prompt,
            frame_b64=frame_b64,
            max_tokens=256,
            thinking_budget=thinking_budget,
        )

    async def plan_task(
        self,
        task: str,
        context: str = "",
        frame_b64: str = "",
        thinking_budget: int = 1024,
    ) -> dict:
        """Decompose a high-level task into typed steps using ER planning.

        Returns: {plan: [...], success_estimate: float, reasoning: str}
        """
        system = (
            "You are the High-Level Brain of an embodied robot. "
            "Given a task and visual context, produce a step-by-step plan as a JSON object with keys: "
            "'plan' (list of typed step objects), 'success_estimate' (0.0-1.0), 'reasoning' (str). "
            "Each step object: {\"type\": \"synthesize_script|verify|execute\", \"description\": \"...\", "
            "\"script_type\": \"python|arduino\" (for synthesize_script only)}. "
            "CRITICAL: You must use type 'synthesize_script' when the task requires writing code, tools, or scripts! "
            "Output valid JSON only."
        )
        user_msg = f"Task: {task}"
        if context:
            user_msg += f"\nContext: {context}"

        raw = await self.generate(
            prompt=user_msg,
            system_prompt=system,
            frame_b64=frame_b64,
            max_tokens=1024,
            thinking_budget=thinking_budget,
        )
        log.info("ERClient plan_task raw output:\n%s", raw)
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        # Fallback: treat lines as plain steps
        steps = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()]
        return {"plan": steps, "success_estimate": 0.5, "reasoning": raw}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_list(text: str) -> list[dict]:
    """Extract first JSON array from text, return [] on failure."""
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return []
