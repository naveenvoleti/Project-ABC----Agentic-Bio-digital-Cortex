"""
MCPToolRegistry — Model Context Protocol Tool Shim (Option A: Lightweight).

Generates a standard OpenAI / Gemini function-calling JSON schema from the
existing hardware drivers. The LLM sees the schema as its `tools` parameter
so it can discover capabilities natively without hardcoded prompt strings.

Flow:
    1. Orchestrator creates MCPToolRegistry(motor_drv, gpio_drv, audio_drv, hw)
    2. MCPToolRegistry.get_tools() → list[dict] injected into LLMRouter.infer()
    3. LLM responds with a tool_use block: {"name": "pan_head", "args": {...}}
    4. Orchestrator calls MCPToolRegistry.dispatch(name, args) → routes to driver
"""
from __future__ import annotations

import asyncio
from typing import Any

from brain.utils.logger import get_logger

log = get_logger(__name__)


# ── Gemini / OpenAI-compatible function schema helpers ────────────────────────

def _prop(type_: str, description: str, enum: list | None = None) -> dict:
    p: dict = {"type": type_, "description": description}
    if enum:
        p["enum"] = enum
    return p


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    """Build a standard OpenAI-compatible function-calling tool schema."""
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


class MCPToolRegistry:
    """Lightweight MCP-compatible tool registry for Project-ABC hardware.

    Wraps MotorDriver, GPIODriver, and AudioDriver into a standard JSON tool
    schema that is injected into every Gemini / LLM call. The LLM can then
    issue structured tool calls which the Orchestrator dispatches here.

    Gracefully handles missing drivers — tools are only advertised when the
    underlying hardware is confirmed available.
    """

    def __init__(
        self,
        motor_drv=None,
        gpio_drv=None,
        audio_drv=None,
        hw=None,          # HardwareCapabilities
    ):
        self._motor = motor_drv
        self._gpio  = gpio_drv
        self._audio = audio_drv
        self._hw    = hw
        self._tools: list[dict] = []
        self._build_schema()
        log.info("MCPToolRegistry: %d tools advertised", len(self._tools))

    # ── Schema builder ────────────────────────────────────────────────────────

    def _build_schema(self) -> None:
        tools: list[dict] = []

        # Pan-tilt head (motor driver)
        if self._motor is not None and getattr(self._hw, "motor_available", False):
            tools.append(_tool(
                name="pan_head",
                description="Rotate the robot's pan-tilt head left or right by a given angle.",
                properties={
                    "degrees": _prop("number", "Pan angle in degrees. Range: -90 (left) to 90 (right)."),
                },
                required=["degrees"],
            ))
            tools.append(_tool(
                name="tilt_head",
                description="Tilt the robot's head up or down by a given angle.",
                properties={
                    "degrees": _prop("number", "Tilt angle in degrees. Range: -45 (down) to 45 (up)."),
                },
                required=["degrees"],
            ))
            tools.append(_tool(
                name="center_head",
                description="Return the robot's head to the center position (0°, 0°).",
                properties={},
            ))
            tools.append(_tool(
                name="nod_head",
                description="Perform a nod gesture (tilt up-down twice).",
                properties={},
            ))
            tools.append(_tool(
                name="shake_head",
                description="Perform a head-shake gesture (pan left-right twice).",
                properties={},
            ))

        # Differential wheel drive
        if getattr(self._hw, "has_differential_drive", False):
            tools.append(_tool(
                name="move_wheels",
                description="Drive the robot's wheels to move forward, backward, or rotate.",
                properties={
                    "direction": _prop("string", "Direction of movement.",
                                       enum=["forward", "backward", "rotate_left", "rotate_right", "stop"]),
                    "speed":    _prop("integer", "Speed as a percentage (0–100). Default: 50."),
                    "duration": _prop("number",  "Duration in seconds. Default: 1.0."),
                },
                required=["direction"],
            ))

        # GPIO / LED control
        if self._gpio is not None and getattr(self._hw, "gpio_available", False):
            tools.append(_tool(
                name="toggle_led",
                description="Turn a GPIO-connected LED on or off.",
                properties={
                    "pin":   _prop("integer", "BCM GPIO pin number."),
                    "value": _prop("boolean", "True = LED on, False = LED off."),
                },
                required=["pin", "value"],
            ))
            tools.append(_tool(
                name="read_gpio",
                description="Read the current value of a GPIO input pin (e.g., a button or sensor).",
                properties={
                    "pin": _prop("integer", "BCM GPIO pin number."),
                },
                required=["pin"],
            ))

        # TTS / Audio (always available when speaker present)
        if self._audio is not None and getattr(self._hw, "speaker_available", False):
            tools.append(_tool(
                name="speak_text",
                description="Synthesize speech and play it aloud through the robot's speaker.",
                properties={
                    "text": _prop("string", "The text to speak aloud."),
                },
                required=["text"],
            ))

        # V10.0 — Epistemic tools (always available — LearningAgent / autonomous learning)
        tools.append(_tool(
            name="web_search",
            description="Search the web (DuckDuckGo) for factual information on a topic. "
                        "Use this to fill knowledge gaps or look up current facts.",
            properties={
                "query": _prop("string", "The search query."),
                "limit": _prop("integer", "Number of results to return (1-10). Default: 5."),
            },
            required=["query"],
        ))
        tools.append(_tool(
            name="semantic_lookup",
            description="Search Brain's long-term semantic memory for stored knowledge about a topic.",
            properties={
                "topic": _prop("string", "The topic to search for in semantic memory."),
                "limit": _prop("integer", "Max results to return (1-5). Default: 3."),
            },
            required=["topic"],
        ))

        self._tools = tools

    # ── Public API ────────────────────────────────────────────────────────────

    def get_tools(self) -> list[dict]:
        """Return the full list of tool schemas for injection into LLM calls."""
        return list(self._tools)

    def get_tool_names(self) -> list[str]:
        return [t["name"] for t in self._tools]

    async def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict:
        """Route an LLM tool call to the appropriate hardware driver.

        Returns a dict with 'result' (success value) and 'error' (or None).
        All driver calls are wrapped in try/except so a hardware fault never
        crashes the LLM response pipeline.
        """
        log.info("MCPToolRegistry: dispatch tool='%s' args=%s", tool_name, args)
        try:
            return await self._dispatch_inner(tool_name, args)
        except Exception as e:
            log.warning("MCPToolRegistry: dispatch error tool='%s': %s", tool_name, e)
            return {"result": None, "error": str(e)}

    async def _dispatch_inner(self, tool_name: str, args: dict) -> dict:
        loop = asyncio.get_event_loop()

        # ── Motor / head tools ────────────────────────────────────────────────
        if tool_name == "pan_head" and self._motor:
            await loop.run_in_executor(None, self._motor.pan, float(args["degrees"]))
            return {"result": f"head panned to {args['degrees']}°"}

        if tool_name == "tilt_head" and self._motor:
            await loop.run_in_executor(None, self._motor.tilt, float(args["degrees"]))
            return {"result": f"head tilted to {args['degrees']}°"}

        if tool_name == "center_head" and self._motor:
            await loop.run_in_executor(None, self._motor.center)
            return {"result": "head centered"}

        if tool_name == "nod_head" and self._motor:
            await loop.run_in_executor(None, self._motor.nod)
            return {"result": "nodded"}

        if tool_name == "shake_head" and self._motor:
            await loop.run_in_executor(None, self._motor.shake)
            return {"result": "shook head"}

        # ── Wheel tools ───────────────────────────────────────────────────────
        if tool_name == "move_wheels" and self._motor:
            direction = args.get("direction", "forward")
            speed     = int(args.get("speed", 50))
            duration  = float(args.get("duration", 1.0))
            await loop.run_in_executor(
                None, self._motor.move, speed, direction, duration
            )
            return {"result": f"moved {direction} speed={speed} for {duration}s"}

        # ── GPIO tools ────────────────────────────────────────────────────────
        if tool_name == "toggle_led" and self._gpio:
            pin   = int(args["pin"])
            value = bool(args["value"])
            await loop.run_in_executor(None, self._gpio.write, pin, value)
            return {"result": f"GPIO pin {pin} set to {'HIGH' if value else 'LOW'}"}

        if tool_name == "read_gpio" and self._gpio:
            pin    = int(args["pin"])
            result = await loop.run_in_executor(None, self._gpio.read, pin)
            return {"result": bool(result)}

        # ── Audio tools ───────────────────────────────────────────────────────
        if tool_name == "speak_text" and self._audio:
            text = str(args.get("text", ""))
            await loop.run_in_executor(None, self._audio.speak, text)
            return {"result": f"spoken: '{text[:40]}'"}

        # ── V10.0 Epistemic tools ─────────────────────────────────────────────
        if tool_name == "web_search":
            query = str(args.get("query", ""))
            limit = int(args.get("limit", 5))
            return await self._web_search(query, limit)

        if tool_name == "semantic_lookup":
            topic = str(args.get("topic", ""))
            limit = int(args.get("limit", 3))
            return await self._semantic_lookup(topic, limit)

        return {"result": None, "error": f"unknown tool: {tool_name}"}

    async def _web_search(self, query: str, limit: int = 5) -> dict:
        """V10.0 — Epistemic tool: DuckDuckGo search for LearningAgent."""
        try:
            import httpx
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url, params=params)
                data = resp.json()
            results = []
            # AbstractText is the Wikipedia summary
            if data.get("AbstractText"):
                results.append({"title": data.get("Heading", query),
                                 "body": data["AbstractText"][:500]})
            # RelatedTopics
            for t in data.get("RelatedTopics", [])[:limit - 1]:
                if isinstance(t, dict) and t.get("Text"):
                    results.append({"title": t.get("FirstURL", "").split("/")[-1].replace("_", " "),
                                    "body": t["Text"][:300]})
            content = "\n\n".join(f"{r['title']}: {r['body']}" for r in results[:limit])
            log.info("MCPToolRegistry: web_search '%s' → %d results", query[:40], len(results))
            return {"result": content or "No results found.", "source": "duckduckgo"}
        except ImportError:
            log.warning("MCPToolRegistry: httpx not installed — web_search unavailable")
            return {"result": None, "error": "httpx not installed"}
        except Exception as e:
            log.warning("MCPToolRegistry: web_search failed: %s", e)
            return {"result": None, "error": str(e)}

    async def _semantic_lookup(self, topic: str, limit: int = 3) -> dict:
        """V10.0 — Epistemic tool: query Brain's SemanticMemory."""
        # SemanticMemory is not injected here by default — return not-available
        # Orchestrator can inject it via set_semantic() if desired
        semantic = getattr(self, "_semantic", None)
        if semantic is None:
            return {"result": "Semantic memory not connected to MCP.", "error": None}
        try:
            results = semantic.search(topic, limit=limit)
            if not results:
                return {"result": f"No memory found about: {topic}", "error": None}
            content = "\n".join(f"- {r.get('content', '')[:200]}" for r in results)
            return {"result": content, "source": "semantic_memory"}
        except Exception as e:
            return {"result": None, "error": str(e)}
