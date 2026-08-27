"""
SynthesisAgent — Neocortical Synthesis (V6.0).
Rewrites failing agent methods and compiles LLM response patterns into local Python reflexes.
Maps to Neuroplasticity / Prefrontal Cortex self-repair.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.utils.logger import get_logger

if TYPE_CHECKING:
    from brain.llm.llm_router import LLMRouter
    from brain.memory.soul_manager import SoulManager

log = get_logger(__name__)

FORBIDDEN_IMPORTS = {
    "os", "shutil", "sys", "subprocess", "importlib",
    "socket", "threading", "multiprocessing", "ctypes",
    "pickle", "marshal", "eval", "exec",
}

_SYNTHESIS_SYSTEM = (
    "You are a Python expert fixing a bug in a robotics brain agent. "
    "Output ONLY valid Python code. No markdown fences. No explanation. No imports except "
    "datetime, re, json, math, random, or collections."
)

_REFLEX_SYSTEM = (
    "You are a Python expert writing a fast local decision function. "
    "Output ONLY valid Python. No markdown. No explanation. "
    "Allowed imports: datetime, re. "
    "Define: trigger_pattern (str regex) and result (str)."
)


class SynthesisAgent(BaseAgent):
    name = "synthesis_agent"
    _SYNTHESIS_LIMIT = 5  # max re-syntheses per agent per brain session

    def __init__(
        self,
        bus: asyncio.Queue,
        llm: "LLMRouter",
        soul: "SoulManager",
        sandbox_dir: Path,
        code_health_path: Path,
        crash_threshold: int = 3,
        confidence_threshold: float = 0.2,
        orchestrator_ref=None,
    ):
        super().__init__(bus)
        self._llm = llm
        self._soul = soul
        self._sandbox_dir = Path(sandbox_dir)
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)
        self._code_health_path = Path(code_health_path)
        self._crash_threshold = crash_threshold
        self._confidence_threshold = confidence_threshold
        self._orchestrator_ref = orchestrator_ref
        # session-scoped synthesis counters and blacklist
        self._synthesis_count: dict[str, int] = {}
        self._blacklisted: set[str] = set()

    async def start(self) -> None:
        await super().start()
        log.info("SynthesisAgent started — sandbox: %s", self._sandbox_dir)
        while self._running:
            await asyncio.sleep(1)

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.NEURO_SYNTHESIS:
            mode = message.data.get("mode", "fix")
            if mode == "reflex_compile":
                await self._compile_reflexes_from_pattern(message.data)
            elif mode == "device_discovery":
                i2c_addr = int(message.data.get("i2c_addr", 0))
                device_info = self._discover_device(i2c_addr)
                if device_info.get("name") != "unknown":
                    await self._synthesize_driver(device_info)
                else:
                    log.info("SynthesisAgent: unknown I2C device at 0x%02X — no driver generated",
                             i2c_addr)
            else:
                await self._synthesize(message.data)
        elif message.type == MessageType.DRIVER_SYNTHESIS:
            await self._synthesize_external_script(message.data)
        elif message.type == MessageType.DREAM_DONE:
            # DreamAgent also fires DREAM_DONE with distillation data
            patterns = message.data.get("distill_patterns", {})
            for intent, responses in patterns.items():
                await self._compile_reflexes_from_pattern({
                    "intent": intent,
                    "sample_responses": responses,
                })

    # ── Code Synthesis ────────────────────────────────────────────────────────

    async def _synthesize(self, data: dict) -> None:
        agent_name = data.get("agent", "unknown")
        error_text = data.get("error", "")

        if agent_name in self._blacklisted:
            log.warning("SynthesisAgent: %s is blacklisted — skipping", agent_name)
            return

        session_count = self._synthesis_count.get(agent_name, 0)
        if session_count >= self._SYNTHESIS_LIMIT:
            log.warning("SynthesisAgent: session limit reached for %s", agent_name)
            self._blacklisted.add(agent_name)
            return

        log.info("SynthesisAgent: synthesizing fix for %s (attempt %d)", agent_name, session_count + 1)
        self._synthesis_count[agent_name] = session_count + 1

        # Collect agent source + instance attrs for context
        source_code, attrs = self._get_agent_context(agent_name)

        prompt = (
            f"Agent class: {agent_name}\n"
            f"Available instance attributes: {attrs}\n"
            f"Do NOT access any private members not in the above list.\n"
            f"Recent errors:\n{error_text}\n\n"
            f"Failing agent source:\n{source_code}\n\n"
            f"Write a corrected version of the failing method ONLY."
        )

        try:
            code = await self._llm.infer(
                user_message=prompt,
                system_prompt=_SYNTHESIS_SYSTEM,
                max_tokens=512,
            )
        except Exception as e:
            log.error("SynthesisAgent: LLM call failed: %s", e)
            return

        if not code or len(code.strip()) < 20:
            log.warning("SynthesisAgent: LLM returned empty/short code for %s", agent_name)
            return

        ok, reason = self._guardrail_check(code)
        if not ok:
            log.warning("SynthesisAgent: guardrail rejected code for %s: %s", agent_name, reason)
            await self._update_code_health(agent_name, error_text, f"REJECTED: {reason}")
            return

        ts = int(time.time())
        file_path = self._sandbox_dir / f"{agent_name}_fix_{ts}.py"
        await asyncio.to_thread(file_path.write_text, code, encoding="utf-8")
        log.info("SynthesisAgent: wrote fix to %s", file_path)
        await self._update_code_health(agent_name, error_text, "PENDING_TEST")

        # Snapshot DNA before any hot-load attempt
        self._snapshot_dna()

        await self.publish(AgentMessage(
            type=MessageType.CODE_VALIDATED,
            source=self.name,
            data={
                "file_path": str(file_path),
                "target_agent": agent_name,
                "fix_type": "method_fix",
                "method_name": self._extract_method_name(code),
            },
            priority=2,
        ))

    def _get_agent_context(self, agent_name: str) -> tuple[str, list[str]]:
        """Return (source_code, instance_attrs) for the named agent."""
        attrs: list[str] = []
        source_code = ""
        if self._orchestrator_ref:
            agent_obj = self._orchestrator_ref.get_agent(agent_name)
            if agent_obj:
                attrs = list(vars(agent_obj).keys())
                try:
                    source_code = inspect.getsource(type(agent_obj))
                    # Truncate to avoid exceeding LLM context
                    if len(source_code) > 3000:
                        source_code = source_code[:3000] + "\n# ... (truncated)"
                except Exception:
                    pass
        return source_code, attrs

    def _extract_method_name(self, code: str) -> str:
        """Extract the first def name from synthesized code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception:
            pass
        return ""

    # ── Reflex Compilation ────────────────────────────────────────────────────

    async def _compile_reflexes_from_pattern(self, data: dict) -> None:
        intent = data.get("intent", "")
        sample_responses = data.get("sample_responses", [])
        if not intent or not sample_responses:
            return

        log.info("SynthesisAgent: compiling reflex for intent '%s'", intent)

        prompt = (
            f"Intent to detect: '{intent}'\n"
            f"Sample responses when this intent is detected:\n"
            f"{sample_responses}\n\n"
            f"Write a Python snippet that:\n"
            f"1. Sets trigger_pattern to a regex string matching this intent.\n"
            f"2. Sets result to an appropriate short response string.\n"
            f"Example:\n"
            f"trigger_pattern = r'^(what time|tell me the time|current time)'\n"
            f"result = __import__('datetime').datetime.now().strftime('%H:%M')\n"
        )

        try:
            code = await self._llm.infer(
                user_message=prompt,
                system_prompt=_REFLEX_SYSTEM,
                max_tokens=200,
            )
        except Exception as e:
            log.error("SynthesisAgent: reflex LLM call failed: %s", e)
            return

        if not code or len(code.strip()) < 10:
            return

        ok, reason = self._guardrail_check(code)
        if not ok:
            log.warning("SynthesisAgent: reflex guardrail rejected for '%s': %s", intent, reason)
            return

        # Generate embedding for semantic matching
        try:
            embedding = await self._llm.embed(intent)
        except Exception:
            embedding = []

        ts = int(time.time())
        reflex_id = f"reflex_{intent[:20].replace(' ', '_')}_{ts}"
        file_path = self._sandbox_dir / f"{reflex_id}.py"
        await asyncio.to_thread(file_path.write_text, code, encoding="utf-8")

        await self.publish(AgentMessage(
            type=MessageType.CODE_VALIDATED,
            source=self.name,
            data={
                "file_path": str(file_path),
                "fix_type": "reflex",
                "reflex_id": reflex_id,
                "intent": intent,
                "trigger_embedding": embedding,
                "python_code": code,
                "source_pattern_count": len(sample_responses),
            },
            priority=4,
        ))

    # ── V8.0 External Script Synthesis (DRIVER_SYNTHESIS) ────────────────────

    async def _synthesize_external_script(self, data: dict) -> None:
        """Generate a standalone .py or .ino utility script for a hardware task.

        Unlike _synthesize() which patches live agent methods, this writes a
        completely independent script to brain/sandbox/ that Orchestrator will
        run in a subprocess after TesterAgent validates it.
        """
        task = data.get("task", "unknown_task")
        description = data.get("description", task)
        script_type = data.get("script_type", "python")
        error_context = data.get("error", "")

        task_slug = re.sub(r"[^a-z0-9_]", "_", task.lower())[:30]
        ts = int(time.time())

        if script_type == "arduino":
            ext = ".ino"
            previous_note = f"\nPrevious attempt failed:\n{error_context}" if error_context else ""
            prompt = (
                f"Write Arduino C++ code to: {description}\n"
                f"Task: {task}{previous_note}\n"
                "Output ONLY valid Arduino .ino code. No markdown. No explanation."
            )
        elif script_type == "html" or "ui" in task.lower() or "display" in task.lower():
            script_type = "html"
            ext = ".html"
            previous_note = f"\nPrevious attempt failed:\n{error_context}" if error_context else ""
            prompt = (
                f"Write the complete HTML code to: {description}\n"
                f"Task: {task}{previous_note}\n"
                "Output ONLY valid HTML. No markdown fences. No explanation."
            )
        else:
            ext = ".py"
            previous_note = f"\nPrevious attempt failed with error:\n{error_context}" if error_context else ""
            prompt = (
                f"Write a standalone Python script to: {description}\n"
                f"Task: {task}{previous_note}\n"
                "Allowed imports: subprocess, pathlib, time, os.path only.\n"
                "The script must complete and exit with returncode 0 on success.\n"
                "Output ONLY valid Python. No markdown. No explanation."
            )

        try:
            code = await self._llm.infer(
                user_message=prompt,
                system_prompt=_SYNTHESIS_SYSTEM,
                max_tokens=512,
            )
        except Exception as e:
            log.error("SynthesisAgent: external script LLM failed: %s", e)
            return

        if not code or len(code.strip()) < 20:
            log.warning("SynthesisAgent: empty external script for '%s'", task)
            return

        # Strip markdown fences if LLM wrapped output
        code = re.sub(r"^```[a-z]*\n?", "", code, flags=re.MULTILINE)
        code = re.sub(r"^```\s*$", "", code, flags=re.MULTILINE).strip()

        if script_type == "python":
            ok, reason = self._guardrail_check(code)
            if not ok:
                log.warning("SynthesisAgent: external script guardrail blocked for '%s': %s",
                            task, reason)
                return

        dest = self._sandbox_dir / f"script_{task_slug}_{ts}{ext}"
        await asyncio.to_thread(dest.write_text, code, encoding="utf-8")
        log.info("SynthesisAgent: synthesized external script → %s", dest)

        await self.publish(AgentMessage(
            type=MessageType.CODE_VALIDATED,
            source=self.name,
            data={
                "file_path": str(dest),
                "fix_type": "external_script",
                "script_type": script_type,
                "task": task,
            },
            priority=4,
        ))

    # ── Guardrails ────────────────────────────────────────────────────────────

    def _guardrail_check(self, code: str) -> tuple[bool, str]:
        """AST-based forbidden import scanner. Returns (ok, reason)."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name.split(".")[0] for alias in node.names]
                bad = FORBIDDEN_IMPORTS.intersection(set(names))
                if bad:
                    return False, f"Forbidden import(s): {bad}"
        return True, "ok"

    # ── Digital DNA ───────────────────────────────────────────────────────────

    def _snapshot_dna(self) -> None:
        """Compute SHA256 of all brain/*.py and store via SoulManager."""
        brain_dir = Path(__file__).parent.parent
        h = hashlib.sha256()
        py_files = sorted(brain_dir.rglob("*.py"))
        # Exclude sandbox itself from DNA to avoid self-referential drift
        py_files = [f for f in py_files if "sandbox" not in f.parts]
        for f in py_files:
            try:
                h.update(f.read_bytes())
            except Exception:
                pass
        checksum = h.hexdigest()
        try:
            self._soul.store_dna_checksum(checksum, len(py_files))
        except Exception as e:
            log.warning("DNA snapshot failed: %s", e)

    # ── V7.0 Native I2C/GPIO Device Discovery ────────────────────────────────

    # Common I2C device addresses → device name + capabilities
    _KNOWN_DEVICES: dict[int, dict] = {
        0x48: {"name": "ADS1115", "type": "adc",         "description": "16-bit ADC, 4 channels"},
        0x40: {"name": "INA219",  "type": "power_sensor", "description": "current/voltage sensor"},
        0x77: {"name": "BMP280",  "type": "env_sensor",   "description": "temperature/pressure sensor"},
        0x76: {"name": "BME280",  "type": "env_sensor",   "description": "humidity/temperature/pressure"},
        0x68: {"name": "MPU6050", "type": "imu",          "description": "6-axis gyro/accelerometer"},
        0x69: {"name": "MPU6050", "type": "imu",          "description": "6-axis gyro/accelerometer (alt)"},
        0x3C: {"name": "SSD1306", "type": "display",      "description": "128x64 OLED display"},
        0x27: {"name": "PCF8574", "type": "gpio_expander", "description": "8-bit I/O expander"},
        0x20: {"name": "PCF8574", "type": "gpio_expander", "description": "8-bit I/O expander (alt)"},
        0x29: {"name": "VL53L0X", "type": "lidar",        "description": "time-of-flight distance sensor"},
        0x10: {"name": "APDS9960","type": "gesture",      "description": "gesture/proximity/colour sensor"},
    }

    def _discover_device(self, i2c_addr: int) -> dict:
        """Look up a known device by I2C address.

        Returns device info dict with name, type, description, and addr.
        Falls back to {"name": "unknown", ...} if address not in registry.
        """
        info = self._KNOWN_DEVICES.get(i2c_addr, {
            "name": "unknown",
            "type": "unknown",
            "description": f"Unrecognised I2C device at 0x{i2c_addr:02X}",
        })
        return {"addr": f"0x{i2c_addr:02X}", **info}

    async def _synthesize_driver(self, device_info: dict) -> None:
        """V7.0 Native Tool Discovery — use ER reasoning to generate a Python driver.

        Writes the driver to brain/sandbox/<device_name>_driver_<ts>.py and
        emits CODE_VALIDATED for TesterAgent to validate before hot-loading.
        """
        name = device_info.get("name", "unknown")
        dtype = device_info.get("type", "unknown")
        addr = device_info.get("addr", "0x00")
        description = device_info.get("description", "")

        log.info("SynthesisAgent: synthesizing driver for %s (%s) at %s", name, dtype, addr)

        prompt = (
            f"Write a minimal Python driver class for the {name} ({description}) "
            f"connected via I2C at address {addr}. "
            f"Use the 'smbus2' library. "
            f"The class must have: __init__(self, bus_number=1), read(self) → dict, close(self). "
            f"read() must return a dict with meaningful sensor keys. "
            f"Output ONLY valid Python. No markdown. No explanation."
        )

        try:
            # Try er_reason first for structured output; fall back to standard infer
            result = await self._llm.er_reason(
                task=f"Generate Python I2C driver for {name}",
                context=prompt,
                max_tokens=512,
            )
            # er_reason returns a plan list — join into code if it looks like code
            plan = result.get("plan", [])
            code = "\n".join(plan) if plan else ""
            # If the plan is just natural language, fall back to direct infer
            if not code.strip().startswith(("class", "import", "def", "#")):
                code = await self._llm.infer(
                    user_message=prompt,
                    system_prompt=_SYNTHESIS_SYSTEM,
                    max_tokens=512,
                )
        except Exception as e:
            log.error("SynthesisAgent: driver synthesis LLM failed: %s", e)
            return

        if not code or len(code.strip()) < 30:
            log.warning("SynthesisAgent: empty driver code for %s", name)
            return

        ok, reason = self._guardrail_check(code)
        if not ok:
            log.warning("SynthesisAgent: driver guardrail rejected for %s: %s", name, reason)
            return

        ts = int(time.time())
        safe_name = name.lower().replace(" ", "_")
        file_path = self._sandbox_dir / f"{safe_name}_driver_{ts}.py"
        await asyncio.to_thread(file_path.write_text, code, encoding="utf-8")
        log.info("SynthesisAgent: driver written to %s", file_path)

        await self.publish(AgentMessage(
            type=MessageType.CODE_VALIDATED,
            source=self.name,
            data={
                "file_path": str(file_path),
                "fix_type": "driver",
                "device_name": name,
                "device_type": dtype,
                "device_addr": addr,
            },
            priority=5,
        ))

    async def _synthesize_external_script(self, data: dict) -> None:
        """V8.0 Synthesize an external script based on ER step description."""
        import time
        task = data.get("task", "")
        desc = data.get("description", "")
        script_type = data.get("script_type", "python")

        log.info("SynthesisAgent: synthesizing external script (%s): %s", script_type, desc)

        prompt = (
            f"Write a {script_type} script to accomplish the following step:\n"
            f"Description: {desc}\n"
            f"Overall Task: {task}\n"
            f"Output ONLY valid {script_type} code. No markdown. No explanation."
        )

        try:
            code = await self._llm.infer(
                user_message=prompt,
                system_prompt="You are a senior developer writing standalone scripts.",
                max_tokens=1024,
            )
        except Exception as e:
            log.error("SynthesisAgent: external script synthesis failed: %s", e)
            return

        if not code or len(code.strip()) < 10:
            log.warning("SynthesisAgent: empty script generated")
            return

        # Strip markdown code blocks if any
        if code.startswith("```"):
            lines = code.splitlines()
            if len(lines) > 2:
                code = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])

        ts = int(time.time())
        ext = "py" if script_type == "python" else ("ino" if script_type == "arduino" else "txt")
        file_path = self._sandbox_dir / f"script_{ts}.{ext}"
        await asyncio.to_thread(file_path.write_text, code, encoding="utf-8")
        log.info("SynthesisAgent: script written to %s", file_path)

        await self.publish(AgentMessage(
            type=MessageType.CODE_VALIDATED,
            source=self.name,
            data={
                "file_path": str(file_path),
                "fix_type": "external_script",
                "script_type": script_type,
                "task": task,
            },
            priority=5,
        ))

    # ── CODE_HEALTH logging ───────────────────────────────────────────────────

    async def _update_code_health(self, agent: str, error: str, outcome: str) -> None:
        try:
            text = await asyncio.to_thread(self._code_health_path.read_text, encoding="utf-8")
            now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
            short_err = error[:80].replace("|", "/").replace("\n", " ")
            count = self._synthesis_count.get(agent, 0)
            blacklisted = "YES" if agent in self._blacklisted else "no"
            row = f"| {agent} | {count} | {short_err} | {count} | {outcome} | {blacklisted} |"
            if agent in text:
                import re
                text = re.sub(rf"\| {re.escape(agent)} \|.*\|", row, text)
            else:
                text = text.rstrip() + f"\n{row}\n"
            await asyncio.to_thread(self._code_health_path.write_text, text, encoding="utf-8")
        except Exception as e:
            log.warning("CODE_HEALTH update failed: %s", e)
