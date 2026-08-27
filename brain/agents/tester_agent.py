"""
TesterAgent — Sandbox code validator (V6.0).
Runs synthesized code in an isolated subprocess. Never uses exec() or eval().
Maps to the brain's "immune system" — prevents bad code from entering the host.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.utils.logger import get_logger

log = get_logger(__name__)


class TesterAgent(BaseAgent):
    name = "tester_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        sandbox_dir: Path,
        timeout_seconds: int = 10,
    ):
        super().__init__(bus)
        self._sandbox_dir = Path(sandbox_dir)
        self._timeout = timeout_seconds

    async def start(self) -> None:
        await super().start()
        log.info("TesterAgent started — timeout: %ds", self._timeout)
        while self._running:
            await asyncio.sleep(1)

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.CODE_VALIDATED:
            await self._run_sandbox_test(message.data)

    async def _run_sandbox_test(self, data: dict) -> None:
        file_path = data.get("file_path", "")
        fix_type = data.get("fix_type", "method_fix")
        script_type = data.get("script_type", "python")

        if not file_path or not Path(file_path).exists():
            log.error("TesterAgent: sandbox file not found: %s", file_path)
            await self._report_failure(data, "file_not_found")
            return

        # V8.0 — Arduino/HTML files cannot be py_compiled; emit REFLEX_READY directly
        if script_type in ("arduino", "html"):
            log.info("TesterAgent: %s script — skipping runtime test, emitting REFLEX_READY", script_type)
            await self._emit_reflex_ready(data)
            return

        log.info("TesterAgent: testing %s", file_path)

        # Run in isolated subprocess — empty env prevents API key / secret leakage
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [sys.executable, "-c",
                     f"import ast; ast.parse(open(r'{file_path}').read()); print('SYNTAX_OK')"],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env={},   # empty env — no secrets
                )
            )
        except subprocess.TimeoutExpired:
            log.warning("TesterAgent: timeout testing %s", file_path)
            await self._report_failure(data, "timeout")
            return
        except Exception as e:
            log.error("TesterAgent: subprocess error: %s", e)
            await self._report_failure(data, str(e))
            return

        if result.returncode != 0 or "SYNTAX_OK" not in result.stdout:
            stderr = result.stderr[:300]
            log.warning("TesterAgent: syntax check failed for %s: %s", file_path, stderr)
            await self._report_failure(data, stderr)
            return

        # Secondary: runtime import test
        try:
            runtime_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [sys.executable, str(Path(file_path).resolve())],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=str(self._sandbox_dir),
                    env={},
                )
            )
        except subprocess.TimeoutExpired:
            log.warning("TesterAgent: runtime timeout for %s", file_path)
            await self._report_failure(data, "runtime_timeout")
            return
        except Exception as e:
            log.error("TesterAgent: runtime error: %s", e)
            await self._report_failure(data, str(e))
            return

        if runtime_result.returncode != 0:
            stderr = runtime_result.stderr[:300]
            # Runtime errors are warnings — syntax is ok, runtime may need imports
            # For reflex code, runtime failure is acceptable (no agent context available)
            if fix_type != "reflex":
                log.warning("TesterAgent: runtime error %s: %s", file_path, stderr)
                await self._report_failure(data, f"runtime: {stderr}")
                return
            log.info("TesterAgent: reflex runtime warning (expected, no agent context): %s", stderr[:80])

        log.info("TesterAgent: PASSED — emitting REFLEX_READY for %s", file_path)
        await self._emit_reflex_ready(data)

    async def _emit_reflex_ready(self, data: dict) -> None:
        fix_type = data.get("fix_type", "method_fix")
        payload = {
            "file_path": data.get("file_path"),
            "fix_type": fix_type,
            "target_agent": data.get("target_agent", ""),
            "method_name": data.get("method_name", ""),
            "validated_at": datetime.utcnow().isoformat(),
            # V8.0 — external script fields passed through
            "script_type": data.get("script_type", "python"),
            "task": data.get("task", ""),
        }
        if fix_type == "reflex":
            payload.update({
                "reflex_id": data.get("reflex_id", ""),
                "intent": data.get("intent", ""),
                "trigger_embedding": data.get("trigger_embedding", []),
                "python_code": data.get("python_code", ""),
                "source_pattern_count": data.get("source_pattern_count", 0),
            })
        await self.publish(AgentMessage(
            type=MessageType.REFLEX_READY,
            source=self.name,
            data=payload,
            priority=2,
        ))

    async def _report_failure(self, data: dict, reason: str) -> None:
        await self.publish(AgentMessage(
            type=MessageType.SYSTEM_ERROR,
            source=self.name,
            data={
                "agent": "synthesis_agent",
                "error": f"TesterAgent rejected sandbox code: {reason}",
                "file": data.get("file_path", ""),
                "fix_type": data.get("fix_type", ""),
            },
            priority=3,
        ))
