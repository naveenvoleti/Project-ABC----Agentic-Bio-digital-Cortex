"""
MirrorAgent — Observational Learning via grounded skill primitives (V6.0).
Watches SmolVLM2 frames, decomposes observed actions into available motor primitives,
and stores them as executable PLAN_STEP macros.
Maps to Mirror Neuron System (premotor cortex).
"""
from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.memory.soul_manager import SoulManager
from brain.utils.logger import get_logger

if TYPE_CHECKING:
    from brain.llm.llm_router import LLMRouter
    from brain.hardware.smolvlm2_processor import SmolVLM2Processor

log = get_logger(__name__)


class MirrorAgent(BaseAgent):
    name = "mirror_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        llm: "LLMRouter",
        soul: SoulManager,
        smolvlm2: "SmolVLM2Processor | None" = None,
        max_steps: int = 8,
    ):
        super().__init__(bus)
        self._llm = llm
        self._soul = soul
        self._smolvlm2 = smolvlm2
        self._max_steps = max_steps
        self._last_mirror_ts: float = 0.0
        self._mirror_cooldown_s: float = 5.0  # prevent rapid re-triggering on burst frames

    async def start(self) -> None:
        await super().start()
        log.info("MirrorAgent started%s", " (no SmolVLM2)" if self._smolvlm2 is None else "")
        while self._running:
            await asyncio.sleep(1)

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.VLM_SCAN_NOW:
            now = time.monotonic()
            if now - self._last_mirror_ts < self._mirror_cooldown_s:
                return
            self._last_mirror_ts = now
            frame = message.data.get("frame")
            if frame is not None:
                await self._mirror_action(frame)

    async def _mirror_action(self, frame) -> None:
        if self._smolvlm2 is None:
            return

        primitives = self._soul.get_primitive_skills()
        prompt = (
            f"Observe the action in this frame. "
            f"Decompose it ONLY into these available primitives: {primitives}. "
            f"List each step as: primitive_name(args). "
            f"Do not invent new primitives. Maximum {self._max_steps} steps."
        )

        try:
            self._smolvlm2.push_frame(frame)
            result = await self._smolvlm2.scan_async(prompt)
            description = result.description if result else ""
        except Exception as e:
            log.warning("MirrorAgent: scan_async failed: %s", e)
            return

        if not description:
            return

        steps = self._parse_grounded_steps(description, primitives)
        if not steps:
            log.debug("MirrorAgent: no grounded steps parsed from VLM output")
            return

        macro_name = f"mirror_{int(time.time())}"
        macro = {
            "name": macro_name,
            "trigger": "mirror",
            "steps": steps,
            "source": "mirror_agent",
            "created": datetime.utcnow().isoformat(),
        }
        try:
            self._soul.add_skill(macro)
            log.info("MirrorAgent: stored macro '%s' (%d steps)", macro_name, len(steps))
        except Exception as e:
            log.warning("MirrorAgent: add_skill failed: %s", e)

        for i, step in enumerate(steps):
            await self.publish(AgentMessage(
                type=MessageType.PLAN_STEP,
                source=self.name,
                data={
                    "step": step,
                    "index": i,
                    "total": len(steps),
                    "macro": True,
                    "macro_name": macro_name,
                },
                priority=5,
            ))

    def _parse_grounded_steps(self, text: str, primitives: list[str]) -> list[str]:
        """Extract only steps that reference a known primitive name."""
        # Build set of primitive function names (e.g. "move_pan", "speak")
        known_names = {p.split("(")[0].strip() for p in primitives}
        steps: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip numbering / bullets
            m = re.match(r'^(\d+[.)]\s*|[-*]\s*)(.*)', line)
            content = m.group(2).strip() if m else line
            # Accept line if it starts with a known primitive name followed by "("
            fn_match = re.match(r'^(\w+)\s*\(', content)
            if fn_match and fn_match.group(1) in known_names:
                steps.append(content)
            if len(steps) >= self._max_steps:
                break
        return steps
