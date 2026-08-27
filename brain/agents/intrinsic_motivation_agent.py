"""IntrinsicMotivationAgent — Orbitofrontal Cortex + Ventral Striatum mapping.

Maintains a soft drive hierarchy with four drives:
  learn (0.4) · help (0.3) · explore (0.2) · rest (0.1)

Weights shift in response to recent events (successes, fatigue, boredom).
Every 5 minutes it publishes MOTIVATION_DRIVE → BehaviorAgent + CuriosityAgent
so they can adjust their curiosity frequency and topic selection.

Also absorbs knowledge gaps from MetacognitionAgent to build a self-directed
learning agenda.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from .base_agent import AgentMessage, BaseAgent, MessageType

if TYPE_CHECKING:
    from brain.llm.llm_router import LLMRouter
    from brain.memory.episodic_memory import EpisodicMemory
    from brain.memory.semantic_memory import SemanticMemory

log = logging.getLogger(__name__)

_RECOMPUTE_INTERVAL = 300.0   # 5 minutes
# Cognitive drives (existing)
_DRIVES_DEFAULT = {"learn": 0.40, "help": 0.30, "explore": 0.20, "rest": 0.10}
# v5.0 Homeostatic drives — track separately, normalised 0.0–1.0
_HOMEO_DEFAULT  = {"curiosity": 0.30, "social": 0.30, "energy": 0.80}
_DRIVE_ADJUST   = 0.05    # amount to nudge per event
_MIN_DRIVE      = 0.05
_MAX_DRIVE      = 0.70
_HOMEO_DECAY    = 0.02    # curiosity/social increase this much per recompute cycle when not stimulated
_ENERGY_DECAY   = 0.01    # energy decreases per recompute cycle (simulates uptime fatigue)


def _clamp(v: float) -> float:
    return max(_MIN_DRIVE, min(_MAX_DRIVE, v))


def _normalise(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values()) or 1.0
    return {k: v / total for k, v in d.items()}


class IntrinsicMotivationAgent(BaseAgent):
    """OFC + Ventral Striatum — goal-directed drive system."""

    name = "intrinsic_motivation_agent"

    def __init__(self, bus: asyncio.Queue, llm: "LLMRouter",
                 episodic: "EpisodicMemory", semantic: "SemanticMemory"):
        super().__init__(bus)
        self._llm      = llm
        self._episodic = episodic
        self._semantic = semantic
        self._drives   = dict(_DRIVES_DEFAULT)
        self._homeo    = dict(_HOMEO_DEFAULT)   # Curiosity / Social / Energy
        self._knowledge_gaps: list[str] = []
        self._last_publish_ts: float = 0.0
        self._workspace_idle_cycles: int = 0    # cycles with no new Workspace input
        self._person_present: bool = False
        self._conversation_active: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("IntrinsicMotivationAgent: started — drives=%s", self._drives)
        asyncio.create_task(self._drive_loop(), name="motivation-loop")
        while self._running:
            await asyncio.sleep(1)

    # ── Message handling ─────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        mtype = message.type

        if mtype == MessageType.COGNITION_RESPONSE:
            if message.data.get("success"):
                self._nudge("help", +_DRIVE_ADJUST)
                self._nudge("learn", -_DRIVE_ADJUST * 0.5)
            # Conversation active → reset Social drive toward neutral
            self._conversation_active = True
            self._homeo["social"] = max(0.0, self._homeo["social"] - _DRIVE_ADJUST * 2)

        elif mtype == MessageType.PERCEPTION_SPEECH:
            self._conversation_active = True
            self._workspace_idle_cycles = 0
            # New input → Curiosity drive satisfied temporarily
            self._homeo["curiosity"] = max(0.0, self._homeo["curiosity"] - _DRIVE_ADJUST)

        elif mtype == MessageType.PERSON_ENTER:
            self._person_present = True

        elif mtype == MessageType.PERSON_LEAVE:
            self._person_present = False
            self._conversation_active = False

        elif mtype == MessageType.CURIOSITY_TRIGGER:
            self._nudge("explore", +_DRIVE_ADJUST)

        elif mtype == MessageType.INTERO_STATE:
            label = message.data.get("label", "")
            if label in ("overwhelmed", "memory_pressure", "hot"):
                self._nudge("rest", +_DRIVE_ADJUST * 2)
                self._nudge("explore", -_DRIVE_ADJUST)
                # High load → Energy drive drops
                self._homeo["energy"] = max(0.0, self._homeo["energy"] - _DRIVE_ADJUST * 2)
            elif label == "comfortable":
                self._homeo["energy"] = min(1.0, self._homeo["energy"] + _DRIVE_ADJUST)

        elif mtype == MessageType.DREAM_DONE:
            self._nudge("learn", +_DRIVE_ADJUST * 2)
            self._nudge("rest", -_DRIVE_ADJUST)
            # Post-dream: Energy partially restored
            self._homeo["energy"] = min(1.0, self._homeo["energy"] + _DRIVE_ADJUST * 3)

        elif mtype == MessageType.METACOG_CONFIDENCE:
            new_gaps = message.data.get("knowledge_gaps", [])
            for gap in new_gaps:
                if gap not in self._knowledge_gaps:
                    self._knowledge_gaps.append(gap)
            if len(self._knowledge_gaps) > 20:
                self._knowledge_gaps = self._knowledge_gaps[-20:]
            if new_gaps:
                self._nudge("learn", +_DRIVE_ADJUST)

        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _nudge(self, drive: str, delta: float) -> None:
        if drive in self._drives:
            self._drives[drive] = _clamp(self._drives[drive] + delta)
            self._drives = _normalise(self._drives)

    async def _drive_loop(self) -> None:
        while self._running:
            await asyncio.sleep(_RECOMPUTE_INTERVAL)
            self._tick_homeostatic_drives()
            await self._publish_drives()

    def _tick_homeostatic_drives(self) -> None:
        """Decay/increase homeostatic drives each recompute cycle.

        Curiosity: increases if no new Workspace input recently (idle)
        Social: increases if person present but no conversation
        Energy: decreases with each tick (uptime fatigue), restored by comfortable intero state
        """
        self._workspace_idle_cycles += 1

        # Curiosity — increases when idle (no new inputs)
        if self._workspace_idle_cycles >= 1:
            self._homeo["curiosity"] = min(1.0, self._homeo["curiosity"] + _HOMEO_DECAY)

        # Social — increases when person present but not talking
        if self._person_present and not self._conversation_active:
            self._homeo["social"] = min(1.0, self._homeo["social"] + _HOMEO_DECAY * 1.5)
        elif not self._person_present:
            # Decay social drive slowly when no one is there
            self._homeo["social"] = max(0.0, self._homeo["social"] - _HOMEO_DECAY * 0.5)

        # Energy — decreases with uptime
        self._homeo["energy"] = max(0.0, self._homeo["energy"] - _ENERGY_DECAY)

        # ECO mode trigger: energy critically low
        if self._homeo["energy"] < 0.2:
            log.info("IntrinsicMotivationAgent: Energy Drive low (%.2f) — nudging rest",
                     self._homeo["energy"])
            self._nudge("rest", +_DRIVE_ADJUST * 2)

        # Reset conversation flag — will be set again on next PERCEPTION_SPEECH
        self._conversation_active = False

    async def _publish_drives(self) -> None:
        top_drive = max(self._drives, key=self._drives.get)
        await self.publish(AgentMessage(
            type=MessageType.MOTIVATION_DRIVE,
            source=self.name,
            data={
                "drives":         dict(self._drives),
                "homeostatic":    {k: round(v, 3) for k, v in self._homeo.items()},
                "top_drive":      top_drive,
                "knowledge_gaps": list(self._knowledge_gaps[:5]),
                "person_present": self._person_present,
                "ts":             time.time(),
            },
            priority=7,
        ))
        log.debug("IntrinsicMotivationAgent: top_drive=%s homeo=%s",
                  top_drive, {k: f"{v:.2f}" for k, v in self._homeo.items()})
