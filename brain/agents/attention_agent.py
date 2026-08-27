"""AttentionAgent — Posterior Parietal Cortex + Dorsal Anterior Cingulate mapping.

Implements a two-channel attention system:
  • Top-down (task-driven): locks onto the current intent/topic from COGNITION_INTENT.
  • Bottom-up (surprise-driven): detects unexpected/high-salience inputs and fires
    ATTENTION_SHIFT to redirect the cognitive pipeline.

Downstream consumers:
  • CognitionAgent — uses ATTENTION_FOCUS.focus_target to bias context assembly.
  • BehaviorAgent  — shifts behaviour state on ATTENTION_SHIFT.
  • EmotionEngine  — gets ATTENTION_SHIFT so surprise can trigger SURPRISED state.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any

from .base_agent import AgentMessage, BaseAgent, MessageType

log = logging.getLogger(__name__)

_FOCUS_DECAY_S   = 30.0   # seconds of silence before focus expires
_SUPPRESS_THRESH = 0.4    # GWT v5.0: salience < 0.4 → message ignored (not just suppressed)
_SHIFT_THRESH    = 0.8    # salience > this → bottom-up ATTENTION_SHIFT
_SPOTLIGHT_WINDOW = 30.0  # seconds — keep recent scored msgs for spotlight


class AttentionAgent(BaseAgent):
    """Posterior Parietal Cortex — GWT Gater and attentional spotlight.

    v5.0 changes:
      • Suppress threshold raised to 0.4 (GWT: messages below this are discarded)
      • Spotlight: rolling window of scored messages; get_spotlight(n) returns top-n
    """

    name = "attention_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        affective_bias_enabled: bool = True,
        affective_multiplier: float = 1.3,
    ):
        super().__init__(bus)
        self._focus_target: str = ""
        self._focus_start:  float = 0.0
        self._suppressed:   set[str] = set()
        self._baseline: dict[str, float] = {}
        self._scored_window: collections.deque = collections.deque(maxlen=200)
        self._current_emotion: str = "NEUTRAL"
        self._affective_bias_enabled = affective_bias_enabled
        self._affective_multiplier = affective_multiplier

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("AttentionAgent: started — focus decay %.0fs, suppress_thresh=%.2f",
                 _FOCUS_DECAY_S, _SUPPRESS_THRESH)
        asyncio.create_task(self._decay_loop(), name="attention-decay")
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        await super().stop()

    # ── Message handling ─────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        mtype = message.type

        if mtype == MessageType.COGNITION_INTENT:
            # Top-down: lock onto the current topic
            topic = (message.data.get("intent") or message.data.get("topic") or "")
            if topic and topic != self._focus_target:
                self._focus_target = topic
                self._focus_start  = time.time()
                self._suppressed.clear()
                await self._publish_focus(topic, reason="intent_lock")

        elif mtype in (
            MessageType.PERCEPTION_VISION,
            MessageType.PERCEPTION_SPEECH,
            MessageType.PERCEPTION_SENSOR,
            MessageType.WORLD_SURPRISE,
        ):
            score = self._score_salience(message)
            # WORLD_SURPRISE always gets maximum salience
            if mtype == MessageType.WORLD_SURPRISE:
                score = 1.0
            source = message.source

            # Record in spotlight window
            self._scored_window.append({
                "ts":       time.time(),
                "salience": round(score, 3),
                "type":     mtype.value,
                "source":   source,
                "summary":  str(message.data.get("text") or message.data.get("scene") or "")[:80],
            })

            # Update rolling baseline (exponential moving average)
            prev = self._baseline.get(source, score)
            self._baseline[source] = 0.8 * prev + 0.2 * score

            # GWT gate: discard if below threshold (not just suppress)
            if score < _SUPPRESS_THRESH and self._focus_target:
                self._suppressed.add(source)
                return None   # gated out — do not further process

            if score >= _SHIFT_THRESH:
                # Bottom-up surprise: override current focus
                topic = message.data.get("text") or message.data.get("objects") or source
                await self._publish_shift(source, score, topic)

        elif mtype == MessageType.EMOTION_CHANGE:
            self._current_emotion = message.data.get("emotion", "NEUTRAL").upper()

        elif mtype == MessageType.BEHAVIOR_CHANGE:
            state = message.data.get("state", "")
            if state == "IDLE":
                self._clear_focus()

        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _score_salience(self, message: AgentMessage) -> float:
        """Heuristic salience: 0.0 – 1.0."""
        score = 0.5  # neutral baseline

        # Priority field (1=high → 10=low)
        prio = message.priority
        score += (5 - prio) * 0.05          # range ±0.25

        # Speech: length is a proxy for content richness
        text = message.data.get("text", "")
        if text:
            words = len(text.split())
            score += min(words / 20.0, 0.2)  # up to +0.2 for 20+ words

        # Motion detected → inherently salient
        if message.data.get("motion_detected"):
            score += 0.25

        # Wake-word fired → maximum salience
        if message.data.get("wake_word"):
            score = 1.0

        # Deviation from recent baseline for this source
        baseline = self._baseline.get(message.source, 0.5)
        deviation = abs(score - baseline)
        score += deviation * 0.1

        # Affective bias: heightened arousal makes perception events more salient.
        if self._affective_bias_enabled and self._current_emotion in ("SURPRISED", "FRUSTRATED"):
            if message.type in (MessageType.PERCEPTION_VISION, MessageType.PERCEPTION_SPEECH,
                                MessageType.PERCEPTION_SENSOR):
                score *= self._affective_multiplier

        return max(0.0, min(1.0, score))

    def get_spotlight(self, n: int = 3) -> list[dict]:
        """Return top-n most salient recent entries (GWT Spotlight).
        Filters out entries older than _SPOTLIGHT_WINDOW seconds."""
        now = time.time()
        recent = [e for e in self._scored_window if now - e["ts"] < _SPOTLIGHT_WINDOW]
        return sorted(recent, key=lambda e: e["salience"], reverse=True)[:n]

    def _clear_focus(self) -> None:
        self._focus_target = ""
        self._focus_start  = 0.0
        self._suppressed.clear()

    async def _decay_loop(self) -> None:
        """Expire focus after _FOCUS_DECAY_S seconds of inactivity."""
        while self._running:
            await asyncio.sleep(5)
            if self._focus_target and (time.time() - self._focus_start) > _FOCUS_DECAY_S:
                log.debug("AttentionAgent: focus '%s' expired", self._focus_target)
                self._clear_focus()

    async def _publish_focus(self, topic: str, reason: str) -> None:
        await self.publish(AgentMessage(
            type=MessageType.ATTENTION_FOCUS,
            source=self.name,
            data={
                "focus_target": topic,
                "reason":       reason,
                "suppressed":   list(self._suppressed),
                "ts":           time.time(),
            },
            priority=3,
        ))
        log.debug("AttentionAgent: FOCUS → '%s' (%s)", topic, reason)

    async def _publish_shift(self, source: str, score: float, topic: Any) -> None:
        await self.publish(AgentMessage(
            type=MessageType.ATTENTION_SHIFT,
            source=self.name,
            data={
                "shift_from":    self._focus_target,
                "shift_to":      str(topic)[:120],
                "trigger_source": source,
                "salience":      round(score, 3),
                "ts":            time.time(),
            },
            priority=2,
        ))
        # Bottom-up shifts reset top-down focus
        self._focus_target = str(topic)[:120]
        self._focus_start  = time.time()
        log.debug("AttentionAgent: SHIFT → salience=%.2f from %s", score, source)
