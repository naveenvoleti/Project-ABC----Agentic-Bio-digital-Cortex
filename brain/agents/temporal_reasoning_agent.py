"""TemporalReasoningAgent — Basal Ganglia + Cerebellum (cognitive) mapping.

Two responsibilities:
  1. Time-of-day pattern learning — tracks when interactions happen and
     predicts the user's active windows.
  2. Causal chain reasoning — detects cause-effect language in user speech
     and uses the LLM to extend the chain (local model, max_tokens=100).

Publishes TEMPORAL_INSIGHT → CognitionAgent (injected into system prompt).
"""
from __future__ import annotations

import asyncio
import collections
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

from .base_agent import AgentMessage, BaseAgent, MessageType

if TYPE_CHECKING:
    from brain.llm.llm_router import LLMRouter
    from brain.memory.episodic_memory import EpisodicMemory

log = logging.getLogger(__name__)

_CAUSAL_PATTERN = re.compile(
    r'\b(because|therefore|so that|caused by|leads? to|results? in|'
    r'due to|as a result|consequently|thus|hence|triggers?|makes?)\b',
    re.IGNORECASE,
)
_PATTERN_UPDATE_INTERVAL = 60.0   # seconds between pattern re-analysis
_PATTERN_WINDOW_HOURS    = 24 * 7  # 7 days of hourly buckets


class TemporalReasoningAgent(BaseAgent):
    """Basal Ganglia + Cerebellum — time awareness and causality."""

    name = "temporal_reasoning_agent"

    def __init__(self, bus: asyncio.Queue, llm: "LLMRouter",
                 episodic: "EpisodicMemory"):
        super().__init__(bus)
        self._llm     = llm
        self._episodic = episodic
        # hour-of-day interaction frequency map (0-23)
        self._hour_counts: dict[int, int] = collections.defaultdict(int)
        self._last_pattern_ts: float = 0.0
        self._last_insight: str = ""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("TemporalReasoningAgent: started")
        asyncio.create_task(self._pattern_loop(), name="temporal-pattern")
        while self._running:
            await asyncio.sleep(1)

    # ── Message handling ─────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        if message.type == MessageType.PERCEPTION_SPEECH:
            # Record time-of-interaction
            self._hour_counts[datetime.now().hour] += 1

            # Check for causal language → LLM chain extension
            text = message.data.get("text", "")
            if text and _CAUSAL_PATTERN.search(text):
                asyncio.create_task(
                    self._build_causal_chain(text),
                    name="temporal-causal",
                )

        elif message.type == MessageType.CURIOSITY_TRIGGER:
            # Inject time-of-day awareness into curiosity cycle
            await self._publish_insight(causal=None)

        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _pattern_loop(self) -> None:
        while self._running:
            await asyncio.sleep(_PATTERN_UPDATE_INTERVAL)
            await self._publish_insight(causal=None)
            # V10.0 — Check for temporal anomalies once per minute
            await self._check_temporal_anomaly()

    async def _check_temporal_anomaly(self) -> None:
        """V10.0 — Detect surprising temporal patterns and fire WORLD_SURPRISE.

        If the user typically interacts at hour H, but today it's 2+ hours off,
        that's a temporal anomaly worth flagging as a surprise.
        """
        if len(self._hour_counts) < 7:
            return   # not enough data to establish patterns

        now_hour = datetime.now().hour
        total_counts = sum(self._hour_counts.values())
        if total_counts < 10:
            return

        # Compute expected activity probability for current hour
        max_count = max(self._hour_counts.values())
        current_hour_count = self._hour_counts.get(now_hour, 0)
        # Fraction of peak activity (0.0 = this hour is never active)
        activity_fraction = current_hour_count / max_count if max_count > 0 else 0.0

        # If we're receiving input at an hour that's historically very quiet (< 10% of peak)
        # AND the user just spoke (recent message), that's temporally surprising
        if activity_fraction < 0.10 and self._hour_counts.get(now_hour, 0) == 0:
            import time as _t
            now = _t.time()
            # Only fire once per 30 min to avoid spam
            if not hasattr(self, "_last_temporal_surprise_ts"):
                self._last_temporal_surprise_ts = 0.0
            if (now - self._last_temporal_surprise_ts) > 1800:
                self._last_temporal_surprise_ts = now
                peak_hours = sorted(self._hour_counts, key=self._hour_counts.get, reverse=True)[:2]
                detail = (
                    f"User active at {now_hour:02d}:xx — "
                    f"normally active around {', '.join(f'{h:02d}:xx' for h in peak_hours)}"
                )
                log.info("TemporalReasoningAgent: TEMPORAL SURPRISE — %s", detail)
                await self.publish(AgentMessage(
                    type=MessageType.WORLD_SURPRISE,
                    source=self.name,
                    data={
                        "source":          "temporal",
                        "type":            "temporal_anomaly",
                        "detail":          detail,
                        "current_hour":    now_hour,
                        "peak_hours":      peak_hours,
                        "activity_frac":   round(activity_fraction, 2),
                        "delta":           0.6,
                    },
                    priority=5,
                ))

    async def _build_causal_chain(self, text: str) -> None:
        try:
            prompt = (
                f"The user said: \"{text[:300]}\"\n"
                "Identify the cause-effect relationship in one sentence, "
                "then extend the chain by predicting one likely next consequence. "
                "Be concise (max 2 sentences)."
            )
            result = await self._llm.infer(
                user_message=prompt,
                system_prompt="You are a causal reasoning assistant. Be brief.",
                max_tokens=100,
            )
            if result:
                await self._publish_insight(causal=result.strip())
        except Exception as exc:
            log.debug("TemporalReasoningAgent: causal chain failed — %s", exc)

    def _summarise_patterns(self) -> str:
        if not self._hour_counts:
            return ""
        top_hours = sorted(self._hour_counts, key=self._hour_counts.get, reverse=True)[:3]
        now_hour  = datetime.now().hour
        time_label = (
            "morning" if 5 <= now_hour < 12 else
            "afternoon" if 12 <= now_hour < 17 else
            "evening" if 17 <= now_hour < 21 else
            "night"
        )
        peak_str = ", ".join(f"{h:02d}:00" for h in top_hours)
        return (
            f"Current time: {time_label} ({now_hour:02d}:xx). "
            f"User is typically most active around {peak_str}."
        )

    async def _publish_insight(self, causal: str | None) -> None:
        pattern_summary = self._summarise_patterns()
        parts = [p for p in (pattern_summary, causal) if p]
        if not parts:
            return
        insight = " | ".join(parts)
        if insight == self._last_insight:
            return   # no change — skip publish
        self._last_insight = insight
        await self.publish(AgentMessage(
            type=MessageType.TEMPORAL_INSIGHT,
            source=self.name,
            data={
                "summary":         insight,
                "hour_counts":     dict(self._hour_counts),
                "causal_chain":    causal or "",
                "ts":              time.time(),
            },
            priority=7,
        ))
        log.debug("TemporalReasoningAgent: insight published")

