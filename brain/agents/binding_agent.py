"""
BindingAgent — Global Workspace Theory Binding Layer. (V10.0)

The "hard problem" bridge: synthesizes all perceptual, emotional, and
cognitive streams into a single unified "present moment" string that
is injected into EVERY LLM system prompt.

This is what prevents the brain from answering questions about "the scene"
while unaware of its own emotional state, or describing emotions without
grounding them in what it actually sees and hears.

The binding moment contains:
  1. Scene (from WorldModel)
  2. Emotion + Valence/Arousal (from EmotionEngine)
  3. Body pose (from ProprioceptionAgent)
  4. Drive state (from IntrinsicMotivationAgent)
  5. Active goal (from GoalStackAgent, if any)
  6. Surprising event (from AttentionAgent WORLD_SURPRISE, if recent)

Published: BINDING_UPDATE → CognitionAgent every 2s
Priority: very low (8) — background integration layer

CognitionAgent prepends this string to system_prompt as:
  "## My Present Moment\n{binding_string}\n\n"
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from .base_agent import AgentMessage, BaseAgent, MessageType

if TYPE_CHECKING:
    from brain.memory.world_model import WorldModel

log = logging.getLogger(__name__)

_PUBLISH_INTERVAL_S = 2.0      # publish binding every 2s
_SURPRISE_TTL_S     = 15.0     # keep surprise in binding for 15s


class BindingAgent(BaseAgent):
    """GWT Binding — unified phenomenal present-moment string.

    Synthesizes all perceptual and affective streams into a single coherent
    statement of "what is happening right now" for every LLM call.
    Without this, the LLM might respond without integrating all modalities.
    """

    name = "binding_agent"

    def __init__(self, bus: asyncio.Queue, world_model: "WorldModel | None" = None):
        super().__init__(bus)
        self._world_model = world_model

        # Perceptual state (refreshed from bus messages)
        self._scene:         str   = ""
        self._audio:         str   = ""
        self._user_present:  bool  = False
        self._user_emotion:  str   = ""

        # Affective state
        self._emotion:       str   = "NEUTRAL"
        self._valence:       float = 0.0
        self._arousal:       float = 0.15

        # Body state
        self._pan_deg:       float = 0.0
        self._tilt_deg:      float = 0.0

        # Motivational state
        self._top_drive:     str   = ""
        self._drive_level:   float = 0.0

        # Autonomous goal state
        self._active_goal:   str   = ""

        # Surprise event (TTL-gated)
        self._last_surprise: str   = ""
        self._surprise_ts:   float = 0.0

        # Last published binding (avoid redundant publishes)
        self._last_binding:  str   = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("BindingAgent: started — phenomenal binding every %.1fs", _PUBLISH_INTERVAL_S)
        asyncio.create_task(self._binding_loop(), name="binding-loop")
        while self._running:
            await asyncio.sleep(1)

    # ── Message handling ──────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        mtype = message.type

        if mtype == MessageType.EMOTION_CHANGE:
            self._emotion  = message.data.get("to", "NEUTRAL")
            self._valence  = float(message.data.get("valence", 0.0))
            self._arousal  = float(message.data.get("arousal", 0.15))

        elif mtype == MessageType.WORLD_SURPRISE:
            self._last_surprise = message.data.get("detail", "something unexpected happened")
            self._surprise_ts   = time.time()

        elif mtype == MessageType.COGNITION_THOUGHT:
            # Use internal monologue as scene hint (if we have no VLM scene yet)
            thought = message.data.get("thought", "")
            if thought and not self._scene:
                self._scene = thought[:200]

        elif mtype == MessageType.PROPRIOCEPTION_STATE:
            self._pan_deg  = float(message.data.get("pan_deg",  0.0))
            self._tilt_deg = float(message.data.get("tilt_deg", 0.0))

        elif mtype == MessageType.MOTIVATION_DRIVE:
            drives = message.data.get("drives", {})
            if drives:
                top = max(drives, key=drives.get)
                self._top_drive  = top
                self._drive_level = drives[top]

        elif mtype == MessageType.GOAL_PUSH:
            goal = message.data.get("goal", {})
            self._active_goal = goal.get("message", "")[:100]

        elif mtype == MessageType.GOAL_COMPLETE:
            self._active_goal = ""

        elif mtype == MessageType.PERSON_ENTER:
            self._user_present = True

        elif mtype == MessageType.PERSON_LEAVE:
            self._user_present = False

        elif mtype == MessageType.PERCEPTION_SPEECH:
            # Infer presence from voice
            self._user_present = True

        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _binding_loop(self) -> None:
        """Publish unified binding string every PUBLISH_INTERVAL_S."""
        while self._running:
            await asyncio.sleep(_PUBLISH_INTERVAL_S)
            await self._publish_binding()

    async def _build_binding_string(self) -> str:
        """Synthesize all active streams into one coherent present-moment string."""
        parts: list[str] = []

        # 1. Perceptual grounding (what I sense right now)
        if self._world_model:
            try:
                snap = await self._world_model.snapshot()
                scene = snap.get("scene", "")
                audio = snap.get("audio", "")
                self._user_emotion = snap.get("user_emotion", "")
                present = snap.get("present_entities", [])
                if scene:
                    self._scene = scene[:150]
                if audio:
                    self._audio = audio[:80]
                self._user_present = bool(present)
            except Exception:
                pass

        if self._scene:
            parts.append(f"I see: {self._scene[:120]}")
        if self._audio:
            parts.append(f"I hear: {self._audio[:70]}")
        if self._user_present:
            person_str = "Someone is present"
            if self._user_emotion:
                person_str += f" (they seem {self._user_emotion.lower()})"
            parts.append(person_str)

        # 2. Affective state
        emotion_desc = self._emotion.lower()
        va_desc = ""
        if abs(self._valence) > 0.15 or self._arousal > 0.4:
            v_word = ("positive" if self._valence > 0.2 else
                      "negative" if self._valence < -0.2 else "neutral")
            a_word = "energized" if self._arousal > 0.6 else "calm" if self._arousal < 0.3 else "moderate"
            va_desc = f" (valence={v_word}, energy={a_word})"
        parts.append(f"I feel: {emotion_desc}{va_desc}")

        # 3. Body orientation
        if abs(self._pan_deg) > 5 or abs(self._tilt_deg) > 5:
            dir_h = ("right" if self._pan_deg > 0 else "left")
            dir_v = ("up" if self._tilt_deg > 0 else "down")
            parts.append(
                f"Looking: {abs(self._pan_deg):.0f}° {dir_h}, "
                f"{abs(self._tilt_deg):.0f}° {dir_v}"
            )

        # 4. Motivational drive
        if self._top_drive and self._drive_level > 0.4:
            parts.append(f"I want to: {self._top_drive} ({self._drive_level:.0%})")

        # 5. Active autonomous goal
        if self._active_goal:
            parts.append(f"Current goal: {self._active_goal}")

        # 6. Recent surprise (TTL-gated)
        surprise_age = time.time() - self._surprise_ts
        if self._last_surprise and surprise_age < _SURPRISE_TTL_S:
            parts.append(f"Something unexpected: {self._last_surprise[:80]}")

        if not parts:
            return ""

        return " | ".join(parts)

    async def _publish_binding(self) -> None:
        """Build and publish the unified binding moment."""
        try:
            binding = await self._build_binding_string()
            if not binding:
                return
            if binding == self._last_binding:
                return   # no change — skip
            self._last_binding = binding
            await self.publish(AgentMessage(
                type=MessageType.BINDING_UPDATE,
                source=self.name,
                data={
                    "binding_moment": binding,
                    "emotion":        self._emotion,
                    "valence":        self._valence,
                    "arousal":        self._arousal,
                    "ts":             time.time(),
                },
                priority=8,  # low priority — background integration
            ))
            log.debug("BindingAgent: '%s...'", binding[:80])
        except Exception as e:
            log.debug("BindingAgent: publish error: %s", e)
