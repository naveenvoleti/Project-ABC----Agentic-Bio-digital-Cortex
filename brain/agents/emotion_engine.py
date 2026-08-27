"""
EmotionEngine — 10-state affective FSM + Continuous Valence/Arousal substrate (V10.0).

Emits EMOTION_CHANGE messages on transitions.

V10.0 addition — Valence/Arousal continuum:
  Runs parallel to the FSM. Drifts smoothly toward the FSM state's
  target (valence, arousal) coordinates over 2-5 seconds.
  This produces natural emotional ramps instead of binary state snaps.
  Mixed states (e.g., anxiously curious = neutral valence + high arousal)
  emerge naturally from the drift dynamics.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum

import yaml

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.utils.logger import get_logger

log = get_logger(__name__)


class Emotion(str, Enum):
    NEUTRAL    = "NEUTRAL"
    HAPPY      = "HAPPY"
    ATTENTIVE  = "ATTENTIVE"
    CURIOUS    = "CURIOUS"
    BORED      = "BORED"
    FOCUSED    = "FOCUSED"
    CONFUSED   = "CONFUSED"
    FRUSTRATED = "FRUSTRATED"
    SURPRISED  = "SURPRISED"
    EXCITED    = "EXCITED"


# Timed auto-transitions: (current_state, seconds) → next_state
TIMED_TRANSITIONS: dict[str, tuple[int, str]] = {
    "HAPPY":     (10, "NEUTRAL"),
    "SURPRISED": (8,  "ATTENTIVE"),
    "EXCITED":   (15, "HAPPY"),
    "FRUSTRATED": (30, "NEUTRAL"),
}

# Maps voice_tone string → internal event name
TONE_TO_EVENT: dict[str, str] = {
    "frustrated": "user_frustrated",
    "sad":        "user_sad",
    "excited":    "user_excited",
    "happy":      "positive_sentiment",
    "calm":       "",   # no transition — calm is baseline
}

# Event-driven transitions: (current_state, event) → next_state
EVENT_TRANSITIONS: dict[tuple[str, str], str] = {
    ("NEUTRAL",    "voice_detected"):        "ATTENTIVE",
    ("NEUTRAL",    "motion_detected"):       "ATTENTIVE",
    ("NEUTRAL",    "idle_timeout"):          "BORED",
    ("NEUTRAL",    "task_success"):          "HAPPY",
    ("NEUTRAL",    "positive_sentiment"):    "HAPPY",
    ("ATTENTIVE",  "complex_query"):         "FOCUSED",
    ("ATTENTIVE",  "positive_sentiment"):    "HAPPY",
    ("ATTENTIVE",  "greeting"):              "HAPPY",
    ("ATTENTIVE",  "input_ends"):            "NEUTRAL",
    ("ATTENTIVE",  "unexpected_event"):      "SURPRISED",
    ("FOCUSED",    "inference_failure_3x"):  "FRUSTRATED",
    ("FOCUSED",    "task_success"):          "HAPPY",
    ("FOCUSED",    "ambiguous_input"):       "CONFUSED",
    ("FOCUSED",    "interesting_discovery"): "EXCITED",
    ("CONFUSED",   "clarification_received"):"FOCUSED",
    ("CONFUSED",   "no_clarification_30s"):  "FRUSTRATED",
    ("BORED",      "curiosity_trigger"):     "CURIOUS",
    ("BORED",      "any_input"):             "ATTENTIVE",
    ("CURIOUS",    "novel_stimulus"):        "ATTENTIVE",
    ("CURIOUS",    "exploration_timeout"):   "NEUTRAL",
    # ── Voice tone empathy transitions ────────────────────────────────────────
    # user_frustrated: Brain shifts to careful/focused mode to help better
    ("NEUTRAL",    "user_frustrated"):       "FOCUSED",
    ("ATTENTIVE",  "user_frustrated"):       "FOCUSED",
    ("HAPPY",      "user_frustrated"):       "FOCUSED",
    ("FOCUSED",    "user_frustrated"):       "CONFUSED",   # already focused, still frustrated = unsure how to help
    # user_sad: Brain becomes attentive and caring
    ("NEUTRAL",    "user_sad"):              "ATTENTIVE",
    ("ATTENTIVE",  "user_sad"):              "FOCUSED",
    ("BORED",      "user_sad"):              "ATTENTIVE",
    # user_excited: Brain mirrors excitement
    ("NEUTRAL",    "user_excited"):          "EXCITED",
    ("ATTENTIVE",  "user_excited"):          "EXCITED",
    ("HAPPY",      "user_excited"):          "EXCITED",
    ("FOCUSED",    "user_excited"):          "HAPPY",
    # ── Interoception (hardware-stress) transitions ───────────────────────────
    # thermal_stress: CPU too hot → conserving/careful mode
    ("NEUTRAL",    "thermal_stress"):       "FOCUSED",
    ("ATTENTIVE",  "thermal_stress"):       "FOCUSED",
    ("HAPPY",      "thermal_stress"):       "FOCUSED",
    ("BORED",      "thermal_stress"):       "FOCUSED",
    # overwhelmed: CPU/RAM saturated → frustrated
    ("FOCUSED",    "hw_overwhelmed"):       "FRUSTRATED",
    ("NEUTRAL",    "hw_overwhelmed"):       "FRUSTRATED",
    # ── V10.0 — World surprise (audio spike, visual anomaly, temporal anomaly) ──
    ("NEUTRAL",    "world_surprise"):       "SURPRISED",
    ("ATTENTIVE",  "world_surprise"):       "SURPRISED",
    ("HAPPY",      "world_surprise"):       "SURPRISED",
    ("FOCUSED",    "world_surprise"):       "SURPRISED",
    ("BORED",      "world_surprise"):       "ATTENTIVE",
    ("CURIOUS",    "world_surprise"):       "EXCITED",
    ("SURPRISED",  "world_surprise"):       "SURPRISED",   # refresh if already surprised
}

# ── V10.0 — Valence/Arousal target coordinates per FSM state ──────────────────
# Valence: -1.0 (very negative) to +1.0 (very positive / pleasant)
# Arousal:  0.0 (very calm)     to +1.0 (very excited / activated)
_STATE_VA: dict[str, tuple[float, float]] = {
    "NEUTRAL":    ( 0.0,  0.15),
    "HAPPY":      ( 0.8,  0.50),
    "ATTENTIVE":  ( 0.1,  0.60),
    "CURIOUS":    ( 0.3,  0.65),
    "BORED":      (-0.2, -0.10),
    "FOCUSED":    ( 0.2,  0.45),
    "CONFUSED":   (-0.1,  0.35),
    "FRUSTRATED": (-0.7,  0.60),
    "SURPRISED":  ( 0.1,  0.90),
    "EXCITED":    ( 0.7,  0.90),
}
_VA_DRIFT_RATE = 0.08   # fraction of gap closed per 0.5s tick


class EmotionEngine(BaseAgent):
    name = "emotion_engine"


    def __init__(self, bus: asyncio.Queue, emotions_config_path: str = "config/emotions.yaml"):
        super().__init__(bus)
        self._state = Emotion.NEUTRAL
        self._state_since = time.time()
        self._failure_count = 0
        self._emotions_cfg: dict = {}
        self._load_config(emotions_config_path)
        # V10.0 — Continuous valence/arousal substrate (initialized to NEUTRAL target)
        self._valence: float = 0.0    # -1.0 (negative) to +1.0 (positive)
        self._arousal: float = 0.15   # 0.0 (calm) to 1.0 (excited)


    def _load_config(self, path: str) -> None:
        try:
            with open(path) as f:
                self._emotions_cfg = yaml.safe_load(f) or {}
        except Exception:
            pass

    # V10.0 — Valence/Arousal properties (read by BindingAgent, SpeechAgent)
    @property
    def valence(self) -> float:
        """Current hedonic valence: -1.0 (very negative) to +1.0 (very positive)."""
        return round(self._valence, 3)

    @property
    def arousal(self) -> float:
        """Current arousal level: 0.0 (very calm) to 1.0 (very excited)."""
        return round(self._arousal, 3)

    @property
    def current_emotion(self) -> str:
        return self._state.value

    @property
    def current_gif(self) -> str:
        em = self._emotions_cfg.get("emotions", {}).get(self._state.value, {})
        return em.get("gif", f"{self._state.value.lower()}.gif")

    @property
    def current_color(self) -> tuple[int, int, int]:
        em = self._emotions_cfg.get("emotions", {}).get(self._state.value, {})
        c = em.get("color", [30, 30, 50])
        return tuple(c)

    @property
    def motor_action(self) -> str:
        em = self._emotions_cfg.get("emotions", {}).get(self._state.value, {})
        return em.get("motor_action", "still")

    async def trigger(self, event: str) -> None:
        key = (self._state.value, event)
        next_state = EVENT_TRANSITIONS.get(key)
        if next_state:
            await self._transition(Emotion(next_state))

    async def _transition(self, new_state: Emotion) -> None:
        if new_state == self._state:
            return
        old = self._state.value
        self._state = new_state
        self._state_since = time.time()
        log.info(f"Emotion: {old} → {new_state.value} (VA target: {_STATE_VA.get(new_state.value, (0,0))})")
        await self.publish(AgentMessage(
            type=MessageType.EMOTION_CHANGE,
            source=self.name,
            target="display_agent",
            data={
                "from": old,
                "to": new_state.value,
                "gif": self.current_gif,
                "color": list(self.current_color),
                "motor_action": self.motor_action,
                # V10.0 — Include current VA readings for SpeechAgent / BindingAgent
                "valence": self._valence,
                "arousal": self._arousal,
            },
            priority=3,
        ))

    async def start(self) -> None:
        await super().start()
        log.info("EmotionEngine started (VA substrate active: drift_rate=%.2f)", _VA_DRIFT_RATE)
        # V10.0 — Launch VA drift loop alongside the timed-transition loop
        asyncio.create_task(self._va_drift_loop(), name="emotion-va-drift")
        while self._running:
            await asyncio.sleep(1)
            await self._check_timed_transitions()

    async def _va_drift_loop(self) -> None:
        """V10.0 — Drift valence/arousal toward current FSM state target.

        Ticks every 0.5s. Moves (_valence, _arousal) by _VA_DRIFT_RATE fraction
        of the remaining gap. This creates exponential approach (never snaps):
        the brain 'feels' its way into a new emotional state rather than jumping.
        """
        while self._running:
            await asyncio.sleep(0.5)
            target_v, target_a = _STATE_VA.get(self._state.value, (0.0, 0.15))
            # Exponential drift: new = current + rate * (target - current)
            self._valence += _VA_DRIFT_RATE * (target_v - self._valence)
            self._arousal += _VA_DRIFT_RATE * (target_a - self._arousal)
            # Clamp to valid ranges
            self._valence = max(-1.0, min(1.0, self._valence))
            self._arousal = max(0.0,  min(1.0, self._arousal))

    async def _check_timed_transitions(self) -> None:
        timed = TIMED_TRANSITIONS.get(self._state.value)
        if timed:
            seconds, next_state = timed
            if time.time() - self._state_since >= seconds:
                await self._transition(Emotion(next_state))

    async def handle(self, message: AgentMessage) -> None:
        data = message.data
        if message.type == MessageType.SYSTEM_ERROR:
            self._failure_count += 1
            if self._failure_count >= 3:
                await self.trigger("inference_failure_3x")
                self._failure_count = 0

        elif message.type == MessageType.COGNITION_RESPONSE:
            if data.get("success"):
                await self.trigger("task_success")
                self._failure_count = 0

        elif message.type == MessageType.PERCEPTION_SPEECH:
            text = data.get("text", "").lower()
            sentiment = data.get("sentiment", "neutral")
            voice_tone = data.get("voice_tone", "calm")

            if data.get("is_wake_word"):
                await self.trigger("voice_detected")
            elif sentiment == "positive":
                await self.trigger("positive_sentiment")
            elif "hello" in text or "hi" in text:
                await self.trigger("greeting")

            # Voice tone empathy — mirrors user's emotional state
            tone_event = TONE_TO_EVENT.get(voice_tone, "")
            if tone_event:
                log.debug(f"EmotionEngine: voice_tone={voice_tone} → event={tone_event}")
                await self.trigger(tone_event)

        elif message.type == MessageType.CURIOSITY_TRIGGER:
            await self.trigger("curiosity_trigger")

        elif message.type == MessageType.INTERO_STATE:
            label = data.get("label", "")
            if label == "overwhelmed" or label == "memory_pressure":
                await self.trigger("hw_overwhelmed")
            elif label == "hot":
                await self.trigger("thermal_stress")
            elif label == "busy" and self._state.value == "BORED":
                await self.trigger("any_input")   # break boredom when system is active

        elif message.type == MessageType.BEHAVIOR_CHANGE:
            # Mood arc trigger from CognitionAgent (F5 — mirror neuron analogue)
            mood_trigger = data.get("trigger", "")
            if mood_trigger:
                log.debug(f"EmotionEngine: mood arc trigger='{mood_trigger}' arc={data.get('mood_arc', 0):.2f}")
                await self.trigger(mood_trigger)
                return   # mood arc takes precedence over behavior state

            # Bridge behavior FSM → emotion FSM
            # BehaviorAgent transitions drive primary emotion changes
            new_behavior = data.get("state", "")
            if new_behavior == "ATTENTIVE":
                await self.trigger("voice_detected")      # NEUTRAL → ATTENTIVE
            elif new_behavior == "FOCUSED":
                await self.trigger("complex_query")       # ATTENTIVE → FOCUSED
            elif new_behavior == "IDLE":
                await self.trigger("input_ends")          # ATTENTIVE → NEUTRAL
            elif new_behavior == "EXPLORING":
                await self.trigger("curiosity_trigger")   # BORED → CURIOUS

        # V10.0 — WORLD_SURPRISE: unexpected audio/vision/temporal events → SURPRISED emotion
        elif message.type == MessageType.WORLD_SURPRISE:
            source = data.get("source", "unknown")
            log.info("EmotionEngine: WORLD_SURPRISE from %s → triggering surprise", source)
            await self.trigger("world_surprise")

