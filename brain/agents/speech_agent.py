"""
SpeechAgent — TTS synthesis and audio output.
Maps to Broca's Area output. Manages speaker queue and BT/USB/AUX selection.

Human-like enhancements:
  - Emotion-aware speech speed: excited → faster, sad → slower
  - Volume matching: louder user → louder Brain response
  - Interrupted speech acknowledgement: "Go ahead." after barge-in
"""
from __future__ import annotations

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.hardware.audio_driver import AudioDriver
from brain.utils.logger import get_logger

log = get_logger(__name__)

# Emotion → Kokoro speed multiplier.  Neutral = 1.0.
_EMOTION_SPEED: dict[str, float] = {
    "EXCITED":       1.15,
    "HAPPY":         1.10,
    "CURIOUS":       1.05,
    "NEUTRAL":       1.00,
    "CALM":          1.00,
    "CONFUSED":      0.95,
    "FRUSTRATED":    0.90,
    "ANGRY":         0.92,
    "SAD":           0.85,
    "FEARFUL":       0.90,
}

# Phrases spoken after Brain's TTS is interrupted by user barge-in
_INTERRUPT_ACKS = [
    "Go ahead.",
    "Yes?",
    "I'm listening.",
    "Sorry — go ahead.",
    "Yeah?",
]


class SpeechAgent(BaseAgent):
    name = "speech_agent"

    def __init__(self, bus: asyncio.Queue, audio: AudioDriver):
        super().__init__(bus)
        self._audio = audio
        # Queue items: (text, volume, speed)
        self._tts_queue: asyncio.Queue[tuple] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._speaking = False
        self._volume = 80
        self._current_emotion: str = "NEUTRAL"
        # Rolling user RMS for volume matching (0.0–0.1 typical range)
        self._user_rms: float = 0.03   # neutral default

        # V10.0 — Track continuous arousal for TTS speed modulation
        self._current_arousal: float = 0.15   # synced from EMOTION_CHANGE.data["arousal"]

    async def start(self) -> None:
        await super().start()
        log.info("SpeechAgent started")
        asyncio.create_task(self._speak_loop())
        while self._running:
            await asyncio.sleep(1)

    async def _speak_loop(self) -> None:
        while self._running:
            try:
                item = await asyncio.wait_for(self._tts_queue.get(), timeout=1.0)
                text, volume, speed = item
                self._speaking = True
                log.info(f"Speaking: '{text[:80]}' speed={speed:.2f}")
                await self._audio.speak_async(text, volume, speed)
                self._speaking = False
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Speech error: {e}")
                self._speaking = False

    def _compute_speed(self) -> float:
        """V10.0 — Blend discrete emotion table with continuous arousal substrate.

        Base speed from emotion table, modulated by live arousal:
        arousal 0.0 = calm  → no extra speed
        arousal 1.0 = peak  → +0.25 extra speed
        This creates a smooth, continuously graded TTS pace.
        """
        base = _EMOTION_SPEED.get(self._current_emotion.upper(), 1.0)
        # Blend: 70% emotion table + 30% arousal modulation
        arousal_bonus = self._current_arousal * 0.25   # 0.0 to +0.25
        return round(min(1.45, max(0.75, base + arousal_bonus * 0.3)), 3)


    def _compute_volume(self) -> int:
        """Map user RMS loudness to response volume.
        rms ~0.01 (quiet) → vol 65, rms ~0.05 (normal) → 80, rms ~0.09 (loud) → 95."""
        vol = int(60 + (self._user_rms / 0.08) * 35)
        return max(60, min(95, vol))

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.EMOTION_CHANGE:
            self._current_emotion = message.data.get("to", "NEUTRAL")
            # V10.0 — Sync continuous arousal for TTS speed modulation
            if "arousal" in message.data:
                self._current_arousal = float(message.data["arousal"])

        elif message.type == MessageType.ACTION_SPEAK:
            text = message.data.get("text", "")
            if text:
                # Clear queue if interrupt requested
                if message.data.get("interrupt"):
                    while not self._tts_queue.empty():
                        self._tts_queue.get_nowait()
                speed = self._compute_speed()
                volume = self._compute_volume()
                await self._tts_queue.put((text, volume, speed))

        elif message.type == MessageType.PERCEPTION_SPEECH:
            # Track user RMS for volume matching (F11)
            rms = message.data.get("voice_rms")
            if rms and isinstance(rms, (int, float)) and rms > 0:
                # Exponential moving average — smooth out spikes
                self._user_rms = 0.7 * self._user_rms + 0.3 * float(rms)

            # Stop TTS whenever new user input arrives (voice barge-in OR chat message)
            was_speaking = self._speaking
            if self._speaking or message.data.get("interrupt"):
                self._audio.stop_speaking()
                while not self._tts_queue.empty():
                    try:
                        self._tts_queue.get_nowait()
                    except Exception:
                        break
                self._speaking = False
                if was_speaking:
                    # Only log + discard context when speech was actually mid-sentence.
                    # If speech had already finished (was_speaking=False), the assistant
                    # response IS valid context — discarding it causes the LLM to
                    # re-generate the same response for the next query.
                    log.info("SpeechAgent: TTS interrupted by user input")
                    await self.publish(AgentMessage(
                        type=MessageType.MEMORY_WRITE,
                        source=self.name,
                        data={"discard_last_assistant": True},
                    ))
                    # F9 — Interrupted speech acknowledgement: brief "Go ahead."
                    ack = random.choice(_INTERRUPT_ACKS)
                    await asyncio.sleep(0.25)
                    await self._tts_queue.put((ack, self._compute_volume(), 1.0))

        elif message.type == MessageType.BEHAVIOR_CHANGE:
            state = message.data.get("state", "")
            if state == "BORED":
                await self._tts_queue.put(("I wonder what's happening out there...",
                                           self._compute_volume(), self._compute_speed()))
            elif state == "CURIOUS":
                await self._tts_queue.put(("Let me look around a bit.",
                                           self._compute_volume(), self._compute_speed()))

    async def stop(self) -> None:
        await super().stop()
        self._executor.shutdown(wait=False)
