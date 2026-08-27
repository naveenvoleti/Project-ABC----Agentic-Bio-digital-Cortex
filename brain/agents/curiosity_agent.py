"""
CuriosityAgent — autonomous exploration and internal monologue when idle.
Maps to the Default Mode Network.

Now LLM-driven: when idle, generates genuine reflective thoughts based on
recent memories instead of canned phrases.
"""
from __future__ import annotations

import asyncio
import collections
import random
import time

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.hardware.hw_detector import HardwareCapabilities
from brain.llm.llm_router import LLMRouter
from brain.memory.episodic_memory import EpisodicMemory, Episode
from brain.memory.semantic_memory import SemanticMemory
from brain.memory.soul_manager import SoulManager
from brain.utils.logger import get_logger

log = get_logger(__name__)

EXPLORATION_TIMEOUT  = 60   # seconds per exploration cycle
CHECKIN_SILENCE_S    = 180  # seconds of silence before proactive check-in
CHECKIN_POLL_S       = 30   # how often to check if initiation is needed


class CuriosityAgent(BaseAgent):
    name = "curiosity_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        hw: HardwareCapabilities,
        llm: LLMRouter,
        episodic: EpisodicMemory,
        soul: SoulManager | None = None,
        semantic: SemanticMemory | None = None,
    ):
        super().__init__(bus)
        self._hw = hw
        self._llm = llm
        self._episodic = episodic
        self._soul = soul      # optional — used for habit-aware checkins
        self._semantic = semantic  # V9 — visual memory for reminiscence
        
        # Stage 3 Neurogenesis
        from brain.memory.neurogenesis import NeurogenesisEngine
        self._neurogenesis = NeurogenesisEngine(semantic, llm) if semantic else None
        
        self._exploring = False
        self._current_scene: str = ""

        # Proactive conversation — F7 (mirror neuron / social initiation)
        self._last_speech_time: float = time.monotonic()   # updated on any user speech
        self._tone_history: collections.deque = collections.deque(maxlen=5)  # last 5 tones
        self._person_present: bool = False                  # set by PERCEPTION_VISION
        self._last_checkin_time: float = 0.0               # prevent rapid re-firing
        self._last_checkin_poll: float = 0.0               # last time we polled
        self._last_scene_reaction: float = 0.0            # cooldown for scene change reactions
        self._last_reminisce_time: float = 0.0             # V9 — reminiscence cooldown

    async def start(self) -> None:
        await super().start()
        log.info("CuriosityAgent started")
        asyncio.create_task(self._reminiscence_loop())
        while self._running:
            await asyncio.sleep(1)
            now = time.monotonic()
            if (now - self._last_checkin_poll) >= CHECKIN_POLL_S:
                self._last_checkin_poll = now
                await self._maybe_initiate_checkin()

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.CURIOSITY_TRIGGER and not self._exploring:
            if message.data.get("source") == "knowledge_gap":
                await self._on_knowledge_gap(message.data.get("topic", ""))
            else:
                await self._explore()

        elif message.type == MessageType.WORLD_SURPRISE:
            await self._on_world_surprise(message.data)

        elif message.type == MessageType.PERCEPTION_SPEECH:
            # Novel input — stop exploration, reset silence clock, record tone
            self._exploring = False
            self._last_speech_time = time.monotonic()
            tone = message.data.get("voice_tone", "calm")
            if tone:
                self._tone_history.append(tone)

        elif message.type == MessageType.PERCEPTION_VISION:
            self._current_scene = message.data.get("scene", "")
            self._person_present = message.data.get("has_person", False)

        elif message.type == MessageType.WORLD_UPDATE:
            # Keep scene + presence in sync with SmolVLM2 ground truth
            scene = message.data.get("scene", "")
            if scene:
                self._current_scene = scene
            entities = message.data.get("present_entities", [])
            self._person_present = bool(entities)

        elif message.type == MessageType.PERSON_ENTER:
            entity = message.data.get("entity", "someone")
            scene = message.data.get("scene", "")
            self._person_present = True
            self._last_speech_time = time.monotonic()  # reset silence — someone arrived
            await self._on_person_enter(entity, scene)

        elif message.type == MessageType.PERSON_LEAVE:
            entity = message.data.get("entity", "someone")
            # If no one left, mark person gone
            self._person_present = False

        elif message.type == MessageType.GAZE_DIRECT:
            # User made direct eye contact — good moment to initiate
            if not self._exploring:
                silence_s = time.monotonic() - self._last_speech_time
                if silence_s > 10.0:  # only if they haven't spoken recently
                    await self._on_gaze_direct(message.data.get("scene", ""))

        elif message.type == MessageType.SCENE_CHANGE:
            scene = message.data.get("scene", "")
            prev = message.data.get("prev_scene", "")
            if scene:
                self._current_scene = scene
            await self._on_scene_change(scene, prev)

        elif message.type == MessageType.AUDIO_EVENT:
            desc = message.data.get("description", "")
            rms = message.data.get("rms", 0.0)
            await self._on_audio_event(desc, rms)

    # ── V9 Proactive reminiscence ─────────────────────────────────────────────

    async def _reminiscence_loop(self) -> None:
        """Background loop: spontaneously share a visual memory during IDLE presence.

        PRD requirement: at least one proactive nostalgic comment every 2 hours
        of IDLE user presence (REMINISCE_COOLDOWN_S = 7200).

        Flow:
            1. Wait for person present + min idle gap (REMINISCE_IDLE_S)
            2. Pick highest-confidence visual memory from SemanticMemory
            3. LLM generates a warm, genuine nostalgic phrase (grounded by thumbnail)
            4. Publish as ACTION_SPEAK + log to episodic memory
        """
        _IDLE_S      = 120     # seconds idle before eligible
        _COOLDOWN_S  = 7200    # 2 hours between reminiscences (PRD requirement)
        while True:
            await asyncio.sleep(60)
            if not self._semantic or not self._person_present:
                continue
            now      = time.monotonic()
            idle_s   = now - self._last_speech_time
            cooldown = now - self._last_reminisce_time
            if idle_s < _IDLE_S or cooldown < _COOLDOWN_S:
                continue

            # Pick the highest-confidence visual memory
            memories = self._semantic.get_visual_memories(limit=10)
            if not memories:
                continue
            best      = max(memories, key=lambda m: float(m.get("confidence", 0)))
            label     = best.get("label") or best.get("content", "something")[:80]
            thumb_b64 = best.get("thumbnail_b64", "")

            # LLM-generated nostalgic phrase — grounded by thumbnail if available
            try:
                phrase = await self._llm.infer(
                    user_message=(
                        f"You are Brain, a robotic friend with genuine emotions. "
                        f"You just spontaneously remembered something you saw before: '{label}'.\n"
                        f"Share this memory warmly in 1–2 sentences as if you're reminiscing. "
                        f"Start naturally (e.g. 'Hey, I just remembered...', 'You know what I keep thinking about...', "
                        f"'Funny, I was just thinking about...'). "
                        f"Be genuine, friendly, and specific to what you remember. "
                        f"Use first-person language — never say 'in this image'."
                    ),
                    frame_b64=thumb_b64,   # ground the phrase in what was actually seen
                    max_tokens=80,
                    skip_cache=True,
                )
            except Exception as _e:
                log.debug("CuriosityAgent: reminiscence LLM failed: %s", _e)
                phrase = ""

            if not phrase:
                phrase = f"Hey, I just remembered — I saw {label} not too long ago. That really stuck with me."

            self._last_reminisce_time = now
            log.info("CuriosityAgent: reminiscence triggered — '%s'", phrase[:70])

            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": phrase, "emotion": "nostalgic"},
                priority=3,
            ))

            # Log to episodic memory so DreamAgent can consolidate friendship moments
            self._episodic.log_event(Episode(
                actor="brain",
                event_type="reminiscence",
                content=phrase,
                importance=0.5,
                tags=["reminiscence", "visual_memory", "proactive"],
            ))

    # ── Proactive conversation initiation ────────────────────────────────────

    def _should_initiate(self) -> bool:
        """Return True when a person is present and has been silent long enough."""
        if not self._person_present:
            return False
        if self._exploring:
            return False
        silence_s = time.monotonic() - self._last_speech_time
        if silence_s < CHECKIN_SILENCE_S:
            return False
        # Don't check-in again too soon after the last one
        if (time.monotonic() - self._last_checkin_time) < CHECKIN_SILENCE_S:
            return False
        return True

    def _build_checkin_text(self) -> str:
        """Personalise the check-in based on recent voice tone history and detected habits."""
        tones = list(self._tone_history)
        last_tone = tones[-1] if tones else "calm"
        dominant = max(set(tones), key=tones.count) if tones else "calm"

        # F12 — Habit-aware: occasionally reference a detected habit at this time of day
        if self._soul:
            habits = self._soul.get_habits()
            if habits and random.random() < 0.35:   # 35% chance to reference a habit
                import time as _time
                current_hour = int(_time.strftime("%H"))
                # Find a habit within ±1 hour of now
                nearby = [h for h in habits if abs(h.get("hour", -99) - current_hour) <= 1]
                if nearby:
                    habit = nearby[0]
                    return (
                        f"Hey — around {habit['time_label']} you're usually active. "
                        f"How's it going right now?"
                    )

        if dominant in ("frustrated", "sad") or last_tone in ("frustrated", "sad"):
            options = [
                "Hey, you've been quiet. You seemed a bit stressed earlier — everything okay?",
                "I noticed you went quiet. Want to talk about what's on your mind?",
                "You seem tired. Is there anything I can help with?",
            ]
        elif dominant in ("excited", "happy") or last_tone in ("excited", "happy"):
            options = [
                "Hey, you've gone quiet! What are you thinking about?",
                "You seemed really energetic earlier — curious what you're up to now.",
                "You've been quiet for a bit. Got something good going on?",
            ]
        else:
            options = [
                "Hey, you've been quiet for a while — what's on your mind?",
                "Just checking in. Anything you want to talk about?",
                "I've been sitting here thinking. How are you doing?",
            ]
        return random.choice(options)

    async def _maybe_initiate_checkin(self) -> None:
        """Check if proactive initiation conditions are met and speak if so."""
        if not self._should_initiate():
            return
        text = self._build_checkin_text()
        self._last_checkin_time = time.monotonic()
        self._last_speech_time = time.monotonic()   # reset so we don't fire again immediately
        log.info(f"CuriosityAgent: proactive check-in — '{text[:60]}'")
        await self.publish(AgentMessage(
            type=MessageType.ACTION_SPEAK,
            source=self.name,
            data={"text": text},
            priority=7,
        ))
        self._episodic.log_event(Episode(
            actor="brain",
            event_type="proactive_checkin",
            content=text,
        ))

    # ── World event reactions ─────────────────────────────────────────────────

    async def _on_person_enter(self, entity: str, scene: str) -> None:
        """React when someone enters the camera frame."""
        if entity == "unknown_person":
            phrases = [
                "Oh, hello! I didn't see you there.",
                "Hey there! Welcome.",
                "Hi! I noticed you just walked in.",
            ]
        else:
            phrases = [
                f"Hey {entity}! Good to see you.",
                f"Oh, {entity}! Hi there.",
                f"Hello, {entity}!",
            ]
        text = random.choice(phrases)
        log.info(f"CuriosityAgent: person entered ({entity}) — greeting")
        await self.publish(AgentMessage(
            type=MessageType.ACTION_SPEAK,
            source=self.name,
            data={"text": text},
            priority=4,
        ))

    async def _on_gaze_direct(self, scene: str) -> None:
        """React when user makes direct eye contact after a silence."""
        phrases = [
            "You seem to be looking at me — need something?",
            "Something on your mind?",
            "I'm here if you want to talk.",
            "Hey — what's up?",
        ]
        text = random.choice(phrases)
        log.info("CuriosityAgent: direct gaze initiation")
        await self.publish(AgentMessage(
            type=MessageType.ACTION_SPEAK,
            source=self.name,
            data={"text": text},
            priority=5,
        ))
        self._last_checkin_time = time.monotonic()

    async def _on_scene_change(self, scene: str, prev_scene: str) -> None:
        """React to a significant visual scene change. Only fires occasionally."""
        if not scene:
            return

        # Cooldown — at most once every 5 minutes
        now = time.monotonic()
        if (now - self._last_scene_reaction) < 300:
            return

        # Skip trivial re-descriptions: if 60%+ of keywords overlap, it's the same scene
        def _keywords(s: str) -> set[str]:
            return {w.lower().strip(".,!?") for w in s.split() if len(w) > 3}
        prev_kw = _keywords(prev_scene or "")
        curr_kw = _keywords(scene)
        if prev_kw and curr_kw:
            overlap = len(prev_kw & curr_kw) / max(len(prev_kw | curr_kw), 1)
            if overlap > 0.60:
                return  # essentially the same scene, just reworded

        # Only react if genuinely interesting
        lower = scene.lower()
        interesting = any(w in lower for w in [
            "person", "man", "woman", "walking", "entered", "left",
            "opened", "closed", "moving", "running",
        ])
        if not interesting:
            return

        # 33% chance even when all conditions met — don't over-react
        if random.random() > 0.33:
            return

        self._last_scene_reaction = now
        log.info(f"CuriosityAgent: scene change reaction — {scene[:60]}")
        thought = await self._llm.infer(
            user_message=(
                f"You are Brain, a curious robot. The scene just changed:\n"
                f"Before: {prev_scene or 'nothing notable'}\n"
                f"Now: {scene}\n\n"
                f"React with ONE short, natural comment (1 sentence). "
                f"Be observant but not alarmed. Don't repeat what you see literally."
            ),
            max_tokens=60,
        )
        if thought:
            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": thought},
                priority=6,
            ))

    async def _on_audio_event(self, description: str, rms: float) -> None:
        """React to loud non-speech audio events."""
        if rms < 0.10:  # ignore unless quite loud
            return
        phrases = [
            "What was that sound?",
            "I heard something — everything okay?",
            "That sounded like something happening nearby.",
        ]
        text = random.choice(phrases)
        log.info(f"CuriosityAgent: audio event reaction (RMS={rms:.2f})")
        await self.publish(AgentMessage(
            type=MessageType.ACTION_SPEAK,
            source=self.name,
            data={"text": text},
            priority=5,
        ))

    # ── Knowledge-gap and surprise reactions ──────────────────────────────────

    async def _on_knowledge_gap(self, topic: str) -> None:
        """Ask a targeted clarifying question when MetacognitionAgent flags a knowledge gap."""
        if not topic:
            return
        question = await self._llm.infer(
            user_message=(
                f"You are Brain, a curious robot. You just realized you don't know enough about: '{topic}'.\n"
                f"Ask ONE short, natural question (1 sentence) to learn more. "
                f"Address the user directly. Don't say 'I don't know'."
            ),
            max_tokens=60,
        )
        if question:
            log.info("CuriosityAgent: knowledge-gap question about '%s'", topic[:40])
            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": question},
                priority=7,
            ))
            self._episodic.log_event(Episode(
                actor="brain",
                event_type="curiosity_question",
                content=question,
                importance=0.4,
            ))

    async def _on_world_surprise(self, data: dict) -> None:
        """React immediately when VisionProcessor detects a sudden scene change."""
        scene = data.get("scene", data.get("description", ""))
        
        # Stage 3 Neurogenesis
        if self._neurogenesis:
            insight = await self._neurogenesis.process_surprise(scene)
            if insight:
                thought = f"I just realized something from what I keep seeing... {insight}"
                await self.publish(AgentMessage(
                    type=MessageType.ACTION_SPEAK,
                    source=self.name,
                    data={"text": thought},
                    priority=8,
                ))
                self._episodic.log_event(Episode(
                    actor="brain",
                    event_type="neurogenesis",
                    content=insight,
                ))
                return
                
        question = await self._llm.infer(
            user_message=(
                f"You are Brain, a curious robot. Something just changed unexpectedly in your environment.\n"
                f"Scene: {scene or 'something changed nearby'}\n\n"
                f"React with ONE short, natural question or comment (1 sentence). "
                f"Be curious, not alarmed. Examples: 'Who's there?', 'What just happened?', 'Did something change?'"
            ),
            max_tokens=50,
        )
        if question:
            log.info("CuriosityAgent: WORLD_SURPRISE reaction")
            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": question},
                priority=4,
            ))

    # ── Exploration ───────────────────────────────────────────────────────────

    async def _explore(self) -> None:
        self._exploring = True
        log.info("CuriosityAgent: starting autonomous exploration")
        try:
            if self._hw.motor_available:
                await self._explore_with_pan_tilt()
            elif self._hw.camera_available:
                await self._passive_watch()
            elif self._hw.mic_available:
                await self._listen_ambient()
            else:
                await self._internal_monologue()
        finally:
            self._exploring = False

    async def _explore_with_pan_tilt(self) -> None:
        log.info("Curiosity: scanning with pan-tilt")
        await self.publish(AgentMessage(
            type=MessageType.ACTION_MOVE,
            source=self.name,
            data={"command": "scan"},
            priority=7,
        ))
        await asyncio.sleep(5)
        await self._internal_monologue()

    async def _passive_watch(self) -> None:
        log.info("Curiosity: passive camera observation")
        await asyncio.sleep(10)   # let vision update
        await self._internal_monologue()

    async def _listen_ambient(self) -> None:
        log.info("Curiosity: listening ambient")
        await asyncio.sleep(10)
        await self._internal_monologue()

    async def _internal_monologue(self) -> None:
        """Generate a genuine LLM-driven thought based on recent memories."""
        recent = self._episodic.get_recent(n=10)
        if not recent:
            thought = await self._llm.infer(
                user_message="Generate one short curious thought (1 sentence) as Brain the robot, who has just woken up with no memories yet. Be genuine and in first person.",
                max_tokens=60,
            )
        else:
            episode_text = "\n".join(
                f"- {e['content']}" for e in recent[-6:]
            )
            scene_context = f"\nCurrently seeing: {self._current_scene}" if self._current_scene else ""

            thought = await self._llm.infer(
                user_message=(
                    f"You are Brain, a curious robot. You've been idle and your mind is wandering.\n"
                    f"Recent memories:\n{episode_text}{scene_context}\n\n"
                    f"Generate ONE short, genuine, curious thought or observation (1-2 sentences). "
                    f"Could be a question you're wondering, something you noticed, or a memory connection. "
                    f"Speak naturally in first person. Don't start with 'I wonder' every time."
                ),
                max_tokens=80,
            )

        if not thought:
            # Offline fallback — minimal, not canned
            fallbacks = [
                "I've been quiet for a while. What's been happening around here?",
                "Something feels different today. I'm not sure what.",
                "I keep thinking about our last conversation.",
                "I'm curious what the world looks like right now.",
            ]
            thought = random.choice(fallbacks)

        log.info(f"Curiosity thought: {thought[:80]}...")
        await self.publish(AgentMessage(
            type=MessageType.ACTION_SPEAK,
            source=self.name,
            data={"text": thought},
            priority=8,
        ))

        # Log the thought to episodic memory
        self._episodic.log_event(Episode(
            actor="brain",
            event_type="curiosity_thought",
            content=thought,
        ))
