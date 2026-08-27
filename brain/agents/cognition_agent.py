"""
CognitionAgent — memory retrieval, context assembly, planning, self-reflection.
Maps to the Hippocampus + Prefrontal Cortex.

Autonomous behaviours:
- Injects live self-state (emotion, behavior, uptime) into every system prompt
- Teaches itself new skills when user says "whenever X do Y"
- Updates USER.md when user reveals personal information
- Handles SELF_REFLECT events: generates introspection and updates soul
"""
from __future__ import annotations

import asyncio
import collections
import re
import time
from datetime import datetime as _dt

from typing import TYPE_CHECKING

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.memory.working_memory import WorkingMemory
from brain.memory.episodic_memory import EpisodicMemory, Episode
from brain.memory.semantic_memory import SemanticMemory
from brain.memory.soul_manager import SoulManager
from brain.memory.gallery_manager import GalleryManager
from brain.llm.llm_router import LLMRouter
from brain.hardware.vision_processor import VisionProcessor
from brain.utils.logger import get_logger
from brain.utils.repo_map import build_repo_map

if TYPE_CHECKING:
    from brain.memory.world_model import WorldModel

log = get_logger(__name__)

# Patterns for extracting user facts from speech
_USER_FACT_PATTERNS = [
    (r"\bmy name is ([A-Z][a-z]+)\b", "name"),
    (r"\bI(?:'m| am) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b", "identity"),
    (r"\bI(?:'m| am) (?:a |an )?([a-z]+(?: [a-z]+)?)\b", "role"),
    (r"\bI work(?:ing)? (?:as |at |for )?([A-Za-z0-9 ]+)\b", "work"),
    (r"\bI live in ([A-Za-z ]+)\b", "location"),
    (r"\bI (?:love|like|enjoy|prefer) ([A-Za-z0-9 ]+)\b", "preference"),
    (r"\bI (?:hate|dislike|don't like) ([A-Za-z0-9 ]+)\b", "dislike"),
    (r"\bI have (?:a |an )?([A-Za-z0-9 ]+)\b", "possession"),
]

# Patterns for skill teaching ("whenever X, do Y" / "if X then Y")
_SKILL_PATTERNS = [
    r"whenever\s+(.+?)[,;]\s+(?:please\s+)?(.+)",
    r"when\s+(.+?)[,;]\s+(?:please\s+)?(.+)",
    r"if\s+(.+?)\s*,?\s*then\s+(.+)",
    r"remember to\s+(.+?)\s+when\s+(.+)",
    r"always\s+(.+?)\s+when\s+(.+)",
]

# N1a — Social Reinforcement Learning: correction phrases the user says to Brain
_CORRECTION_PHRASES: frozenset[str] = frozenset({
    "no that's wrong", "not what i meant", "that's incorrect", "you misunderstood",
    "wrong answer", "that's not right", "you're wrong", "that was wrong",
    "no not that", "that's not what", "incorrect", "you got it wrong",
    "that's not it", "no, that's", "no that is",
})

# V9 — Visual teaching phrases ("this is my X", "remember this as X")
_VISUAL_TEACH_PATTERNS = [
    re.compile(r"\bthis is (?:my |our )?([\w\s]{2,30})\b", re.I),
    re.compile(r"\bremember this as ([\w\s]{2,30})\b", re.I),
    re.compile(r"\bsave (?:this|that) as ([\w\s]{2,30})\b", re.I),
    re.compile(r"\bcall (?:this|that) (?:my )?([\w\s]{2,30})\b", re.I),
]

# N2a — Scenario-mapped preferences: time-contextual patterns
_SCENARIO_PATTERNS = [
    (r"in the (morning|afternoon|evening|night)\s+i\s+(?:like|love|prefer|enjoy|want|usually have)\s+([A-Za-z0-9 ]+)", 1, 2),
    (r"i\s+(?:like|love|prefer|usually have)\s+([A-Za-z0-9 ]+)\s+in the (morning|afternoon|evening|night)", 2, 1),
    (r"every (morning|afternoon|evening|night)\s+i\s+(?:have|drink|eat|do)\s+([A-Za-z0-9 ]+)", 1, 2),
]


# ── Mood arc sentiment lexicons ────────────────────────────────────────────────
# Lightweight word-count approach — no NLTK/TextBlob needed.
# Covers the most common positive/negative speech patterns in conversation.
_POS_WORDS: frozenset[str] = frozenset({
    "love", "great", "good", "thanks", "thank", "awesome", "amazing", "nice",
    "wonderful", "happy", "glad", "excited", "perfect", "excellent", "fantastic",
    "brilliant", "cool", "enjoy", "enjoyed", "fun", "interesting", "helpful",
    "appreciate", "beautiful", "yes", "please", "sure", "absolutely",
})
_NEG_WORDS: frozenset[str] = frozenset({
    "hate", "bad", "wrong", "boring", "stupid", "useless", "terrible", "awful",
    "frustrated", "annoying", "annoyed", "disappointed", "confused", "difficult",
    "hard", "problem", "error", "broken", "worst", "horrible", "sad", "tired",
    "no", "not", "don't", "doesn't", "can't", "won't", "fail", "failed", "ugh",
})


class CognitionAgent(BaseAgent):
    name = "cognition_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        soul: SoulManager,
        llm: LLMRouter,
        world_model: "WorldModel | None" = None,
        gallery: "GalleryManager | None" = None,
        vision_proc: "VisionProcessor | None" = None,
    ):
        super().__init__(bus)
        self._working = working
        self._episodic = episodic
        self._semantic = semantic
        self._soul = soul
        self._llm = llm
        self._world_model = world_model
        self._gallery = gallery
        self._vision_proc = vision_proc  # Shared CLIP encoder for Visual RAG text queries
        self._last_frame_b64: str = ""
        self._repo_map: str = build_repo_map()  # built once at startup, stable across turns

        # Live self-state — updated by incoming events
        self._current_emotion: str = "NEUTRAL"
        self._current_behavior: str = "IDLE"
        self._current_scene: str = ""
        self._current_objects: list[str] = []
        self._start_time: float = time.time()
        self._interaction_count: int = 0

        # Compaction + insight config
        self._COMPACT_THRESHOLD = 16   # user turns before auto-compact
        self._INSIGHT_INTERVAL = 5     # interactions between user insight passes
        self._last_insight_at: int = 0 # interaction count when last insight ran

        # VLM synchronisation — VisionAgent signals this when on-demand Moondream
        # query finishes so _process_intent can include fresh scene before LLM call
        self._vlm_event: asyncio.Event = asyncio.Event()

        # Mood arc — rolling sentiment across last 10 turns (mirror neuron analogue)
        # Score: -1.0 (very negative) to +1.0 (very positive)
        self._sentiment_history: collections.deque = collections.deque(maxlen=10)
        self._mood_arc: float = 0.0
        self._last_mood_trigger: str = ""   # prevents re-triggering same event

        # ── Higher-order cognition state (updated by new agents) ──────────────
        self._attention_focus:  str  = ""    # AttentionAgent: current focus topic
        self._user_model:       dict = {}    # TheoryOfMindAgent: expertise + confusion
        self._temporal_insight: str  = ""    # TemporalReasoningAgent: pattern summary
        self._pending_idea:     str  = ""    # IdeationAgent: creative context (consumed once)
        self._metacog_threshold: float = 1.0 # MetacognitionAgent: confidence (unused here)
        self._hw_caps: dict = {}             # HW capability flags — set by Orchestrator after build
        self._active_goal: str = ""          # Current planner goal — injected into stream4
        # v5.0 GWT slots
        self._intero_state:     dict = {}    # InteroceptionAgent: hardware affective state
        self._world_surprise:   bool = False # VisionAgent: prediction delta >30%
        # V10.0 — Phenomenal binding moment (BindingAgent) + relationship trust context
        self._binding_moment:   str  = ""    # BindingAgent: unified present-moment string
        self._trust_context:    str  = ""    # SoulManager: relationship depth directive

    async def start(self) -> None:
        await super().start()
        log.info("CognitionAgent started")
        while self._running:
            await asyncio.sleep(0.5)

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.EMOTION_CHANGE:
            self._current_emotion = message.data.get("to", "NEUTRAL")

        # V10.0 — Receive unified binding moment from BindingAgent
        elif message.type == MessageType.BINDING_UPDATE:
            self._binding_moment = message.data.get("binding_moment", "")

        # V10.0 — Receive body pose updates from ProprioceptionAgent
        elif message.type == MessageType.PROPRIOCEPTION_STATE:
            pose = message.data.get("pose", {})
            if pose:
                pan  = pose.get("pan_deg", 0.0)
                tilt = pose.get("tilt_deg", 0.0)
                hdg  = pose.get("heading_deg", 0.0)
                dist = pose.get("distance_cm", 0.0)
                self._current_scene = (
                    self._current_scene.split(" [Body:")[0]
                    + f" [Body: pan={pan:.0f}° tilt={tilt:.0f}° hdg={hdg:.0f}° dist={dist:.0f}cm]"
                )


        elif message.type == MessageType.BEHAVIOR_CHANGE:
            self._current_behavior = message.data.get("state", "IDLE")
            # ECO mode handoff — switch LLM router to minimal compute profile
            eco = message.data.get("eco")
            if eco is True:
                self._llm.set_eco_mode(True)
            elif eco is False:
                self._llm.set_eco_mode(False)

        elif message.type == MessageType.WORLD_UPDATE:
            # WorldModel broadcast — keep _current_scene in sync for legacy soul prompt
            scene = message.data.get("scene", "")
            if scene:
                self._current_scene = scene
            entities = message.data.get("present_entities", [])
            if entities and not scene:
                self._current_scene = f"Present: {', '.join(entities)}"

        elif message.type == MessageType.PERCEPTION_VISION:
            scene = message.data.get("scene", "")
            # Append recognized face names to scene description for LLM context
            faces: list[dict] = message.data.get("faces", [])
            if faces:
                known = [f["name"] for f in faces if f.get("name") and f["name"] != "unknown"]
                if known:
                    face_str = f"Recognized: {', '.join(known)}."
                    scene = f"{scene} {face_str}".strip() if scene else face_str
            if scene:
                self._current_scene = scene
            self._current_objects = message.data.get("objects", self._current_objects)
            frame_b64 = message.data.get("frame_b64", "")
            if frame_b64:
                self._last_frame_b64 = frame_b64
            # On-demand VLM query just finished — wake up any waiting _process_intent
            if message.data.get("on_demand"):
                self._vlm_event.set()

        elif message.type == MessageType.COGNITION_INTENT:
            await self._process_intent(message)

        elif message.type == MessageType.ACTION_SPEAK:
            # Store verified assistant responses in context.
            # Source is verifier_agent (after verification) or reasoning_agent
            # (direct fallback path). Exclude curiosity/emotion chatter.
            if message.source in ("reasoning_agent", "verifier_agent"):
                text = message.data.get("text", "")
                if text:
                    self._working.add_to_context("assistant", text)

        elif message.type == MessageType.MEMORY_WRITE:
            if message.data.get("discard_last_assistant"):
                # TTS was interrupted mid-speech — remove the partial response from
                # context so the LLM doesn't treat it as a complete answer next turn
                self._working.remove_last_assistant()
                log.debug("CognitionAgent: discarded interrupted assistant response from context")

        elif message.type == MessageType.INTERO_STATE:
            self._intero_state = message.data

        elif message.type == MessageType.WORLD_SURPRISE:
            # Force re-evaluation on the next intent — flag cleared after use
            self._world_surprise = True
            log.info("CognitionAgent: WORLD_SURPRISE received (delta=%.0f%%)",
                     message.data.get("delta", 0) * 100)

        elif message.type == MessageType.ATTENTION_FOCUS:
            self._attention_focus = message.data.get("focus_target", "")

        elif message.type == MessageType.USER_MODEL_UPDATE:
            self._user_model = message.data

        elif message.type == MessageType.TEMPORAL_INSIGHT:
            self._temporal_insight = message.data.get("summary", "")

        elif message.type == MessageType.IDEATION_RESULT:
            self._pending_idea = message.data.get("idea", "")

        elif message.type == MessageType.METACOG_CONFIDENCE:
            self._metacog_threshold = message.data.get("score", 1.0)

        elif message.type == MessageType.SELF_REFLECT:
            await self._run_self_reflection()

    # ── Mood arc ───────────────────────────────────────────────────────────────

    def _score_sentiment(self, text: str) -> float:
        """Score a single utterance in [-1.0, +1.0] using word-count heuristics.
        Positive words score +1 each, negative words score -1 each.
        Normalised by total matched words so length doesn't dominate."""
        words = re.findall(r"\b[a-z]+\b", text.lower())
        pos = sum(1 for w in words if w in _POS_WORDS)
        neg = sum(1 for w in words if w in _NEG_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    async def _update_mood_arc(self, text: str) -> None:
        """Score utterance, update rolling arc, emit BEHAVIOR_CHANGE if threshold crossed."""
        score = self._score_sentiment(text)
        self._sentiment_history.append(score)
        self._mood_arc = sum(self._sentiment_history) / len(self._sentiment_history)

        # Map arc to emotion trigger — fire only when crossing a new threshold
        if self._mood_arc < -0.4 and self._last_mood_trigger != "user_frustrated":
            self._last_mood_trigger = "user_frustrated"
            await self.publish(AgentMessage(
                type=MessageType.BEHAVIOR_CHANGE,
                source=self.name,
                data={"mood_arc": round(self._mood_arc, 2), "trigger": "user_frustrated"},
                priority=5,
            ))
            log.debug(f"Mood arc → negative ({self._mood_arc:.2f}) — signalling EmotionEngine")
        elif self._mood_arc > 0.4 and self._last_mood_trigger != "positive_sentiment":
            self._last_mood_trigger = "positive_sentiment"
            await self.publish(AgentMessage(
                type=MessageType.BEHAVIOR_CHANGE,
                source=self.name,
                data={"mood_arc": round(self._mood_arc, 2), "trigger": "positive_sentiment"},
                priority=5,
            ))
            log.debug(f"Mood arc → positive ({self._mood_arc:.2f}) — signalling EmotionEngine")
        elif -0.2 <= self._mood_arc <= 0.2:
            # Arc returned to neutral — reset so next crossing re-fires
            self._last_mood_trigger = ""

    # ── Intent processing ──────────────────────────────────────────────────

    async def _process_intent(self, message: AgentMessage) -> None:
        text = message.data.get("text", "")
        intent = message.data.get("intent", "general_query")
        entities = message.data.get("entities", {})

        if not text:
            return

        self._interaction_count += 1
        self._soul.increment_interactions()

        # ── Handle memory reset command ──────────────────────────────────────
        if intent == "memory_reset":
            self._working.clear_context()
            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": "Done. I've cleared our conversation history. Fresh start."},
                priority=2,
            ))
            return
            
        # ── Handle task intent (intercepted by PlannerAgent) ─────────────────
        if intent in ("task", "plan", "do", "execute", "embodied"):
            return


        # ── Auto-compact context when it gets long ───────────────────────────
        if self._working.turn_count >= self._COMPACT_THRESHOLD:
            await self._compact_context()

        # ── Skill teaching: "whenever X, do Y" ──────────────────────────────
        if intent == "skill_teach" or self._detect_skill_teaching(text):
            await self._handle_skill_teaching(text)
            return

        # ── Passive user-fact capture ────────────────────────────────────────
        self._capture_user_facts(text)
        await self._capture_visual_teaching(text)
        self._capture_scenario_preferences(text)

        # ── N1a: Correction detection → NEURO_REWEIGHT ───────────────────────
        lower_text = text.lower()
        if any(phrase in lower_text for phrase in _CORRECTION_PHRASES):
            self._episodic.log_event(Episode(
                actor="user",
                event_type="correction",
                content=text,
                emotion=self._current_emotion,
                outcome="correction",
                importance=0.9,
                tags=["correction"],
            ))
            log.info("CognitionAgent: correction detected — publishing NEURO_REWEIGHT")
            # Extract what the user is correcting to (text after "actually", "it's", etc.)
            corrected_text = text
            for marker in ["actually", "it's", "its", "it is", "the answer is", "no,", "no it"]:
                idx = lower_text.find(marker)
                if idx != -1:
                    corrected_text = text[idx + len(marker):].strip(" ,")
                    break
            # Pre-compute embedding for the corrected fact (reused by MetacognitionAgent)
            correction_embedding: list[float] = []
            try:
                correction_embedding = await self._llm.embed(corrected_text) or []
            except Exception:
                pass
            await self.publish(AgentMessage(
                type=MessageType.NEURO_REWEIGHT,
                source=self.name,
                data={
                    "memory_id":      getattr(self, "_last_retrieved_doc_id", ""),
                    "penalty":        0.3,
                    "original_query": text,
                    "corrected_text": corrected_text[:200],
                    "embedding":      correction_embedding,
                },
                priority=8,
            ))

        # ── Mood arc update — score sentiment, mirror to EmotionEngine ───────
        await self._update_mood_arc(text)

        # ── LLM-driven user insight (every N interactions) ───────────────────
        if (self._interaction_count - self._last_insight_at) >= self._INSIGHT_INTERVAL:
            self._last_insight_at = self._interaction_count
            asyncio.create_task(self._extract_user_insights())

        # ── Retrieve episodic context (salient-first) ────────────────────────
        # get_salient ranks by importance + emotional arousal boost so the LLM
        # sees emotionally significant moments before mundane recent ones.
        recent_episodes = self._episodic.get_salient(n=8)
        recent_text = "\n".join(
            f"- [{e['actor']}] {e['content']}" + (f" [felt: {e['emotion']}]" if e.get("emotion") else "")
            for e in recent_episodes
        )

        # ── Semantic memory search ───────────────────────────────────────────
        query_embedding = await self._llm.embed(text)
        semantic_facts = self._semantic.search_similar(query_embedding, k=5)
        facts_text = "\n".join(f"- {f['content']}" for f in semantic_facts)
        # Track top result for NEURO_REWEIGHT (so corrections know what to penalize)
        if semantic_facts:
            import hashlib as _hl
            self._last_retrieved_doc_id = _hl.md5(semantic_facts[0]["content"].encode()).hexdigest()
        else:
            self._last_retrieved_doc_id = ""

        # ── Visual RAG: Agentic Visual Retrieval ─────────────────────────────
        # For visual questions ("where", "who", "what's in", etc.) we perform a
        # cross-modal CLIP search: text query → 512-d vector → brain_visual_memory.
        # Retrieved frames are injected into the LLM context as grounded visual evidence.
        retrieved_frames_b64: list[str] = []
        visual_memory_context: str = ""
        if VisionProcessor.is_visual_question(text) and self._semantic.visual_memory_count() > 0:
            clip_vec: list[float] = []
            try:
                # Use VisionProcessor.encode_text_features for cross-modal embedding
                vp = getattr(self, "_vision_proc", None)
                if vp is None:
                    from brain.hardware.vision_processor import VisionProcessor as _VP
                    vp = _VP(mode="auto")
                loop = asyncio.get_event_loop()
                clip_vec = await loop.run_in_executor(
                    None, vp.encode_text_features, text
                )
            except Exception as _e:
                log.debug("CognitionAgent: CLIP text encode failed: %s", _e)

            if clip_vec:
                visual_hits = self._semantic.search_visual_memories(clip_vec, k=2)
                if visual_hits:
                    log.info(
                        "CognitionAgent: Visual RAG — %d historical frames retrieved for '%s'",
                        len(visual_hits), text[:60],
                    )
                    mem_lines = []
                    for hit in visual_hits:
                        mem_lines.append(
                            f"- [score={hit['score']:.2f}] {hit['content']}"
                            + (f" (seen at {hit['created_at'][:16]})" if hit.get("created_at") else "")
                        )
                        # Prefer inline thumbnail (no disk I/O) over legacy disk path
                        thumb = hit.get("thumbnail_b64", "")
                        if thumb:
                            retrieved_frames_b64.append(thumb)
                        else:
                            # Legacy fallback: load from disk if image_path is set
                            img_path = hit.get("image_path", "")
                            if img_path:
                                try:
                                    from pathlib import Path
                                    import base64 as _b64
                                    raw = Path(img_path).read_bytes()
                                    retrieved_frames_b64.append(_b64.b64encode(raw).decode())
                                except Exception as _le:
                                    log.debug("CognitionAgent: could not load visual memory image: %s", _le)
                    if mem_lines:
                        visual_memory_context = (
                            "## Visual Memories (Historical Frames)\n"
                            "The following scenes were recorded by the camera in the past.\n"
                            + "\n".join(mem_lines)
                        )

        # ── Wait for on-demand VLM answer on visual questions ───────────────
        # VisionAgent runs a Moondream query in parallel when it detects visual
        # keywords. Wait up to 4 s so the fresh scene is included in the prompt.
        if VisionProcessor.is_visual_question(text):
            self._vlm_event.clear()
            try:
                await asyncio.wait_for(self._vlm_event.wait(), timeout=4.0)
                log.debug("CognitionAgent: VLM answer received, proceeding with LLM")
            except asyncio.TimeoutError:
                log.debug("CognitionAgent: VLM timeout — using cached scene")

        # ── Build context window ─────────────────────────────────────────────
        context_messages = self._working.get_context()

        # ── Time-of-day context (F5) ─────────────────────────────────────────
        now_dt = _dt.now()
        hour = now_dt.hour
        if 5 <= hour < 12:
            time_label = "morning"
        elif 12 <= hour < 17:
            time_label = "afternoon"
        elif 17 <= hour < 21:
            time_label = "evening"
        else:
            time_label = "night"
        time_context = (
            f"It is currently {time_label} ({now_dt.strftime('%H:%M')} local time). "
            f"Greet and respond in a way that fits the time — e.g. warm 'good morning' "
            f"energy in the morning, calmer and quieter late at night."
        )

        # ── 4-Stream Contextual Brain Assembly ───────────────────────────────
        # Stream 1 → Stream 2 → Stream 3 → Stream 4 (top = first token LLM reads)
        uptime = time.time() - self._start_time

        # ── Stream 1: Identity — who is talking to Brain (always first) ──────
        # _EMBODIMENT_ANCHOR is a raw-string prefix — ultra-compressed, processed
        # in the first tokens even under extreme affective-fatigue token caps.
        # soul_manager.get_identity_header() provides the full SENSORY RULES detail.
        _EMBODIMENT_ANCHOR = (
            "⚡ YOU ARE BRAIN — AN EMBODIED ROBOT. "
            "You physically SEE through your camera right now. "
            "NEVER say 'I can see in this image' or 'thanks for sharing'. "
            "ALWAYS say 'I see', 'I'm looking at', 'right now in our space'.\n\n"
        )
        stream1 = _EMBODIMENT_ANCHOR + self._soul.get_identity_header()

        # ── Stream 2: Vitals + Environment ───────────────────────────────────
        intero = self._intero_state or {}
        hw_label = intero.get("label", "comfortable")
        cpu  = intero.get("cpu_pct", 0)
        ram  = intero.get("ram_pct", 0)
        temp = intero.get("temp_c", 0)
        feeling_map = {
            "overwhelmed":     "overwhelmed — keep replies very brief",
            "memory_pressure": "under memory pressure — keep replies brief",
            "hot":             "running hot",
            "busy":            "busy",
            "comfortable":     "comfortable",
        }
        hw_feeling = feeling_map.get(hw_label, hw_label)
        vitals_line = f"Hardware: {hw_feeling}"
        if cpu or ram:
            vitals_line += f" (CPU {cpu:.0f}%, RAM {ram:.0f}%"
            if temp > 0:
                vitals_line += f", {temp:.0f}°C"
            vitals_line += ")"
        scene = self._current_scene or "no visual data yet"
        stream2 = (
            f"## CURRENT VITALS & ENVIRONMENT\n"
            f"Time: {now_dt.strftime('%A %H:%M')}. {vitals_line}.\n"
            f"Visual: {scene}\n\n"
        )

        # ── Stream 3: Memory — episodic + semantic ────────────────────────────
        stream3 = ""
        if recent_text:
            stream3 += f"## WHAT I REMEMBER\n{recent_text}\n\n"
        if facts_text:
            stream3 += f"## WHAT I KNOW\n{facts_text}\n\n"

        # ── Stream 4: Soul + Attention + World + Capabilities + Goal ──────────
        # Attention spotlight and live world snapshot go here (context for soul)
        stream4_prefix = ""
        if self._attention_focus:
            stream4_prefix += (
                f"## Attention Focus\nCurrently focused on: \"{self._attention_focus}\". "
                f"Prioritise this topic.\n\n"
            )
        if self._world_model:
            world_ctx = await self._world_model.format_for_prompt()
            if world_ctx and world_ctx != "No current perceptual context.":
                surprise_note = " ⚠ Unexpected change detected." if self._world_surprise else ""
                stream4_prefix += f"## World Snapshot{surprise_note}\n{world_ctx}\n\n"
                self._world_surprise = False

        stream4 = stream4_prefix + self._soul.get_system_prompt(
            emotion=self._current_emotion,
            hw_summary="",           # vitals already in stream2
            behavior_state=self._current_behavior,
            uptime_seconds=uptime,
            interaction_count=self._interaction_count,
            current_scene="",        # scene already in stream2
            active_skills=self._soul.skills,
            time_context=time_context,
            repo_map=self._repo_map,
        )
        if self._hw_caps:
            stream4 += "\n\n" + self._soul.get_capabilities_block(self._hw_caps)
        if self._active_goal:
            stream4 += f"\n\n## CURRENT GOAL\n{self._active_goal}\n"

        system_prompt = stream1 + stream2 + stream3 + stream4

        # ── Remaining higher-order enrichments ────────────────────────────────
        # Mood arc — rolling sentiment over last 10 turns
        if len(self._sentiment_history) >= 2:
            if self._mood_arc > 0.3:
                mood_label = "positive"
            elif self._mood_arc < -0.3:
                mood_label = "negative"
            else:
                mood_label = "neutral"
            system_prompt += (
                f"\n\n## Conversation Mood Arc\n"
                f"User mood over last {len(self._sentiment_history)} turns: "
                f"{mood_label} ({self._mood_arc:+.2f}). "
                f"Adjust your tone — be warmer if negative, match energy if positive."
            )

        if self._user_model:
            expertise = self._user_model.get("expertise_level", "medium")
            depth     = self._user_model.get("explanation_depth", "normal")
            conf      = self._user_model.get("confusion_score", 0.0)
            frust     = self._user_model.get("frustration_score", 0.0)
            conf_note = "Simplify your explanation." if conf > 0.6 else "Current depth is appropriate."
            frust_note = " User seems frustrated — be patient and concise." if frust > 0.5 else ""
            system_prompt += (
                f"\n\n## User Model\n"
                f"Expertise: {expertise}. Explanation depth: {depth}. "
                f"{conf_note}{frust_note}"
            )

        if self._temporal_insight:
            system_prompt += f"\n\n## Temporal Context\n{self._temporal_insight}"

        if self._pending_idea:
            system_prompt += f"\n\n## Creative Context\n{self._pending_idea}"
            self._pending_idea = ""   # consume once per response

        # ── Inject Visual Memory context into system_prompt ──────────────────
        if visual_memory_context:
            system_prompt += f"\n\n{visual_memory_context}"
            system_prompt += (
                "\n\nUse these visual memories to ground your answer. "
                "If you can see the object or person in a past frame, "
                "reference approximately when it was seen."
            )

        # N2b — Predicted Needs: inject time-contextual scenario preferences
        scenario_prefs = self._soul.get_scenario_preferences()
        habits = self._soul.get_habits()
        predicted_needs: list[str] = []
        if scenario_prefs.get(time_label):
            predicted_needs.append(f"user may want {scenario_prefs[time_label]}")
        habit_match = [h for h in habits if abs(h.get("hour", -99) - now_dt.hour) <= 1]
        if habit_match:
            predicted_needs.append(f"usual habit: {habit_match[0].get('activity', '')}")
        if predicted_needs:
            system_prompt += (
                f"\n\n## Predicted Needs\nIt is {time_label}. "
                + " | ".join(predicted_needs) + "."
            )

        # V10.0 — Inject phenomenal binding moment (BindingAgent)
        # This is the unified "present moment" across all perceptual and affective streams.
        if self._binding_moment:
            system_prompt += f"\n\n## My Present Moment\n{self._binding_moment}"

        # V10.0 — Inject relationship trust context (SoulManager)
        # Modulates behavioral tone: stranger vs friend vs companion.
        try:
            trust_ctx = self._soul.get_trust_context()
            if trust_ctx:
                system_prompt += f"\n\n## Relationship\n{trust_ctx}"
        except AttributeError:
            pass  # Old SoulManager without V10.0 trust methods


        log.debug(f"Cognition processing intent={intent}: '{text[:60]}'")

        # Vision squelch — code/UI intents must not be distracted by camera frames
        _CODE_INTENTS = {"ui_update", "code_modification", "code_change", "software_update"}
        _CODE_KEYWORDS = ("update ui", "change ui", "modify ui", "update code",
                          "change code", "modify code", "update the", "change the title",
                          "execute: modify", "[execute:")
        _is_code_intent = (
            intent in _CODE_INTENTS
            or any(kw in text.lower() for kw in _CODE_KEYWORDS)
        )
        if _is_code_intent:
            self._current_scene = ""
            retrieved_frames_b64 = []
            log.info("CognitionAgent: code intent detected — squelching visual input")

        # Affective feedback: combine cached intero_state with freshest WorldModel reading.
        # WorldModel.get_intero() is always current; _intero_state may lag by up to one
        # INTERO_STATE event. Take the more conservative (smaller) cap so hardware strain
        # is never understated when routing to ReasoningAgent.
        suggested_max = self._intero_state.get("suggested_max_tokens") if self._intero_state else None
        if self._world_model:
            fresh_cap = (await self._world_model.get_intero()).get("suggested_max_tokens")
            if fresh_cap is not None:
                suggested_max = min(suggested_max, fresh_cap) if suggested_max is not None else fresh_cap

        # Route to reasoning agent
        await self.publish(AgentMessage(
            type=MessageType.COGNITION_INTENT,
            source=self.name,
            target="reasoning_agent",
            data={
                "text":                 text,
                "intent":               intent,
                "entities":             entities,
                "system_prompt":        system_prompt,
                "context_messages":     context_messages,
                "current_scene":        self._current_scene,
                "suggested_max_tokens": suggested_max,
                "emotion":              self._current_emotion,
                # Visual RAG: live frame is already injected by ReasoningAgent;
                # we additionally supply retrieved historical frames (max 2)
                # so the LLM can compare 'what I see now' vs 'what I saw before'.
                "retrieved_frames_b64": retrieved_frames_b64,
            },
            priority=3,
        ))

        # Store user message in working memory context
        self._working.add_to_context("user", text)

        # Log to episodic memory with current emotion so DreamAgent and future
        # salient retrieval can surface emotionally significant interactions.
        self._episodic.log_event(Episode(
            actor="user",
            event_type="speech",
            content=text,
            emotion=self._current_emotion,
            outcome="neutral",
            importance=0.6,
        ))

    # ── Skill teaching ─────────────────────────────────────────────────────

    def _detect_skill_teaching(self, text: str) -> bool:
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in _SKILL_PATTERNS)

    async def _handle_skill_teaching(self, text: str) -> None:
        text_lower = text.lower()
        trigger, action = "", ""

        for pattern in _SKILL_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                # Pattern order determines which group is trigger vs action
                if "remember to" in pattern or "always" in pattern:
                    action, trigger = m.group(1).strip(), m.group(2).strip()
                else:
                    trigger, action = m.group(1).strip(), m.group(2).strip()
                break

        if trigger and action:
            skill = {"trigger": trigger, "action": action, "taught_at": _now_iso()}
            self._soul.add_skill(skill)
            log.info(f"Skill learned: when '{trigger}' → '{action}'")
            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": f"Got it. I'll remember: when {trigger}, I'll {action}."},
                priority=2,
            ))
        else:
            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": "I want to learn that skill but couldn't quite parse the trigger and action. Could you rephrase it as 'whenever X, do Y'?"},
                priority=2,
            ))

    # ── User fact capture ──────────────────────────────────────────────────

    def _capture_user_facts(self, text: str) -> None:
        for pattern, fact_type in _USER_FACT_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                value = m.group(1).strip()
                if 1 < len(value) < 60:
                    self._soul.upsert_user_fact(fact_type, value)
                    log.info(f"USER.json updated: {fact_type}={value}")

    async def _capture_visual_teaching(self, text: str) -> None:
        """V9 — Detect 'this is my X' phrases, save current frame to gallery + SemanticMemory."""
        if not self._gallery or not self._last_frame_b64:
            return
        text_lower = text.lower()
        for pat in _VISUAL_TEACH_PATTERNS:
            m = pat.search(text_lower)
            if m:
                label = m.group(1).strip()
                if len(label) < 2 or len(label) > 40:
                    continue
                image_id, file_path = self._gallery.save_frame(self._last_frame_b64, description=label)
                if not file_path:
                    break
                if self._semantic:
                    description = f"{label}: {self._current_scene or 'visual memory'}"
                    embed = await self._llm.embed(description)
                    if embed:
                        self._semantic.upsert(
                            content=description,
                            embedding=embed,
                            category="visual_memory",
                            confidence=0.9,
                            source="user_teaching",
                            extra={
                                "image_path": file_path,
                                "image_id": image_id,
                                "label": label,
                                "visual_description": self._current_scene or "",
                            },
                        )
                log.info("Visual teaching captured: label='%s' image=%s", label, image_id)
                break

    def _capture_scenario_preferences(self, text: str) -> None:
        """N2a — Detect time-contextual preferences ('In the morning I like coffee')."""
        for pattern, scenario_group, pref_group in _SCENARIO_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                scenario = m.group(scenario_group).strip().lower()
                preference = m.group(pref_group).strip()
                if 1 < len(preference) < 60 and scenario in ("morning", "afternoon", "evening", "night"):
                    self._soul.upsert_scenario_preference(scenario, preference)
                    log.info("CognitionAgent: scenario preference captured — %s → %s", scenario, preference)
                    break

    # ── Context compaction ─────────────────────────────────────────────────

    async def _compact_context(self) -> None:
        """Summarize old conversation turns to keep context window lean."""
        context = self._working.get_context()
        if len(context) < 6:
            return

        # Build a plain-text transcript to summarize (exclude existing summaries)
        turns_to_summarize = [
            m for m in context[:-4]   # keep last 4 turns verbatim
            if m["role"] != "system"
        ]
        if not turns_to_summarize:
            return

        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in turns_to_summarize
        )
        log.info(f"CognitionAgent: compacting {len(turns_to_summarize)} turns...")

        summary = await self._llm.infer(
            user_message=transcript,
            system_prompt=(
                "You are summarizing a conversation between a user and Brain (a robot). "
                "Write a concise summary (3-5 sentences) capturing the key topics discussed, "
                "any facts the user shared, and Brain's main responses. "
                "Be factual and preserve important details."
            ),
            max_tokens=200,
        )

        if summary:
            self._working.replace_with_summary(summary, keep_last=4)
            # Store in episodic memory
            self._episodic.log_event(Episode(
                actor="brain",
                event_type="context_compact",
                emotion=self._current_emotion,
                content=f"Context compacted: {summary}",
            ))
            log.info("Context compacted successfully")

    # ── LLM user insight extraction ────────────────────────────────────────

    async def _extract_user_insights(self) -> None:
        """Run an async LLM pass to extract nuanced user facts not caught by regex."""
        recent = self._episodic.get_salient(n=12)
        if not recent:
            return

        episode_text = "\n".join(
            f"- [{e['actor']}] {e['content']}" for e in recent
            if e.get("actor") == "user" or e.get("event_type") == "speech"
        )
        if not episode_text.strip():
            return

        current_summary = self._soul.get_user_summary()

        insight = await self._llm.infer(
            user_message=episode_text,
            system_prompt=(
                f"You are analyzing recent interactions to learn about the user.\n"
                f"Current user profile:\n{current_summary}\n\n"
                f"From the recent interactions above, identify ONLY genuinely new facts "
                f"about the user that are NOT already in the profile above. "
                f"Focus on: communication style, topics they care about, preferences, "
                f"how they feel about Brain, their habits or context.\n"
                f"If there's nothing new to add, reply with exactly: NO_NEW_INSIGHTS\n"
                f"Otherwise write ONE concise insight sentence (not a list)."
            ),
            max_tokens=100,
        )

        if insight and "NO_NEW_INSIGHTS" not in insight and len(insight.strip()) > 10:
            self._soul.add_llm_insight(insight.strip())
            log.info(f"User insight added to USER.json: {insight[:80]}...")

    # ── Self-reflection ────────────────────────────────────────────────────

    async def _run_self_reflection(self) -> None:
        log.info("CognitionAgent: running self-reflection...")

        recent = self._episodic.get_salient(n=20)
        if not recent:
            log.debug("Self-reflect: no recent episodes, skipping")
            return

        episode_text = "\n".join(
            f"- [{e['timestamp'][:16]}] {e['content']}" for e in recent
        )
        uptime = self._soul._format_uptime(time.time() - self._start_time)

        prompt = f"""You are Brain, reflecting on your recent experiences.

Current state:
- Emotion: {self._current_emotion}
- Uptime: {uptime}
- Interactions: {self._interaction_count}

Recent experiences:
{episode_text}

Reflect briefly (2-3 sentences):
1. What have you experienced or learned?
2. How do you feel about it?
3. Is there anything you're curious about or want to explore?

Speak in first person as Brain. Be genuine, not robotic."""

        reflection = await self._llm.infer(
            user_message="Reflect on recent experiences.",
            system_prompt=prompt,
            max_tokens=200,
        )

        if reflection:
            log.info(f"Self-reflection: {reflection[:100]}...")
            # Speak the reflection aloud
            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": reflection},
                priority=8,
            ))
            # Store in episodic memory
            self._episodic.log_event(Episode(
                actor="brain",
                event_type="self_reflection",
                content=reflection,
                emotion=self._current_emotion,
            ))


def _now_iso() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
