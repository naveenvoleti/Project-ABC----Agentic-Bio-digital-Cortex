"""MetacognitionAgent — Anterior Prefrontal Cortex mapping.

v5.0: Self-Correction agent. Scores confidence in every COGNITION_RESPONSE using
keyword heuristics AND validates the response against known facts in WORLD.md.
If the response contradicts a world fact, the confidence is penalised and a
contradiction note is included in METACOG_CONFIDENCE.

Publishes METACOG_CONFIDENCE so VerifierAgent can:
  - Apply a "I'm not fully certain, but: " prefix on very low confidence.
  - Flag world-fact contradictions before the response reaches SpeechAgent.
  - Log knowledge gaps for IntrinsicMotivationAgent to pick up.

No LLM calls — entirely heuristic, ~2 MB.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from .base_agent import AgentMessage, BaseAgent, MessageType
from brain.memory.soul_manager import SoulManager
from brain.memory.semantic_memory import SemanticMemory

log = logging.getLogger(__name__)

_WORLD_FILE = Path("data/WORLD.md")

# ── Confidence scoring constants ─────────────────────────────────────────────
_UNCERTAINTY_PHRASES = frozenset({
    "i think", "i believe", "probably", "maybe", "perhaps",
    "not sure", "i'm not certain", "i don't know", "uncertain",
    "approximately", "i'm not sure", "it's possible", "might be",
    "i cannot say", "i'm unsure",
})
_REFUSAL_PHRASES = frozenset({
    "i cannot", "i don't have access", "i'm unable", "i lack",
    "no information", "i have no data",
})
_FACT_PATTERN = re.compile(
    r'\b(\d{4}|\d+\.\d+|[A-Z][a-z]+(?:\s[A-Z][a-z]+)+|\d{1,3}%)\b'
)

_LOW_CONF_THRESHOLD  = 0.3   # below this → flag to verifier for prefix
_BASE_SCORE          = 0.80  # start optimistic
_HISTORY_LEN         = 10


class MetacognitionAgent(BaseAgent):
    """Anterior PFC — knows what it doesn't know; validates against WORLD.md."""

    name = "metacognition_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        soul: SoulManager | None = None,
        persist_knowledge_gaps: bool = True,
    ):
        super().__init__(bus)
        self._soul = soul
        self._persist_knowledge_gaps = persist_knowledge_gaps
        self._scores: collections.deque[float] = collections.deque(maxlen=_HISTORY_LEN)
        self._last_confidence: float = 0.8     # V7.0: polled by Orchestrator synthesis trigger
        self._current_task_intent: str = ""    # V8.0: stores active TASK_EXECUTE goal
        self._knowledge_gaps: list[str] = []   # shared with IntrinsicMotivationAgent
        self._world_facts: list[str] = []       # loaded from WORLD.md on start
        self._world_loaded_at: float = 0.0
        self._WORLD_RELOAD_S = 300.0            # reload WORLD.md every 5 min
        self._last_retrieved_memory_id: str = ""   # V9: set when a semantic result is used
        self._last_retrieved_doc_id: str = ""       # NEURO_REWEIGHT: ChromaDB doc_id to penalize
        self._correction_ledger: list[dict] = []   # V9: in-session correction log
        self._semantic: SemanticMemory | None = None  # injected via set_semantic()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        # Restore persisted knowledge gaps from previous session
        if self._soul:
            self._knowledge_gaps = self._soul.get_knowledge_gaps()
            log.info("MetacognitionAgent: restored %d knowledge gaps from USER.json",
                     len(self._knowledge_gaps))
        await self._load_world_facts()
        log.info("MetacognitionAgent: started — low_conf_threshold=%.2f, world_facts=%d",
                 _LOW_CONF_THRESHOLD, len(self._world_facts))
        asyncio.create_task(self._world_reload_loop(), name="metacog-world-reload")
        while self._running:
            await asyncio.sleep(1)

    async def _load_world_facts(self) -> None:
        """Parse WORLD.md into a list of fact sentences for contradiction checking."""
        try:
            if _WORLD_FILE.exists():
                text = await asyncio.to_thread(_WORLD_FILE.read_text, encoding="utf-8")
                # Extract non-empty, non-header lines as facts
                facts = [
                    line.strip().lstrip("- *")
                    for line in text.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                    and len(line.strip()) > 10
                ]
                self._world_facts = facts[:50]   # cap to avoid bloat
                self._world_loaded_at = time.time()
        except Exception as exc:
            log.debug("MetacognitionAgent: could not load WORLD.md — %s", exc)

    async def _world_reload_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._WORLD_RELOAD_S)
            await self._load_world_facts()
            log.debug("MetacognitionAgent: WORLD.md reloaded (%d facts)", len(self._world_facts))

    # ── Message handling ─────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        # V8.0 — Store active task intent for outcome comparison
        if message.type == MessageType.TASK_EXECUTE:
            self._current_task_intent = message.data.get("goal", message.data.get("text", ""))
            return None

        # V8.0 — Compare VisionAgent observed state against stored intent
        if message.type == MessageType.OUTCOME_ANALYSIS:
            await self._analyse_outcome(message.data)
            return None

        # V9 — Detect user corrections and lower confidence of last-used memory
        if message.type == MessageType.PERCEPTION_SPEECH:
            text = message.data.get("text", "").lower()
            _CORRECTION_WORDS = {"wrong", "incorrect", "actually", "not quite", "that's not", "no that"}
            if any(w in text for w in _CORRECTION_WORDS) and self._last_retrieved_memory_id:
                await self._lower_memory_confidence(self._last_retrieved_memory_id, text)
            return None

        # NEURO_REWEIGHT — CognitionAgent detected a user correction; penalize wrong memory
        if message.type == MessageType.NEURO_REWEIGHT:
            await self._handle_neuro_reweight(message.data)
            return None

        if message.type == MessageType.COGNITION_RESPONSE:
            text   = message.data.get("text", "")
            intent = message.data.get("intent", "")
            score  = self._score_confidence(text, intent)

            # v5.0 Self-Correction: check response against WORLD.md facts
            contradiction, contradiction_note = self._check_world_facts(text)
            if contradiction:
                score = max(0.0, score - 0.30)   # penalise heavily
                log.info("MetacognitionAgent: world contradiction detected — '%s'",
                         contradiction_note[:80])
                await self.publish(AgentMessage(
                    type=MessageType.NEURO_SYNTHESIS,
                    source=self.name,
                    data={"trigger": "world_contradiction", "note": contradiction_note, "text": text},
                    priority=8,
                ))

            self._scores.append(score)
            self._last_confidence = score

            uncertain = score < _LOW_CONF_THRESHOLD
            if uncertain:
                topic = message.data.get("topic") or intent or "unknown_topic"
                is_new_gap = self._track_knowledge_gap(topic, text)
                if is_new_gap:
                    await self.publish(AgentMessage(
                        type=MessageType.CURIOSITY_TRIGGER,
                        source=self.name,
                        data={"source": "knowledge_gap", "topic": topic},
                        priority=7,
                    ))

            await self.publish(AgentMessage(
                type=MessageType.METACOG_CONFIDENCE,
                source=self.name,
                data={
                    "score":                round(score, 3),
                    "uncertain":            uncertain,
                    "avg_score":            round(sum(self._scores) / len(self._scores), 3)
                                            if self._scores else score,
                    "knowledge_gaps":       list(self._knowledge_gaps[-5:]),
                    "world_contradiction":  contradiction,
                    "contradiction_note":   contradiction_note,
                    "ts":                   time.time(),
                },
                priority=6,
            ))
            log.debug("MetacognitionAgent: score=%.2f uncertain=%s contradiction=%s",
                      score, uncertain, contradiction)
        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _score_confidence(self, text: str, intent: str) -> float:
        if not text:
            return 0.4

        lower = text.lower()
        score = _BASE_SCORE

        # Penalise uncertainty language
        for phrase in _UNCERTAINTY_PHRASES:
            if phrase in lower:
                score -= 0.12
                break   # one hit is enough

        # Penalise outright refusal / lack-of-knowledge phrases
        for phrase in _REFUSAL_PHRASES:
            if phrase in lower:
                score -= 0.25
                break

        # Penalise very short responses on what looks like a knowledge-seeking intent
        knowledge_intents = {"question", "definition", "explanation", "search", "fact"}
        if any(ki in intent.lower() for ki in knowledge_intents) and len(text.split()) < 15:
            score -= 0.15

        # Reward specific, factual content (numbers, dates, proper nouns)
        fact_hits = len(_FACT_PATTERN.findall(text))
        score += min(fact_hits * 0.05, 0.15)

        return max(0.0, min(1.0, score))

    def _check_world_facts(self, response_text: str) -> tuple[bool, str]:
        """Check response against WORLD.md facts.
        Returns (contradiction_found, note_string).
        Uses simple negation-pattern heuristic — no LLM needed."""
        if not self._world_facts or not response_text:
            return False, ""

        lower_resp = response_text.lower()
        negation_prefixes = ("not ", "isn't ", "aren't ", "doesn't ", "don't ",
                              "no ", "never ", "wasn't ", "weren't ")

        for fact in self._world_facts[:20]:   # check up to 20 facts
            fact_lower = fact.lower()
            # Skip very short or vague facts
            if len(fact_lower) < 15:
                continue
            # Check if response negates a known-true fact
            key_words = [w for w in fact_lower.split() if len(w) > 4][:4]
            if not key_words:
                continue
            # If response contains key words from the fact with a negation nearby
            for kw in key_words:
                kw_pos = lower_resp.find(kw)
                if kw_pos == -1:
                    continue
                window = lower_resp[max(0, kw_pos - 30): kw_pos + 30]
                if any(neg in window for neg in negation_prefixes):
                    return True, f"Response may contradict world fact: '{fact[:60]}'"

        return False, ""

    async def _analyse_outcome(self, data: dict) -> None:
        """V8.0 — Compare observed sensory state against stored task intent."""
        intent = data.get("intent", self._current_task_intent)
        observed = data.get("observed_state", "")

        positive_words = {"blinking", "on", "active", "running", "moving", "flashing",
                          "yes", "confirmed", "detected", "visible", "complete", "working"}
        negative_words = {"off", "not", "no ", "failed", "error", "nothing", "static",
                          "dark", "unresponsive", "absent", "stopped"}

        obs_lower = observed.lower()
        pos_score = sum(1 for w in positive_words if w in obs_lower)
        neg_score = sum(1 for w in negative_words if w in obs_lower)

        if pos_score > neg_score:
            status = "success"
        elif neg_score > pos_score:
            status = "failure"
        else:
            status = "uncertain"

        total = pos_score + neg_score
        confidence = abs(pos_score - neg_score) / max(total, 1)

        log.info("MetacognitionAgent: outcome intent='%s' observed='%s' → %s (conf=%.2f)",
                 intent[:60], observed[:60], status, confidence)

        await self.publish(AgentMessage(
            type=MessageType.TASK_OUTCOME,
            source=self.name,
            data={"status": status, "intent": intent, "observed_state": observed,
                  "confidence": round(confidence, 3)},
            priority=7,
        ))

        if status == "success":
            self._current_task_intent = ""

    def _track_knowledge_gap(self, topic: str, response_text: str) -> bool:
        """Log this topic as a knowledge gap if the response admitted uncertainty.
        Returns True only when a *new* gap is added (deduplicates repeat triggers)."""
        if topic and topic not in self._knowledge_gaps:
            self._knowledge_gaps.append(topic)
            if len(self._knowledge_gaps) > 20:
                self._knowledge_gaps.pop(0)
            log.debug("MetacognitionAgent: knowledge gap logged — '%s'", topic)
            if self._soul and self._persist_knowledge_gaps:
                self._soul.update_knowledge_gaps(list(self._knowledge_gaps))
            return True
        return False

    async def _lower_memory_confidence(self, memory_hint: str, correction_text: str) -> None:
        """V9 — Log user correction; emit low-confidence event so verifier can re-check."""
        entry = {
            "at": datetime.utcnow().isoformat(),
            "memory_hint": memory_hint,
            "correction": correction_text[:120],
        }
        self._correction_ledger.append(entry)
        try:
            path = Path("data/CORRECTIONS.md")
            existing = await asyncio.to_thread(path.read_text, encoding="utf-8") if path.exists() else ""
            await asyncio.to_thread(
                path.write_text,
                existing + f"\n- [{entry['at']}] {entry['correction'][:100]}\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        log.info("MetacognitionAgent: correction logged for memory='%s'", memory_hint[:40])
        self._last_retrieved_memory_id = ""   # clear so we don't double-fire
        await self.publish(AgentMessage(
            type=MessageType.METACOG_CONFIDENCE,
            source=self.name,
            data={"score": 0.1, "uncertain": True, "reason": "user_correction"},
            priority=8,
        ))

    def set_semantic(self, semantic: "SemanticMemory") -> None:
        """Inject SemanticMemory reference for NEURO_REWEIGHT operations."""
        self._semantic = semantic

    async def _handle_neuro_reweight(self, data: dict) -> None:
        """Full reinforcement correction loop triggered by NEURO_REWEIGHT.

        1. Penalize the wrong memory (reduce confidence in ChromaDB).
        2. Upsert the corrected fact at confidence=1.0.
        3. Log to SoulManager (USER.json + CODE_HEALTH.md).
        4. Emit low-confidence signal for VerifierAgent.
        """
        doc_id          = data.get("memory_id", "")
        penalty         = float(data.get("penalty", 0.3))
        corrected_text  = data.get("corrected_text", "")
        original_query  = data.get("original_query", "")
        embedding       = data.get("embedding", [])   # pre-computed by CognitionAgent

        log.info("MetacognitionAgent: NEURO_REWEIGHT — penalizing memory id='%s'", doc_id[:20])

        # Step 1: Penalize the wrong memory
        if self._semantic and doc_id:
            self._semantic.penalize_memory(doc_id, penalty)

        # Step 2: Store the corrected fact as authoritative (confidence=1.0)
        if self._semantic and corrected_text and embedding:
            self._semantic.upsert_correction(original_query, corrected_text, embedding)

        # Step 3: Log to SoulManager
        if self._soul and corrected_text:
            self._soul.log_correction_event(corrected_text)

        # Step 4: Persist to CORRECTIONS.md
        try:
            path = Path("data/CORRECTIONS.md")
            existing = await asyncio.to_thread(path.read_text, encoding="utf-8") if path.exists() else "# Correction Log\n"
            now = datetime.utcnow().isoformat()
            await asyncio.to_thread(
                path.write_text,
                existing + f"\n- [{now[:19]}] Corrected: \"{corrected_text[:100]}\"\n",
                encoding="utf-8",
            )
        except Exception as e:
            log.debug("MetacognitionAgent: CORRECTIONS.md write error: %s", e)

        # Step 5: Broadcast low-confidence so VerifierAgent / CognitionAgent can adapt
        await self.publish(AgentMessage(
            type=MessageType.METACOG_CONFIDENCE,
            source=self.name,
            data={
                "score": 0.05,
                "uncertain": True,
                "reason": "neuro_reweight",
                "corrected_text": corrected_text[:80],
            },
            priority=8,
        ))
