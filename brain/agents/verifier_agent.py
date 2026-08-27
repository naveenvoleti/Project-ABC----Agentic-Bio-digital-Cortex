"""
VerifierAgent — output validation, hallucination guard, and uncertainty flagging.
Maps to the Anterior Cingulate Cortex.
Cross-checks responses before they reach the user.

Production changes:
- Inline confidence scoring (no race condition dependency on MetacognitionAgent)
- MetacognitionAgent METACOG_CONFIDENCE acts as a secondary boost signal
- Bounded _recheck_count with TTL cleanup to prevent memory leak
"""
from __future__ import annotations

import asyncio
import collections
import re
import time

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.utils.logger import get_logger

log = get_logger(__name__)

HALLUCINATION_PATTERNS = [
    r"\b(I (am|was) born|I have a body|I (ate|eat|sleep|dream))\b",
    r"\b(my (mother|father|parents|family|house|car))\b",
    r"\b(I (went to|attended|graduated from) (school|university|college))\b",
]

MAX_RESPONSE_LENGTH = 1000
RECHECK_TTL = 120.0   # seconds before a retry-key is expired

_UNCERTAINTY_PHRASES = frozenset({
    "i think", "i believe", "probably", "maybe", "perhaps",
    "not sure", "i'm not certain", "i don't know", "uncertain",
    "approximately", "i'm not sure", "it's possible", "might be",
    "i cannot say", "i'm unsure",
})
_REFUSAL_PHRASES = frozenset({
    "i cannot", "i don't have access", "i'm unable",
    "i have no information", "i have no data",
})


def _score_confidence(text: str) -> float:
    """Quick heuristic confidence score 0.0–1.0. No LLM needed."""
    if not text:
        return 0.4
    lower = text.lower()
    score = 0.85
    for phrase in _UNCERTAINTY_PHRASES:
        if phrase in lower:
            score -= 0.18
            break
    for phrase in _REFUSAL_PHRASES:
        if phrase in lower:
            score -= 0.30
            break
    # Very short response to what looks like a content query
    if len(text.split()) < 12:
        score -= 0.10
    return max(0.0, min(1.0, score))


class VerifierAgent(BaseAgent):
    name = "verifier_agent"

    def __init__(self, bus: asyncio.Queue):
        super().__init__(bus)
        # Bounded recheck tracker: key → (count, last_seen_ts)
        self._recheck: dict[str, tuple[int, float]] = {}
        # Secondary confidence signal from MetacognitionAgent (previous cycle)
        self._metacog_uncertain: bool = False
        # Semantic consistency signals from MetacognitionAgent
        self._world_contradiction: bool = False
        self._contradiction_note: str = ""
        # N3d: Internal monologue consistency — store last private thought
        self._last_thought: str = ""

    async def start(self) -> None:
        await super().start()
        log.info("VerifierAgent started")
        asyncio.create_task(self._cleanup_loop(), name="verifier-cleanup")
        while self._running:
            await asyncio.sleep(1)

    async def _cleanup_loop(self) -> None:
        """Periodically expire stale recheck keys so the dict doesn't leak."""
        while self._running:
            await asyncio.sleep(60)
            now = time.time()
            stale = [k for k, (_, ts) in self._recheck.items() if now - ts > RECHECK_TTL]
            for k in stale:
                self._recheck.pop(k, None)

    def verify(self, response: str) -> tuple[bool, str]:
        # Length guard
        if len(response) > MAX_RESPONSE_LENGTH:
            response = response[:MAX_RESPONSE_LENGTH] + "..."

        # Hallucination check
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                return False, f"Possible hallucination: {pattern}"

        # Empty / error check
        if not response.strip():
            return False, "Empty response"

        # Repetition check
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False, "Response appears repetitive"

        return True, response

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.METACOG_CONFIDENCE:
            # Secondary signal: store for next response cycle
            self._metacog_uncertain = message.data.get("uncertain", False)
            self._world_contradiction = message.data.get("world_contradiction", False)
            self._contradiction_note = message.data.get("contradiction_note", "")
            return

        if message.type == MessageType.COGNITION_THOUGHT:
            self._last_thought = message.data.get("thought", "")
            return

        if message.type != MessageType.COGNITION_RESPONSE:
            return

        response = message.data.get("response", "")
        original_text = message.data.get("original_text", "")

        ok, result = self.verify(response)

        if ok:
            # Inline confidence scoring — self-contained, no race condition
            conf = _score_confidence(result)
            # Also honour secondary MetacognitionAgent signal from previous cycle
            uncertain = conf < 0.35 or self._metacog_uncertain
            self._metacog_uncertain = False  # consume

            # Semantic consistency hedge: if MetacognitionAgent flagged a WORLD.md
            # contradiction, prepend an honest epistemic disclaimer before uncertainty prefix.
            world_contra = self._world_contradiction
            self._world_contradiction = False  # consume
            self._contradiction_note = ""

            if world_contra:
                result = "I can only confirm what I know for certain — " + result
                log.info("VerifierAgent: world-contradiction hedge applied")
            elif uncertain:
                result = "I'm not fully certain, but: " + result
                log.debug("VerifierAgent: uncertainty prefix applied (conf=%.2f)", conf)

            # N3d: Thought-response consistency — enforce what the private thought intended.
            thought = self._last_thought
            self._last_thought = ""  # consume
            if thought:
                thought_lower = thought.lower()
                # Brevity enforcement: thought said keep it brief but response is long
                if any(w in thought_lower for w in ("brief", "short", "concise", "quick")) and len(result) > 200:
                    result = result[:200].rsplit(" ", 1)[0] + "..."
                    log.debug("VerifierAgent: brevity cap applied (thought-consistency)")
                # Empathy enforcement: thought said empathetic but no softeners present
                elif any(w in thought_lower for w in ("empathetic", "warm", "supportive", "gentle")):
                    softeners = ("i understand", "i see", "that makes sense", "i hear you", "of course")
                    if not any(s in result.lower() for s in softeners):
                        result = "I understand — " + result
                        log.debug("VerifierAgent: empathy prefix applied (thought-consistency)")

            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={
                    "text": result,
                    "original_query": original_text,
                    "verified": True,
                    "success": True,
                    "confidence": round(conf, 3),
                },
                priority=2,
            ))
        else:
            key = original_text[:50]
            count, _ = self._recheck.get(key, (0, 0.0))
            count += 1
            self._recheck[key] = (count, time.time())

            log.warning("VerifierAgent rejected: %s (attempt %d)", result, count)

            if count >= 3:
                self._recheck.pop(key, None)
                await self.publish(AgentMessage(
                    type=MessageType.ACTION_SPEAK,
                    source=self.name,
                    data={
                        "text": "I'm having a bit of trouble with that. Could you rephrase?",
                        "verified": False,
                        "success": False,
                    },
                    priority=2,
                ))
            else:
                await self.publish(AgentMessage(
                    type=MessageType.COGNITION_INTENT,
                    source=self.name,
                    target="reasoning_agent",
                    data={
                        "text": original_text,
                        "retry": True,
                        "attempt": count,
                    },
                    priority=3,
                ))
