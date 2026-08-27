"""TheoryOfMindAgent — Medial PFC + Temporal Poles mapping.

Models the user's mental state: expertise level, current confusion,
frustration signals, and inferred goals. Updates are heuristic on every
message and LLM-assisted every N interactions.

Publishes USER_MODEL_UPDATE → CognitionAgent enriches its system prompt
with adaptive explanation depth.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING

from .base_agent import AgentMessage, BaseAgent, MessageType

if TYPE_CHECKING:
    from brain.llm.llm_router import LLMRouter
    from brain.memory.soul_manager import SoulManager

log = logging.getLogger(__name__)

_CONFUSION_WORDS = frozenset({
    "what do you mean", "don't understand", "confused", "can you explain",
    "clarify", "huh", "what?", "i'm lost", "not following",
    "explain again", "say that again", "what does that mean",
})
_FRUSTRATION_WORDS = frozenset({
    "that's wrong", "no that's not", "stop", "ugh", "why can't you",
    "you're wrong", "incorrect", "that doesn't make sense", "nonsense",
    "what the", "come on",
})
_TECH_VOCAB = re.compile(
    r'\b(API|SDK|async|await|docker|kubernetes|neural|algorithm|'
    r'recursive|polymorphism|bytecode|compiler|hyperparameter|tensor|'
    r'git|regex|lambda|coroutine|mutex|semaphore)\b',
    re.IGNORECASE,
)

_UPDATE_EVERY_N = 5   # LLM-assisted update every N interactions


class TheoryOfMindAgent(BaseAgent):
    """Medial PFC — models user's beliefs, expertise, and emotional state."""

    name = "theory_of_mind_agent"

    def __init__(self, bus: asyncio.Queue, llm: "LLMRouter",
                 soul: "SoulManager"):
        super().__init__(bus)
        self._llm  = llm
        self._soul = soul
        self._user_model: dict = {
            "expertise_level":    "medium",   # low | medium | high
            "confusion_score":    0.0,         # 0.0 – 1.0
            "frustration_score":  0.0,
            "explanation_depth":  "normal",    # simple | normal | technical
            "inferred_goals":     [],
            "interaction_count":  0,
        }
        self._interaction_count = 0
        self._recent_texts: list[str] = []    # last 5 user utterances for LLM context

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("TheoryOfMindAgent: started — LLM update every %d interactions", _UPDATE_EVERY_N)
        while self._running:
            await asyncio.sleep(1)

    # ── Message handling ─────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        if message.type == MessageType.PERCEPTION_SPEECH:
            text = message.data.get("text", "").strip()
            if not text:
                return None

            self._interaction_count += 1
            self._recent_texts.append(text[:200])
            if len(self._recent_texts) > 5:
                self._recent_texts.pop(0)

            # Heuristic update (instant, no LLM)
            self._update_heuristic(text)

            # LLM-assisted update every N interactions
            if self._interaction_count % _UPDATE_EVERY_N == 0:
                asyncio.create_task(
                    self._llm_update_model(),
                    name="tom-llm-update",
                )

            await self._publish_model()  # debounced internally

        elif message.type == MessageType.COGNITION_RESPONSE:
            # Very short response after long user input → possible frustration signal
            response_len = len(message.data.get("text", "").split())
            if response_len < 10 and self._user_model["confusion_score"] > 0.5:
                self._user_model["frustration_score"] = min(
                    self._user_model["frustration_score"] + 0.1, 1.0
                )

        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update_heuristic(self, text: str) -> None:
        lower = text.lower()

        # Confusion detection
        confusion = 0.0
        for phrase in _CONFUSION_WORDS:
            if phrase in lower:
                confusion = 0.7
                break
        # Decay confusion if no signal
        if confusion > 0:
            self._user_model["confusion_score"] = min(
                self._user_model["confusion_score"] + confusion, 1.0
            )
        else:
            self._user_model["confusion_score"] = max(
                self._user_model["confusion_score"] - 0.1, 0.0
            )

        # Frustration detection
        frustration = 0.0
        for phrase in _FRUSTRATION_WORDS:
            if phrase in lower:
                frustration = 0.6
                break
        if frustration > 0:
            self._user_model["frustration_score"] = min(
                self._user_model["frustration_score"] + frustration, 1.0
            )
        else:
            self._user_model["frustration_score"] = max(
                self._user_model["frustration_score"] - 0.05, 0.0
            )

        # Expertise inference from vocabulary
        tech_hits = len(_TECH_VOCAB.findall(text))
        if tech_hits >= 2:
            self._user_model["expertise_level"] = "high"
        elif tech_hits == 0 and len(text.split()) < 8:
            # Short, simple sentences with no tech vocab → lean toward low
            current = self._user_model["expertise_level"]
            if current == "high":
                self._user_model["expertise_level"] = "medium"

        # Map confusion → explanation depth
        conf = self._user_model["confusion_score"]
        exp  = self._user_model["expertise_level"]
        if conf > 0.6:
            self._user_model["explanation_depth"] = "simple"
        elif exp == "high" and conf < 0.2:
            self._user_model["explanation_depth"] = "technical"
        else:
            self._user_model["explanation_depth"] = "normal"

    async def _llm_update_model(self) -> None:
        try:
            context = "\n".join(f"- {t}" for t in self._recent_texts)
            prompt = (
                f"Based on these recent user messages:\n{context}\n\n"
                "Infer in JSON: {\"expertise\": \"low|medium|high\", "
                "\"main_goal\": \"one sentence\", "
                "\"confusion\": 0.0-1.0}. "
                "Reply ONLY with the JSON object, nothing else."
            )
            result = await self._llm.infer(
                user_message=prompt,
                system_prompt="You are a user psychology analyst. Reply only with JSON.",
                max_tokens=80,
            )
            if result:
                # Strip markdown code fences if present
                clean = result.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                parsed = json.loads(clean)
                if "expertise" in parsed:
                    self._user_model["expertise_level"] = parsed["expertise"]
                if "main_goal" in parsed:
                    goals = self._user_model["inferred_goals"]
                    goal  = parsed["main_goal"]
                    if goal not in goals:
                        goals.append(goal)
                        if len(goals) > 5:
                            goals.pop(0)
                if "confusion" in parsed:
                    self._user_model["confusion_score"] = max(0.0, min(1.0, float(parsed["confusion"])))
                # Persist expertise to soul — correct two-arg signature
                await self._soul.upsert_user_fact(
                    "expertise", self._user_model["expertise_level"]
                )
                await self._publish_model(force=True)
                log.debug("TheoryOfMindAgent: LLM model update complete")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.debug("TheoryOfMindAgent: LLM JSON parse failed — %s", exc)
        except Exception as exc:
            log.debug("TheoryOfMindAgent: LLM update failed — %s", exc)

    async def _publish_model(self, force: bool = False) -> None:
        """Publish USER_MODEL_UPDATE only when the model has meaningfully changed.
        Debounced: skips publish if expertise+depth+confusion haven't shifted."""
        self._user_model["interaction_count"] = self._interaction_count
        # Debounce: only publish when key fields changed or forced (LLM update)
        sig = (
            self._user_model["expertise_level"],
            self._user_model["explanation_depth"],
            round(self._user_model["confusion_score"], 1),
            round(self._user_model["frustration_score"], 1),
        )
        if not force and sig == getattr(self, "_last_sig", None):
            return
        self._last_sig = sig  # type: ignore[attr-defined]
        await self.publish(AgentMessage(
            type=MessageType.USER_MODEL_UPDATE,
            source=self.name,
            data=dict(self._user_model, ts=time.time()),
            priority=6,
        ))
