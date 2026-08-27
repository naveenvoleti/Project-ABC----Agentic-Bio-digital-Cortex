"""IdeationAgent — DLPFC + Angular Gyrus + Default Mode Network mapping.

Generates novel ideas by pulling top-K semantically related concepts from
SemanticMemory and asking the LLM to synthesise a creative connection or analogy.

Triggered by:
  • IDEATION_REQUEST  — explicit request
  • CURIOSITY_TRIGGER — 30% probabilistic firing during idle exploration

Publishes IDEATION_RESULT → CognitionAgent (stored as "Creative Context" in
the system prompt, consumed once per LLM call).

60s cooldown prevents runaway LLM calls on the Pi.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import TYPE_CHECKING

from .base_agent import AgentMessage, BaseAgent, MessageType

if TYPE_CHECKING:
    from brain.llm.llm_router import LLMRouter
    from brain.memory.episodic_memory import EpisodicMemory
    from brain.memory.semantic_memory import SemanticMemory

log = logging.getLogger(__name__)

_COOLDOWN_S       = 60.0    # min seconds between ideation LLM calls
_SEMANTIC_K       = 5       # top-K neighbors to seed cross-domain synthesis
_CURIOSITY_PROB   = 0.30    # probability of firing on CURIOSITY_TRIGGER


class IdeationAgent(BaseAgent):
    """DLPFC + Angular Gyrus — cross-domain idea synthesis and analogy generation."""

    name = "ideation_agent"

    def __init__(self, bus: asyncio.Queue, llm: "LLMRouter",
                 semantic: "SemanticMemory", episodic: "EpisodicMemory"):
        super().__init__(bus)
        self._llm      = llm
        self._semantic = semantic
        self._episodic = episodic
        self._last_ideation_ts: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("IdeationAgent: started — cooldown=%.0fs, curiosity_prob=%.0f%%",
                 _COOLDOWN_S, _CURIOSITY_PROB * 100)
        while self._running:
            await asyncio.sleep(1)

    # ── Message handling ─────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        if message.type == MessageType.IDEATION_REQUEST:
            seed = message.data.get("topic") or message.data.get("text", "")
            asyncio.create_task(self._synthesise(seed), name="ideation-synth")

        elif message.type == MessageType.CURIOSITY_TRIGGER:
            if random.random() < _CURIOSITY_PROB:
                # Pick seed from recent episodic memory
                recent = self._episodic.get_salient(n=3)
                seed = recent[0]["content"][:100] if recent else ""
                if seed:
                    asyncio.create_task(self._synthesise(seed), name="ideation-curiosity")

        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _is_cooled_down(self) -> bool:
        return (time.time() - self._last_ideation_ts) >= _COOLDOWN_S

    async def _synthesise(self, seed: str) -> None:
        if not seed or not self._is_cooled_down():
            return
        self._last_ideation_ts = time.time()

        try:
            # 1. Embed seed and retrieve semantic neighbours
            embedding = await self._llm.embed(seed[:200])
            if embedding is None:
                return
            neighbours = self._semantic.search_similar(embedding, k=_SEMANTIC_K)
            if not neighbours:
                return

            # 2. Build cross-domain synthesis prompt
            concepts = [n["content"][:80] for n in neighbours]
            concepts_str = "; ".join(f"[{i+1}] {c}" for i, c in enumerate(concepts))
            prompt = (
                f"Seed topic: \"{seed[:150]}\"\n"
                f"Related concepts from memory: {concepts_str}\n\n"
                "Generate ONE novel, creative connection or analogy linking the seed "
                "to one or more of these concepts. Be imaginative but grounded. "
                "One sentence only."
            )
            idea = await self._llm.infer(
                user_message=prompt,
                system_prompt="You are a creative synthesiser. Produce one original idea sentence.",
                max_tokens=80,
            )

            if idea and idea.strip():
                await self.publish(AgentMessage(
                    type=MessageType.IDEATION_RESULT,
                    source=self.name,
                    data={
                        "idea":       idea.strip(),
                        "seed_topic": seed[:100],
                        "ts":         time.time(),
                    },
                    priority=8,
                ))
                log.debug("IdeationAgent: idea generated — '%s'", idea[:60])

        except Exception as exc:
            log.debug("IdeationAgent: synthesis failed — %s", exc)

    async def _generate_analogy(self, concept_a: str, concept_b: str) -> str | None:
        """Utility: generate an analogy between two concepts (callable externally)."""
        try:
            prompt = (
                f"What does \"{concept_a}\" have in common with \"{concept_b}\"? "
                "Generate a creative analogy in one sentence."
            )
            return await self._llm.infer(
                user_message=prompt,
                system_prompt="You are a creative analogist.",
                max_tokens=60,
            )
        except Exception:
            return None
