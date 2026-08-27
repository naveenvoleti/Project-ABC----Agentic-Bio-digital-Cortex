"""ImaginationAgent — Hippocampus + PFC (forward projection) mapping.

Performs mental simulation: given a "what if" scenario, builds a plausible
future using recent episodic context and LLM inference.

Triggered by:
  • IMAGINATION_SIMULATE  — explicit request (e.g. from CognitionAgent)
  • PERCEPTION_SPEECH     — auto-detection of "what if" phrasing in user speech

v5.0 addition — Predictive Processing loop:
  • Every _PREDICT_INTERVAL_S seconds, proactively predicts the next world state
    from recent episodic context and publishes IMAGINATION_SIMULATE with
    "predicted_scene" field. VisionAgent compares the real scene against this
    prediction and fires WORLD_SURPRISE if the delta exceeds 30%.

120s cooldown for on-demand "what if" simulation. Proactive prediction uses a
separate 30s interval and lighter prompts.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING

from .base_agent import AgentMessage, BaseAgent, MessageType

if TYPE_CHECKING:
    from brain.llm.llm_router import LLMRouter
    from brain.memory.episodic_memory import EpisodicMemory
    from brain.memory.working_memory import WorkingMemory

log = logging.getLogger(__name__)

_COOLDOWN_S        = 120.0  # 2-minute cooldown for on-demand "what if" sims
_PREDICT_INTERVAL_S = 30.0  # proactive world-state prediction every 30s
_WHAT_IF_PATTERN = re.compile(
    r'\b(what if|what would happen if|imagine if|suppose|hypothetically|'
    r"let'?s? say|pretend that|what happens when|if we|if you)\b",
    re.IGNORECASE,
)


class ImaginationAgent(BaseAgent):
    """Hippocampus + PFC (forward) — episodic future thinking and mental simulation."""

    name = "imagination_agent"

    def __init__(self, bus: asyncio.Queue, llm: "LLMRouter",
                 episodic: "EpisodicMemory", working: "WorkingMemory"):
        super().__init__(bus)
        self._llm      = llm
        self._episodic = episodic
        self._working  = working
        self._last_sim_ts:     float = 0.0
        self._last_predict_ts: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("ImaginationAgent: started — cooldown=%.0fs, predict_interval=%.0fs",
                 _COOLDOWN_S, _PREDICT_INTERVAL_S)
        asyncio.create_task(self._prediction_loop(), name="imagination-predict")
        while self._running:
            await asyncio.sleep(1)

    # ── Message handling ─────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        if message.type == MessageType.IMAGINATION_SIMULATE:
            scenario = message.data.get("scenario") or message.data.get("text", "")
            asyncio.create_task(
                self._simulate(scenario, source_msg=message),
                name="imagination-sim",
            )

        elif message.type == MessageType.PERCEPTION_SPEECH:
            text = message.data.get("text", "")
            if text and _WHAT_IF_PATTERN.search(text):
                asyncio.create_task(
                    self._simulate(text, source_msg=message),
                    name="imagination-whatif",
                )

        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _is_cooled_down(self) -> bool:
        return (time.time() - self._last_sim_ts) >= _COOLDOWN_S

    async def _simulate(self, scenario: str, source_msg: AgentMessage) -> None:
        if not scenario or not self._is_cooled_down():
            return
        self._last_sim_ts = time.time()

        try:
            # Build context from recent episodic memory
            recent = self._episodic.get_salient(n=5)
            context_str = "\n".join(
                f"- {e['content'][:100]}" for e in recent
            ) if recent else "No recent context."

            prompt = (
                f"Recent context:\n{context_str}\n\n"
                f"Scenario to simulate: \"{scenario[:300]}\"\n\n"
                "Simulate what would plausibly happen. "
                "Describe 2-3 concise consequences or outcomes. "
                "Be specific and grounded in the context above. "
                "Max 3 sentences total."
            )
            result = await self._llm.infer(
                user_message=prompt,
                system_prompt=(
                    "You are a mental simulation engine. "
                    "Reason about plausible futures based on context. Be brief."
                ),
                max_tokens=150,
            )

            if result and result.strip():
                await self.publish(AgentMessage(
                    type=MessageType.IMAGINATION_SIMULATE,
                    source=self.name,
                    data={
                        "scenario":    scenario[:200],
                        "simulation":  result.strip(),
                        "ts":          time.time(),
                    },
                    priority=7,
                ))
                log.debug("ImaginationAgent: simulation complete — '%s'", result[:60])

        except Exception as exc:
            log.debug("ImaginationAgent: simulation failed — %s", exc)

    async def _prediction_loop(self) -> None:
        """v5.0 Predictive Processing — proactively predict next world state every 30s."""
        while self._running:
            await asyncio.sleep(_PREDICT_INTERVAL_S)
            await self._predict_world_state()

    async def _predict_world_state(self) -> None:
        """Generate a short prediction of the next scene state from recent context.
        Published as IMAGINATION_SIMULATE with 'predicted_scene' so VisionAgent can
        compare and fire WORLD_SURPRISE when the real scene diverges by >30%."""
        try:
            recent = self._episodic.get_salient(n=4)
            if not recent:
                return

            context_str = "\n".join(
                f"- {e['content'][:80]}" for e in recent
            )
            prompt = (
                f"Recent events:\n{context_str}\n\n"
                "In one sentence, describe what the scene/situation will most likely "
                "look like in the next 30 seconds. Be very brief and specific."
            )
            predicted = await self._llm.infer(
                user_message=prompt,
                system_prompt=(
                    "You are a world-state predictor. Given recent context, predict "
                    "the immediate next state in one concise sentence."
                ),
                max_tokens=60,
            )
            if predicted and predicted.strip():
                predicted = predicted.strip()
                await self.publish(AgentMessage(
                    type=MessageType.IMAGINATION_SIMULATE,
                    source=self.name,
                    data={
                        "scenario":        "proactive_prediction",
                        "simulation":      predicted,
                        "predicted_scene": predicted,
                        "ts":              time.time(),
                    },
                    priority=8,  # low priority — background prediction
                ))
                log.debug("ImaginationAgent: world prediction → '%s'", predicted[:60])
        except Exception as exc:
            log.debug("ImaginationAgent: world prediction failed — %s", exc)

    async def _predict_consequence(self, action: str) -> str | None:
        """Pre-action check: what are likely user reactions to this action?
        Callable by CognitionAgent for consequential responses."""
        try:
            prompt = (
                f"If I say: \"{action[:200]}\", "
                "what are 1-2 likely user reactions? "
                "Be brief — one sentence."
            )
            return await self._llm.infer(
                user_message=prompt,
                system_prompt="You predict social consequences of statements. Be concise.",
                max_tokens=60,
            )
        except Exception:
            return None
