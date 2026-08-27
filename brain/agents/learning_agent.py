"""
LearningAgent — Hippocampus + Prefrontal Cortex (learning) mapping. (V10.0)

Self-directed learning: detects knowledge gaps in Brain's semantic memory and
autonomously fills them using web search (DuckDuckGo MCP tool) or LLM synthesis.

Flow:
  1. Subscribes to CURIOSITY_TRIGGER (with topic) and GOAL_PUSH (type='learn')
  2. Checks if SemanticMemory already has relevant content
  3. If gap confirmed: searches DuckDuckGo via MCPToolRegistry → processes → stores
  4. Publishes MEMORY_WRITE with new semantic fact + COGNITION_RESPONSE if standalone
  5. Publishes TASK_OUTCOME so GoalStackAgent can mark the goal complete

Gap detection also runs on COGNITION_RESPONSE failures (failed LLM inference)
to add the failed topic to the learning queue for next idle cycle.

Rate limiting:
  - Max 3 learning tasks per hour (configurable)
  - 60s minimum between tasks
  - Only triggers when human has been silent for 20s
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import TYPE_CHECKING

from .base_agent import AgentMessage, BaseAgent, MessageType

if TYPE_CHECKING:
    from brain.llm.llm_router import LLMRouter
    from brain.llm.mcp_tool_registry import MCPToolRegistry
    from brain.memory.semantic_memory import SemanticMemory

log = logging.getLogger(__name__)

_HUMAN_SILENCE_S     = 20.0   # wait this long after last human speech
_MIN_TASK_INTERVAL_S = 60.0   # minimum seconds between learning tasks
_MAX_TASKS_PER_HOUR  = 3      # rate limit


class LearningAgent(BaseAgent):
    """Hippocampus + PFC — autonomous self-directed knowledge gap filling.

    Monitors curiosity signals and failed queries, then fills gaps using
    web search or LLM synthesis during human silence periods.
    """

    name = "learning_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        llm: "LLMRouter",
        semantic: "SemanticMemory",
        mcp: "MCPToolRegistry | None" = None,
    ):
        super().__init__(bus)
        self._llm      = llm
        self._semantic = semantic
        self._mcp      = mcp

        # Learning queue: deque of (topic, priority, source)
        self._queue: collections.deque[tuple[str, float, str]] = collections.deque(maxlen=20)
        self._active_topic: str | None = None
        self._last_human_ts:   float = 0.0
        self._last_task_ts:    float = 0.0
        self._tasks_this_hour: int   = 0
        self._hour_bucket:     int   = -1
        self._completed_topics: set[str] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("LearningAgent: started (max %d tasks/hour, silence_threshold=%.0fs)",
                 _MAX_TASKS_PER_HOUR, _HUMAN_SILENCE_S)
        asyncio.create_task(self._learning_loop(), name="learning-loop")
        while self._running:
            await asyncio.sleep(1)

    # ── Message handling ──────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        mtype = message.type

        if mtype == MessageType.PERCEPTION_SPEECH:
            self._last_human_ts = time.time()

        elif mtype == MessageType.CURIOSITY_TRIGGER:
            topic = message.data.get("topic", "")
            if topic and topic not in self._completed_topics:
                self._enqueue(topic, priority=0.7, source="curiosity")

        elif mtype == MessageType.GOAL_PUSH:
            goal = message.data.get("goal", {})
            if goal.get("type") == "learn":
                topic = goal.get("topic", "")
                if topic and topic not in self._completed_topics:
                    self._enqueue(topic, priority=0.9, source="goal_stack")

        elif mtype == MessageType.COGNITION_RESPONSE:
            # If inference failed on a topic, add to learning queue
            if not message.data.get("success", True):
                failed_text = message.data.get("original_text", "")
                if failed_text and len(failed_text) > 5:
                    self._enqueue(failed_text[:80], priority=0.5, source="failure")

        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _enqueue(self, topic: str, priority: float, source: str) -> None:
        """Add a topic to the learning queue (no duplicates, sorted by priority)."""
        # Check for near-duplicates (simple prefix match)
        topic_lower = topic.lower().strip()
        for queued_topic, _, _ in self._queue:
            if queued_topic.lower().startswith(topic_lower[:30]):
                log.debug("LearningAgent: topic already queued: '%s'", topic[:40])
                return
        self._queue.append((topic, priority, source))
        log.info("LearningAgent: queued topic '%s' (priority=%.2f, source=%s)",
                 topic[:40], priority, source)

    async def _learning_loop(self) -> None:
        """Main autonomous learning loop — runs every 30s."""
        while self._running:
            await asyncio.sleep(30)
            await self._maybe_learn()

    async def _maybe_learn(self) -> None:
        """Attempt one learning task if conditions allow."""
        now = time.time()

        # Guard: human recently spoke
        if (now - self._last_human_ts) < _HUMAN_SILENCE_S:
            return

        # Guard: rate limit — hours reset
        current_hour = int(now / 3600)
        if current_hour != self._hour_bucket:
            self._hour_bucket = current_hour
            self._tasks_this_hour = 0

        if self._tasks_this_hour >= _MAX_TASKS_PER_HOUR:
            return

        # Guard: minimum interval
        if (now - self._last_task_ts) < _MIN_TASK_INTERVAL_S:
            return

        # Guard: no active task, queue non-empty
        if self._active_topic or not self._queue:
            return

        # Pop highest-priority topic
        best_topic, best_priority, source = max(self._queue, key=lambda x: x[1])
        self._queue.remove((best_topic, best_priority, source))
        self._active_topic = best_topic
        self._last_task_ts = now
        self._tasks_this_hour += 1

        log.info("LearningAgent: starting autonomous learning — topic='%s' (source=%s)",
                 best_topic[:60], source)

        try:
            await self._fill_gap(best_topic, source)
        except Exception as e:
            log.warning("LearningAgent: learning task failed for '%s': %s", best_topic[:40], e)
        finally:
            self._completed_topics.add(best_topic)
            self._active_topic = None

    async def _fill_gap(self, topic: str, source: str) -> None:
        """Core knowledge gap filling: search → synthesize → store."""

        # Step 1: Check if SemanticMemory already has relevant content
        try:
            existing = self._semantic.search(topic, limit=3) if hasattr(self._semantic, "search") else []
        except Exception:
            existing = []

        if existing:
            # If we have 3+ good hits, the gap might already be filled
            high_quality = [r for r in existing if r.get("relevance", 0) > 0.7]
            if len(high_quality) >= 2:
                log.info("LearningAgent: gap '%s' already filled in semantic memory", topic[:40])
                await self._publish_outcome(topic, success=True, learned_from="memory")
                return

        # Step 2: Try web search via MCP (DuckDuckGo) if available
        search_result = ""
        if self._mcp and hasattr(self._mcp, "dispatch"):
            try:
                search_result = await self._mcp.dispatch("web_search", {"query": topic, "limit": 5})
                if search_result and isinstance(search_result, dict):
                    search_result = str(search_result.get("content", ""))[:800]
                elif isinstance(search_result, str):
                    search_result = search_result[:800]
                if search_result:
                    log.info("LearningAgent: web search returned %d chars for '%s'",
                             len(search_result), topic[:40])
            except Exception as e:
                log.debug("LearningAgent: MCP web search failed: %s", e)

        # Step 3: LLM synthesis — from search results or from parametric knowledge
        prompt_context = f"Search results:\n{search_result}\n\n" if search_result else ""
        try:
            synthesis = await self._llm.infer(
                user_message=f"Topic to learn about: {topic}",
                system_prompt=(
                    f"You are Brain, filling a knowledge gap about: \"{topic}\".\n"
                    f"{prompt_context}"
                    "Synthesize a clear, factual summary (3-5 sentences) suitable for "
                    "storing as a long-term memory. Include specific facts, not vague generalities. "
                    "Start with the most important fact."
                ),
                max_tokens=200,
            )
        except Exception as e:
            log.warning("LearningAgent: LLM synthesis failed for '%s': %s", topic[:40], e)
            synthesis = ""

        if not synthesis or len(synthesis.strip()) < 30:
            await self._publish_outcome(topic, success=False, learned_from="none")
            return

        # Step 4: Store in SemanticMemory
        synthesis_clean = synthesis.strip()
        try:
            if hasattr(self._semantic, "upsert"):
                self._semantic.upsert(
                    content=synthesis_clean,
                    metadata={"topic": topic, "source": source, "method": "autonomous_learning",
                               "ts": time.time()},
                )
            log.info("LearningAgent: stored %d chars about '%s' in semantic memory",
                     len(synthesis_clean), topic[:40])
        except Exception as e:
            log.warning("LearningAgent: semantic memory storage failed: %s", e)

        await self._publish_outcome(topic, success=True, learned_from="synthesis",
                                    content=synthesis_clean)

    async def _publish_outcome(
        self, topic: str, success: bool, learned_from: str, content: str = ""
    ) -> None:
        """Publish TASK_OUTCOME so GoalStackAgent can complete the goal."""
        await self.publish(AgentMessage(
            type=MessageType.TASK_OUTCOME,
            source=self.name,
            data={
                "success":      success,
                "topic":        topic,
                "learned_from": learned_from,
                "content":      content[:200],
                "ts":           time.time(),
            },
            priority=6,
        ))
