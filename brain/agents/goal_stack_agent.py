"""
GoalStackAgent — Anterior Cingulate Cortex + Orbitofrontal Cortex mapping. (V10.0)

Converts drive signals from IntrinsicMotivationAgent into concrete, autonomous goals.
Maintains a priority-weighted goal stack and feeds top goals to PlannerAgent
when no human interaction is active.

This is the bridge between "I feel curious" and "I will now do something about it."

Drive → Goal mapping:
  learn   + knowledge_gaps  →  Autonomous learning task (→ LearningAgent)
  explore + idle            →  Scene scan + spatial exploration (→ VisionAgent)
  social  + person_present  →  Proactive conversation initiation (→ CognitionAgent)
  help    + active_goal     →  Offer unsolicited assistance (→ CognitionAgent)

Autonomy Guard:
  If PERCEPTION_SPEECH received within last 30s → human is talking → NO autonomous goals.
  Human input always takes absolute priority.
"""
from __future__ import annotations

import asyncio
import heapq
import logging
import time
from typing import Any

from .base_agent import AgentMessage, BaseAgent, MessageType

log = logging.getLogger(__name__)

# Seconds without human input before autonomous goals are dispatched
_HUMAN_SILENCE_THRESHOLD_S = 30.0
# Minimum seconds between autonomous goal dispatches
_DISPATCH_INTERVAL_S = 60.0
# Maximum goals on the stack at once
_MAX_STACK_SIZE = 10

# Drive thresholds to trigger goal synthesis
_LEARN_THRESHOLD   = 0.45
_EXPLORE_THRESHOLD = 0.35
_SOCIAL_THRESHOLD  = 0.55
_HELP_THRESHOLD    = 0.45


class GoalStackAgent(BaseAgent):
    """ACC + OFC — autonomous goal generation and prioritization.

    Bridges the gap between internal drive states and concrete executed behaviors.
    Only dispatches goals during human silence to avoid interrupting conversation.
    """

    name = "goal_stack_agent"

    def __init__(self, bus: asyncio.Queue):
        super().__init__(bus)
        # Min-heap: (negative_priority, timestamp, goal_dict)
        # Using negative priority so heapq (min-heap) returns highest priority first.
        self._goal_heap: list[tuple[float, float, dict]] = []
        self._last_human_input_ts: float = 0.0
        self._last_dispatch_ts: float = 0.0
        self._active_goal: dict | None = None
        self._current_drives: dict[str, float] = {}
        self._current_homeo: dict[str, float] = {}
        self._knowledge_gaps: list[str] = []
        self._person_present: bool = False
        self._dispatched_goals: set[str] = set()  # prevent duplicate dispatches

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("GoalStackAgent: started — silence_threshold=%.0fs dispatch_interval=%.0fs",
                 _HUMAN_SILENCE_THRESHOLD_S, _DISPATCH_INTERVAL_S)
        asyncio.create_task(self._autonomous_loop(), name="goalstack-loop")
        while self._running:
            await asyncio.sleep(1)

    # ── Message handling ──────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        mtype = message.type

        if mtype == MessageType.PERCEPTION_SPEECH:
            # Human is talking — reset the silence timer
            self._last_human_input_ts = time.time()
            log.debug("GoalStackAgent: human input detected — autonomy paused")

        elif mtype == MessageType.MOTIVATION_DRIVE:
            # Receive latest drive state from IntrinsicMotivationAgent
            self._current_drives = message.data.get("drives", {})
            self._current_homeo  = message.data.get("homeostatic", {})
            self._knowledge_gaps = message.data.get("knowledge_gaps", [])
            self._person_present = message.data.get("person_present", False)
            # Try to synthesize new goals from updated drives
            await self._synthesize_goals()

        elif mtype == MessageType.PERSON_ENTER:
            self._person_present = True

        elif mtype == MessageType.PERSON_LEAVE:
            self._person_present = False

        elif mtype == MessageType.TASK_OUTCOME:
            # An autonomous goal finished — clear it and emit GOAL_COMPLETE
            if self._active_goal:
                success = message.data.get("success", False)
                log.info("GoalStackAgent: goal '%s' completed (success=%s)",
                         self._active_goal.get("id", "?"), success)
                await self.publish(AgentMessage(
                    type=MessageType.GOAL_COMPLETE,
                    source=self.name,
                    data={
                        "goal":    self._active_goal,
                        "success": success,
                        "ts":      time.time(),
                    },
                    priority=6,
                ))
                self._active_goal = None

        elif mtype == MessageType.PLAN_CANCEL:
            # User cancelled — clear active goal
            self._active_goal = None

        elif mtype == MessageType.WORLD_SURPRISE:
            # Surprising event → generate an "investigate" goal with high priority
            source = message.data.get("source", "unknown")
            description = message.data.get("description", "unexpected event")
            goal_id = f"investigate:{source}:{int(time.time())}"
            self._push_goal({
                "id":       goal_id,
                "type":     "investigate",
                "action":   "vlm_scan",
                "priority": 0.85,   # high — surprise warrants immediate attention
                "message":  f"Something unexpected happened ({source}): {description}",
            })
            log.info("GoalStackAgent: WORLD_SURPRISE → investigate goal pushed (source=%s)", source)

        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _synthesize_goals(self) -> None:
        """Convert current drive state into concrete goal entries on the stack."""
        now = time.time()

        # LEARN drive: if high and knowledge gaps exist
        learn_drive = self._current_drives.get("learn", 0.0)
        curiosity   = self._current_homeo.get("curiosity", 0.0)
        if (learn_drive > _LEARN_THRESHOLD or curiosity > 0.6) and self._knowledge_gaps:
            gap = self._knowledge_gaps[0]
            goal_id = f"learn:{gap[:40]}"
            if goal_id not in self._dispatched_goals:
                self._push_goal({
                    "id":       goal_id,
                    "type":     "learn",
                    "topic":    gap,
                    "action":   "curiosity_trigger",
                    "priority": learn_drive + curiosity * 0.5,
                    "message":  f"I want to learn about: {gap}",
                })

        # EXPLORE drive: if high and no one is present
        explore_drive = self._current_drives.get("explore", 0.0)
        if explore_drive > _EXPLORE_THRESHOLD and not self._person_present:
            goal_id = "explore:scan"
            if goal_id not in self._dispatched_goals:
                self._push_goal({
                    "id":       goal_id,
                    "type":     "explore",
                    "action":   "vlm_scan",
                    "priority": explore_drive,
                    "message":  "Let me look around and observe what's here.",
                })

        # SOCIAL drive: if high and person present but not talking
        social_drive = self._current_homeo.get("social", 0.0)
        silence_s = now - self._last_human_input_ts
        if (social_drive > _SOCIAL_THRESHOLD and self._person_present
                and silence_s > 45):  # person present but quiet for 45s
            goal_id = f"social:checkin:{int(now / 120)}"  # at most one per 2 min
            if goal_id not in self._dispatched_goals:
                self._push_goal({
                    "id":       goal_id,
                    "type":     "social",
                    "action":   "proactive_speak",
                    "priority": social_drive,
                    "message":  "Hey, is there anything on your mind?",
                })

    def _push_goal(self, goal: dict[str, Any]) -> None:
        """Push a goal onto the priority heap."""
        if len(self._goal_heap) >= _MAX_STACK_SIZE:
            return  # stack full — skip
        priority = goal.get("priority", 0.5)
        heapq.heappush(self._goal_heap, (-priority, time.time(), goal))
        log.debug("GoalStackAgent: goal pushed '%s' (priority=%.2f)", goal["id"], priority)

    async def _autonomous_loop(self) -> None:
        """Periodically dispatch the top goal if human is silent."""
        while self._running:
            await asyncio.sleep(10)
            await self._maybe_dispatch()

    async def _maybe_dispatch(self) -> None:
        """Dispatch the top-priority goal if autonomy conditions are met."""
        now = time.time()

        # Guard: human recently spoke → don't interrupt
        if (now - self._last_human_input_ts) < _HUMAN_SILENCE_THRESHOLD_S:
            return

        # Guard: rate limit
        if (now - self._last_dispatch_ts) < _DISPATCH_INTERVAL_S:
            return

        # Guard: already have an active goal
        if self._active_goal is not None:
            return

        # Guard: empty stack
        if not self._goal_heap:
            return

        _, _, goal = heapq.heappop(self._goal_heap)
        self._active_goal = goal
        self._last_dispatch_ts = now
        self._dispatched_goals.add(goal["id"])

        log.info("GoalStackAgent: dispatching autonomous goal '%s' (type=%s)",
                 goal["id"], goal["type"])

        await self.publish(AgentMessage(
            type=MessageType.GOAL_PUSH,
            source=self.name,
            data={"goal": goal, "ts": now},
            priority=6,
        ))

        # Route to appropriate downstream agent based on goal type
        action = goal.get("action", "")
        if action == "curiosity_trigger":
            await self.publish(AgentMessage(
                type=MessageType.CURIOSITY_TRIGGER,
                source=self.name,
                data={"topic": goal.get("topic", ""), "source": "goal_stack",
                      "goal_id": goal["id"]},
                priority=6,
            ))
        elif action == "vlm_scan":
            await self.publish(AgentMessage(
                type=MessageType.VLM_SCAN_NOW,
                source=self.name,
                data={"reason": "autonomous_explore", "goal_id": goal["id"]},
                priority=7,
            ))
        elif action == "proactive_speak":
            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": goal.get("message", ""), "goal_id": goal["id"]},
                priority=6,
            ))
            # Mark social goal complete immediately (it's a one-shot speak)
            self._active_goal = None
