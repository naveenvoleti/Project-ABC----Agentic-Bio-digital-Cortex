"""
PlannerAgent — multi-step task execution.
Maps to the Prefrontal Cortex's planning and sequencing function.

Detects planning intent in LLM responses (reminders, sequences, stories + quiz)
and executes them as ordered steps via the message bus.

Supported patterns:
  "remind you in N minutes"    → delayed ACTION_SPEAK
  "tell you in N seconds"      → delayed ACTION_SPEAK
  "first X, then Y"            → immediate X, then Y after short pause
  "tell a story then quiz you" → story step, then quiz step
  "check back in N minutes"    → delayed greeting
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.utils.logger import get_logger
from brain.utils.repo_map import resolve_target_file

log = get_logger(__name__)

_SUCCESS_THRESHOLD = 0.4   # RE_PLAN fires when ER success_estimate < this


@dataclass
class TaskStep:
    """A single step in a plan."""
    action: str                    # "speak" | "remind" | "ask"
    data: dict[str, Any] = field(default_factory=dict)
    delay_s: float = 0.0           # seconds to wait before executing
    condition: str | None = None   # future: condition to check before running


# ── Pattern definitions ────────────────────────────────────────────────────────
# Each entry: (regex, builder_fn) where builder_fn(match) → list[TaskStep]

def _build_reminder(m: re.Match) -> list[TaskStep]:
    """remind/tell in N minutes/seconds."""
    amount = float(m.group(1))
    unit   = m.group(2).lower()
    delay  = amount * 60 if "min" in unit else amount
    msg    = m.group(3).strip() if m.lastindex >= 3 and m.group(3) else ""
    if not msg:
        msg = "Just a reminder — the time you asked about has passed."
    return [TaskStep(action="speak", data={"text": msg}, delay_s=delay)]


def _build_check_back(m: re.Match) -> list[TaskStep]:
    """check back / follow up in N minutes."""
    amount = float(m.group(1))
    unit   = m.group(2).lower()
    delay  = amount * 60 if "min" in unit else amount
    return [TaskStep(
        action="speak",
        data={"text": "Hey, checking back in — how's it going?"},
        delay_s=delay,
    )]


def _build_story_quiz(m: re.Match) -> list[TaskStep]:
    """tell a story then quiz."""
    return [
        TaskStep(action="speak", data={"text": "__story__"}, delay_s=0.0),
        TaskStep(action="ask",   data={"text": "__quiz__"},  delay_s=3.0),
    ]


def _build_first_then(m: re.Match) -> list[TaskStep]:
    """first X, then Y — two sequential actions with a 2s gap."""
    part_a = m.group(1).strip()
    part_b = m.group(2).strip()
    return [
        TaskStep(action="speak", data={"text": part_a}, delay_s=0.0),
        TaskStep(action="speak", data={"text": part_b}, delay_s=2.0),
    ]


PLAN_PATTERNS: list[tuple[str, Any]] = [
    # "remind you in 10 minutes to drink water"
    (r"remind\s+you\s+in\s+(\d+(?:\.\d+)?)\s+(minute|second|min|sec)s?\s+(?:to\s+)?(.{0,80})",
     _build_reminder),
    # "tell you in 5 minutes"
    (r"tell\s+you\s+in\s+(\d+(?:\.\d+)?)\s+(minute|second|min|sec)s?",
     _build_reminder),
    # "check back in 3 minutes" / "follow up in 2 minutes"
    (r"(?:check\s+back|follow\s+up)\s+in\s+(\d+(?:\.\d+)?)\s+(minute|second|min|sec)s?",
     _build_check_back),
    # "tell a story then quiz you" / "tell you a story and then test you"
    (r"tell\s+(?:you\s+)?a\s+story.*?(?:then|and\s+then).*?(?:quiz|test)",
     _build_story_quiz),
    # "first I'll X, then Y" — two-step sequence
    (r"first\s+(?:I.ll\s+)?(.{5,60}?),?\s+(?:then|and\s+then)\s+(.{5,60})",
     _build_first_then),
]

_CANCEL_WORDS = frozenset({"cancel", "stop", "nevermind", "forget it", "abort", "never mind"})


class PlannerAgent(BaseAgent):
    name = "planner_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        llm=None,          # LLMRouter — injected by Orchestrator for ER reasoning
        world_model=None,  # WorldModel — injected for scene context
    ):
        super().__init__(bus)
        self._llm = llm
        self._world_model = world_model
        self._queue: list[TaskStep] = []
        self._executing: bool = False
        self._cancel_flag: asyncio.Event = asyncio.Event()
        self._exec_task: asyncio.Task | None = None
        self._active_task: str = ""          # V7.0: current high-level task description
        self._replan_count: int = 0          # guard against infinite replan loops

    async def start(self) -> None:
        await super().start()
        log.info("PlannerAgent started")
        while self._running:
            self._beat()
            await asyncio.sleep(10)

    # ── V7.0 High-Level Brain (ER) planning ──────────────────────────────────

    async def _decompose_with_er(self, task: str) -> list[TaskStep] | None:
        """Use Gemini Robotics ER to decompose a complex task into motor steps.

        Returns a list of TaskSteps if ER produces a valid plan, else None
        so the caller can fall back to pattern matching.
        """
        if not self._llm:
            return None

        task_lower = task.lower()
        if "synthesize" in task_lower and ("script" in task_lower or "python" in task_lower or "code" in task_lower):
            log.info("PlannerAgent: task matches explicit synthesis request — routing directly to SynthesisAgent")
            await self.publish(AgentMessage(
                type=MessageType.DRIVER_SYNTHESIS,
                source=self.name,
                data={"task": task, "description": task, "script_type": "python"},
                priority=6,
            ))
            # Return empty list so handle() doesn't start an execution queue for this task
            return []

        context = ""
        if self._world_model:
            try:
                context = await self._world_model.format_for_prompt()
            except Exception:
                pass

        try:
            result = await self._llm.er_reason(task=task, context=context)
        except Exception as e:
            log.warning("PlannerAgent: er_reason failed: %s", e)
            return None

        plan = result.get("plan", [])
        success_est = float(result.get("success_estimate", 0.5))
        log.info("PlannerAgent ER plan: %d steps, success_est=%.2f", len(plan), success_est)

        if success_est < _SUCCESS_THRESHOLD:
            log.warning("PlannerAgent: ER success_estimate %.2f < %.2f → RE_PLAN",
                        success_est, _SUCCESS_THRESHOLD)
            await self.publish(AgentMessage(
                type=MessageType.RE_PLAN,
                source=self.name,
                data={"task": task, "success_estimate": success_est,
                      "reason": result.get("reasoning", "")},
                priority=3,
            ))
            return None

        if not plan:
            return None

        # V8.0 — structured step dispatch: synthesize_script / verify / execute
        self._active_task = task
        self._replan_count = 0
        steps: list[TaskStep] = []
        for i, step_raw in enumerate(plan):
            step_type = step_raw.get("type", "execute") if isinstance(step_raw, dict) else "execute"
            desc = step_raw.get("description", step_raw) if isinstance(step_raw, dict) else step_raw

            if step_type == "synthesize_script":
                await self.publish(AgentMessage(
                    type=MessageType.DRIVER_SYNTHESIS,
                    source=self.name,
                    data={"task": task, "description": desc,
                          "script_type": step_raw.get("script_type", "python")
                          if isinstance(step_raw, dict) else "python"},
                    priority=6,
                ))
            elif step_type == "verify":
                verify_target = (step_raw.get("verify_target", desc)
                                 if isinstance(step_raw, dict) else desc)
                await self.publish(AgentMessage(
                    type=MessageType.TASK_EXECUTE,
                    source=self.name,
                    data={"goal": task, "verify_target": verify_target,
                          "verification_only": True},
                    priority=5,
                ))
            else:
                steps.append(TaskStep(action="speak", data={"text": desc}, delay_s=0.5 * i))

        return steps if steps else None

    async def _estimate_success(self, step_text: str) -> float:
        """After a motor step completes, ask ER to estimate residual success probability.

        Returns 0.5 if ER unavailable. Emits RE_PLAN if below threshold.
        """
        if not self._llm or not self._active_task:
            return 0.5

        context = ""
        if self._world_model:
            try:
                context = await self._world_model.format_for_prompt()
            except Exception:
                pass

        try:
            result = await self._llm.er_reason(
                task=f"Evaluate progress: task='{self._active_task}', last_step='{step_text}'",
                context=context,
                max_tokens=256,
            )
        except Exception as e:
            log.debug("PlannerAgent: _estimate_success er_reason failed: %s", e)
            return 0.5

        prob = float(result.get("success_estimate", 0.5))
        await self.publish(AgentMessage(
            type=MessageType.SUCCESS_ESTIMATE,
            source=self.name,
            data={"step": step_text, "probability": prob, "task": self._active_task},
            priority=7,
        ))

        if prob < _SUCCESS_THRESHOLD and self._replan_count < 3:
            self._replan_count += 1
            log.warning("PlannerAgent: step success_estimate %.2f → RE_PLAN (attempt %d)",
                        prob, self._replan_count)
            await self.publish(AgentMessage(
                type=MessageType.RE_PLAN,
                source=self.name,
                data={"task": self._active_task, "success_estimate": prob,
                      "failed_step": step_text},
                priority=3,
            ))
        return prob

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_plan(self, text: str) -> list[TaskStep] | None:
        """Scan LLM response text for planning intent. Returns steps or None."""
        lower = text.lower()
        for pattern, builder in PLAN_PATTERNS:
            m = re.search(pattern, lower)
            if m:
                try:
                    steps = builder(m)
                    if steps:
                        log.info(f"PlannerAgent: detected plan ({len(steps)} steps) — pattern='{pattern[:40]}'")
                        return steps
                except Exception as e:
                    log.debug(f"PlannerAgent: plan builder error: {e}")
        return None

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute_queue(self) -> None:
        """Run through the step queue sequentially, respecting delays.
        Takes a snapshot of the queue at start so concurrent _queue.clear() calls
        can't corrupt the iteration mid-flight."""
        self._executing = True
        self._cancel_flag.clear()
        steps = list(self._queue)   # snapshot — cancel can clear self._queue safely
        log.info(f"PlannerAgent: executing {len(steps)} step(s)")

        for i, step in enumerate(steps):
            if self._cancel_flag.is_set():
                log.info("PlannerAgent: plan cancelled mid-execution")
                break

            # Wait for the step's delay (interruptible by cancel)
            if step.delay_s > 0:
                log.debug(f"PlannerAgent: step {i+1} waiting {step.delay_s:.0f}s")
                try:
                    await asyncio.wait_for(
                        self._cancel_flag.wait(),
                        timeout=step.delay_s,
                    )
                    # cancel_flag was set during the wait
                    log.info("PlannerAgent: plan cancelled during delay")
                    break
                except asyncio.TimeoutError:
                    pass   # delay elapsed normally

            if self._cancel_flag.is_set():
                break

            await self._execute_step(step)

        self._queue.clear()
        self._executing = False
        log.info("PlannerAgent: plan complete")

    async def _execute_step(self, step: TaskStep) -> None:
        """Dispatch a single step onto the message bus."""
        if step.action in ("speak", "remind"):
            text = step.data.get("text", "")
            # Placeholder tokens — signal to LLM to generate the content
            if text in ("__story__", "__quiz__"):
                prompt = (
                    "Tell a short creative story (5-6 sentences)."
                    if text == "__story__"
                    else "Now ask me 2 fun quiz questions about the story you just told."
                )
                await self.publish(AgentMessage(
                    type=MessageType.COGNITION_INTENT,
                    source=self.name,
                    data={"text": prompt, "intent": "general_query", "entities": {}},
                    priority=3,
                ))
            elif text:
                await self.publish(AgentMessage(
                    type=MessageType.ACTION_SPEAK,
                    source=self.name,
                    data={"text": text},
                    priority=2,
                ))

        elif step.action == "ask":
            text = step.data.get("text", "")
            if text == "__quiz__":
                await self.publish(AgentMessage(
                    type=MessageType.COGNITION_INTENT,
                    source=self.name,
                    data={"text": "Now ask me 2 quiz questions about the story.", "intent": "general_query", "entities": {}},
                    priority=3,
                ))
            elif text:
                await self.publish(AgentMessage(
                    type=MessageType.ACTION_SPEAK,
                    source=self.name,
                    data={"text": text},
                    priority=2,
                ))

        log.debug(f"PlannerAgent: executed step action={step.action}")

    # ── Message handling ───────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> None:
        # V8.0 — direct high-level goal dispatch
        if message.type == MessageType.TASK_EXECUTE:
            goal = message.data.get("goal", message.data.get("text", ""))
            if goal and not message.data.get("verification_only"):
                self._active_task = goal
                self._replan_count = 0
                er_steps = await self._decompose_with_er(goal)
                if er_steps:
                    if self._executing:
                        self._cancel_flag.set()
                        await asyncio.sleep(0.05)
                    self._queue = er_steps
                    self._exec_task = asyncio.create_task(
                        self._execute_queue(), name="planner-task-exec"
                    )
            return

        if message.type == MessageType.COGNITION_INTENT:
            # V7.0: try ER decomposition for explicit multi-step task requests
            text = message.data.get("text", "")
            intent = message.data.get("intent", "")
            if text and intent in ("task", "plan", "do", "execute", "embodied"):
                er_steps = await self._decompose_with_er(text)
                if er_steps:
                    if self._executing:
                        self._cancel_flag.set()
                        await asyncio.sleep(0.05)
                    self._queue = er_steps
                    self._exec_task = asyncio.create_task(
                        self._execute_queue(), name="planner-er-exec"
                    )
                    return

        elif message.type == MessageType.RE_PLAN:
            # Another agent (or self) requests a re-plan — re-decompose with ER
            task = message.data.get("task", self._active_task)
            if task and self._replan_count < 3:
                er_steps = await self._decompose_with_er(task)
                if er_steps:
                    if self._executing:
                        self._cancel_flag.set()
                        await asyncio.sleep(0.05)
                    self._queue = er_steps
                    self._exec_task = asyncio.create_task(
                        self._execute_queue(), name="planner-replan-exec"
                    )
            return

        if message.type == MessageType.COGNITION_RESPONSE:
            # Inspect LLM output for planning intent
            text = message.data.get("text", "")
            if not text:
                return

            # Detect code-change intent — LLM accepted a synthesis directive
            _CODE_INTENT_PHRASES = (
                "update the ui", "change the ui", "update ui", "modify the ui",
                "update the codebase", "change the code", "update the code",
                "initiating code", "initiating a codebase", "will update",
                "will modify", "will change", "applying the change",
                "i will implement", "executing the update",
            )
            text_lower = text.lower()

            # Detect explicit [EXECUTE: ...] action tokens from the LLM
            _execute_token = re.search(r"\[EXECUTE:\s*([A-Z_]+)\]", text, re.IGNORECASE)

            if _execute_token or any(phrase in text_lower for phrase in _CODE_INTENT_PHRASES):
                target_file = resolve_target_file(text)
                action = _execute_token.group(1).upper() if _execute_token else "MODIFY_UI"
                log.info("PlannerAgent: %s detected — target=%s", action, target_file)
                await self.publish(AgentMessage(
                    type=MessageType.NEURO_SYNTHESIS,
                    source=self.name,
                    data={
                        "trigger": action,
                        "intent_text": text[:200],
                        "target_file": target_file,
                    },
                    priority=8,
                ))

            steps = self._detect_plan(text)
            if steps:
                # Cancel any in-progress plan before starting new one
                if self._executing:
                    self._cancel_flag.set()
                    if self._exec_task and not self._exec_task.done():
                        await asyncio.sleep(0.1)   # let cancel propagate

                self._queue = steps
                self._exec_task = asyncio.create_task(
                    self._execute_queue(),
                    name="planner-exec",
                )

        elif message.type == MessageType.PERCEPTION_SPEECH:
            # Cancel plan on user command
            text = message.data.get("text", "").lower().strip()
            if self._executing and any(w in text for w in _CANCEL_WORDS):
                log.info("PlannerAgent: user cancelled plan")
                self._cancel_flag.set()
                self._queue.clear()
                await self.publish(AgentMessage(
                    type=MessageType.ACTION_SPEAK,
                    source=self.name,
                    data={"text": "Okay, I've cancelled that."},
                    priority=2,
                ))

        elif message.type == MessageType.PLAN_CANCEL:
            # Programmatic cancel (from other agents)
            self._cancel_flag.set()
            self._queue.clear()

    async def stop(self) -> None:
        await super().stop()
        self._cancel_flag.set()
        if self._exec_task and not self._exec_task.done():
            self._exec_task.cancel()
