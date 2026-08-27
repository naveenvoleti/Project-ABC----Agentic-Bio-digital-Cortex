"""Base agent interface — all agents inherit from this."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    PERCEPTION_VISION   = "perception.vision"
    PERCEPTION_SPEECH   = "perception.speech"
    PERCEPTION_SENSOR   = "perception.sensor"
    COGNITION_INTENT    = "cognition.intent"
    COGNITION_RESPONSE  = "cognition.response"
    EMOTION_CHANGE      = "emotion.change"
    ACTION_SPEAK        = "action.speak"
    ACTION_DISPLAY      = "action.display"
    ACTION_MOVE         = "action.move"
    MEMORY_WRITE        = "memory.write"
    MEMORY_READ         = "memory.read"
    SYSTEM_HEARTBEAT    = "system.heartbeat"
    SYSTEM_ERROR        = "system.error"
    BEHAVIOR_CHANGE     = "behavior.change"
    CURIOSITY_TRIGGER   = "curiosity.trigger"
    DREAM_START         = "dream.start"
    DREAM_DONE          = "dream.done"
    PRIVACY_MODE        = "system.privacy"
    SELF_REFLECT        = "system.self_reflect"   # periodic introspection
    SKILL_LEARN         = "skill.learn"            # user taught a new skill
    PLAN_STEP           = "plan.step"              # planner executing a queued step
    PLAN_CANCEL         = "plan.cancel"            # user cancelled active plan
    # ── Higher-Order Cognition ──────────────────────────────────────────────
    ATTENTION_FOCUS     = "attention.focus"        # AttentionAgent: spotlight on topic
    ATTENTION_SHIFT     = "attention.shift"        # AttentionAgent: bottom-up surprise
    IDEATION_REQUEST    = "ideation.request"       # request creative synthesis
    IDEATION_RESULT     = "ideation.result"        # IdeationAgent: idea/analogy output
    IMAGINATION_SIMULATE = "imagination.simulate"  # ImaginationAgent: mental sim result
    USER_MODEL_UPDATE   = "user_model.update"      # TheoryOfMindAgent: user expertise/confusion
    METACOG_CONFIDENCE  = "metacog.confidence"     # MetacognitionAgent: response confidence score
    MOTIVATION_DRIVE    = "motivation.drive"       # IntrinsicMotivationAgent: drive weights
    INTERO_STATE        = "intero.state"           # InteroceptionAgent: hardware → feelings
    TEMPORAL_INSIGHT    = "temporal.insight"       # TemporalReasoningAgent: patterns + causality
    # ── World Model / Conscious Perception ────────────────────────────────────
    SCENE_CHANGE        = "scene.change"           # SmolVLM2: significant scene shift detected
    GAZE_DIRECT         = "gaze.direct"            # SmolVLM2: user making direct eye contact
    PERSON_ENTER        = "person.enter"           # SmolVLM2: new person entered frame
    PERSON_LEAVE        = "person.leave"           # SmolVLM2: person left frame
    AUDIO_EVENT         = "audio.event"            # AuditoryAgent: non-speech audio spike
    VLM_SCAN_NOW        = "vlm.scan_now"           # Force immediate SmolVLM2 scan
    WORLD_UPDATE        = "world.update"           # WorldModel snapshot broadcast
    WORLD_SURPRISE      = "world.surprise"         # VisionAgent: prediction delta >30% → immediate re-eval
    COGNITION_THOUGHT   = "cognition.thought"      # ReasoningAgent: private pre-response thought (N3b)
    # ── V6.0 Self-Evolution ───────────────────────────────────────────────────
    WHEELED_ROTATE      = "motor.wheeled_rotate"   # differential rotation to re-center gaze
    NEURO_SYNTHESIS     = "neuro.synthesis"        # trigger self-coding (SynthesisAgent)
    CODE_VALIDATED      = "neuro.code_validated"   # sandbox test request (TesterAgent)
    REFLEX_READY        = "neuro.reflex_ready"     # validated module ready for hot-load
    # ── V7.0 Embodied Brain ───────────────────────────────────────────────────
    SPATIAL_POINT       = "spatial.point"          # VisionProcessor: normalised (x,y) for gaze snap
    SPATIAL_3D          = "spatial.3d"             # VisionProcessor: 3D bounding box list
    RE_PLAN             = "plan.replan"            # PlannerAgent: success prob < threshold → re-plan
    VLA_CONTROL_TOKEN   = "vla.control_token"      # Gemini VLA action token → WheelDriver/MotorDriver
    THINKING_BUDGET     = "cognition.thinking_budget"  # InteroceptionAgent: set ER token budget
    SUCCESS_ESTIMATE    = "plan.success_estimate"  # PlannerAgent: per-step success probability
    # ── V8.0 Self-Healing Task Organism ──────────────────────────────────────
    TASK_EXECUTE      = "task.execute"            # high-level goal dispatched to planner + vision
    TASK_OUTCOME      = "task.outcome"            # sensory verification result (success/failure/uncertain)
    DRIVER_SYNTHESIS  = "neuro.driver_synthesis"  # request to write external .py or .ino script
    # ── Streaming ────────────────────────────────────────────────────────────
    STREAM_TOKEN      = "stream.token"            # incremental LLM token for UI streaming
    OUTCOME_ANALYSIS  = "meta.outcome_analysis"   # metacognitive intent-vs-reality comparison
    NEURO_REWEIGHT    = "neuro.reweight"           # user correction → penalize wrong memory, upsert corrected fact
    # ── V10.0 Sentient Embodied Autonomous Brain ──────────────────────────────
    GOAL_PUSH           = "goal.push"            # GoalStackAgent: new autonomous goal added to stack
    GOAL_COMPLETE       = "goal.complete"        # GoalStackAgent: autonomous goal finished
    PROPRIOCEPTION_STATE = "body.proprioception" # ProprioceptionAgent: pan/tilt/heading/odometry
    BINDING_UPDATE      = "binding.moment"       # BindingAgent: unified present-moment phenomenal state


@dataclass
class AgentMessage:
    type: MessageType
    source: str
    data: dict[str, Any] = field(default_factory=dict)
    target: str = "orchestrator"
    priority: int = 5          # 1 = highest, 10 = lowest


class BaseAgent:
    name: str = "base"

    def __init__(self, bus: asyncio.Queue):
        self._bus = bus
        self._running = False
        self._watchdog = None   # injected by Orchestrator after construction

    def set_watchdog(self, watchdog) -> None:
        """Called by Orchestrator to inject the watchdog instance."""
        self._watchdog = watchdog

    def _beat(self) -> None:
        """Update watchdog heartbeat for this agent."""
        if self._watchdog:
            self._watchdog.heartbeat(self.name)

    async def _auto_beat_loop(self) -> None:
        """Background task: beats every 20s so passive agents stay alive
        without needing to call _beat() manually in their loops."""
        while self._running:
            self._beat()
            await asyncio.sleep(20)

    async def start(self) -> None:
        self._running = True
        self._beat()   # immediate beat on start
        # Launch background heartbeat — survives even if subclass loop is slow
        asyncio.create_task(self._auto_beat_loop(), name=f"{self.name}-heartbeat")

    async def stop(self) -> None:
        self._running = False

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        return None

    async def publish(self, message: AgentMessage) -> None:
        self._beat()   # every message also resets the heartbeat
        await self._bus.put(message)
