"""
Orchestrator — master coordinator.
Spawns all agents, routes messages between them, and manages lifecycle.
Maps to the Prefrontal Cortex — the "executive function" of the brain.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.agents.planner_agent import PlannerAgent
from brain.agents.vision_agent import VisionAgent
from brain.agents.auditory_agent import AuditoryAgent
from brain.agents.sensory_agent import SensoryAgent
from brain.agents.language_agent import LanguageAgent
from brain.agents.reasoning_agent import ReasoningAgent
from brain.agents.logic_agent import LogicAgent
from brain.agents.cognition_agent import CognitionAgent
from brain.agents.emotion_engine import EmotionEngine
from brain.agents.motor_agent import MotorAgent
from brain.agents.display_agent import DisplayAgent
from brain.agents.speech_agent import SpeechAgent
from brain.agents.behavior_agent import BehaviorAgent
from brain.agents.curiosity_agent import CuriosityAgent
from brain.agents.verifier_agent import VerifierAgent
from brain.agents.dream_agent import DreamAgent
from brain.agents.attention_agent import AttentionAgent
from brain.agents.interoception_agent import InteroceptionAgent
from brain.agents.metacognition_agent import MetacognitionAgent
from brain.agents.theory_of_mind_agent import TheoryOfMindAgent
from brain.agents.temporal_reasoning_agent import TemporalReasoningAgent
from brain.agents.intrinsic_motivation_agent import IntrinsicMotivationAgent
from brain.agents.ideation_agent import IdeationAgent
from brain.agents.imagination_agent import ImaginationAgent
from brain.agents.synthesis_agent import SynthesisAgent
from brain.agents.tester_agent import TesterAgent
from brain.agents.mirror_agent import MirrorAgent
# V10.0 — Sentient Embodied Autonomous Brain
from brain.agents.goal_stack_agent import GoalStackAgent
from brain.agents.proprioception_agent import ProprioceptionAgent
from brain.agents.learning_agent import LearningAgent
from brain.agents.binding_agent import BindingAgent
from brain.hardware.hw_detector import HardwareCapabilities
from brain.hardware.audio_driver import AudioDriver
from brain.hardware.display_driver import DisplayDriver
from brain.hardware.motor_driver import MotorDriver
from brain.hardware.wheel_driver import WheelDriver
from brain.hardware.gpio_driver import GPIODriver
from brain.hardware.face_recognizer import FaceRecognizer
from brain.hardware.vision_processor import VisionProcessor
from brain.hardware.smolvlm2_processor import SmolVLM2Processor
from brain.memory.world_model import WorldModel
from brain.memory.working_memory import WorkingMemory
from brain.memory.episodic_memory import EpisodicMemory
from brain.memory.semantic_memory import SemanticMemory
from brain.memory.gallery_manager import GalleryManager
from brain.memory.soul_manager import SoulManager
from brain.llm.llm_router import LLMRouter
from brain.llm.mcp_tool_registry import MCPToolRegistry
from brain.utils.logger import get_logger
from brain.utils.watchdog import Watchdog

log = get_logger(__name__)

# Reflexive message types: routed immediately, bypassing the GWT workspace buffer.
# These are hardware commands, safety signals, and high-priority lifecycle events
# that must not wait for the 10Hz Pulse cycle.
# Agents suppressed during ECO/overwhelmed state — Brain Fog effect
_ECO_BLOCKED_AGENTS: frozenset[str] = frozenset({"imagination_agent"})

_REFLEXIVE_TYPES: frozenset[MessageType] = frozenset({
    MessageType.ACTION_MOVE,
    MessageType.ACTION_SPEAK,
    MessageType.ACTION_DISPLAY,
    MessageType.SYSTEM_ERROR,
    MessageType.PRIVACY_MODE,
    MessageType.DREAM_START,
    MessageType.DREAM_DONE,
    MessageType.PLAN_STEP,
    MessageType.PLAN_CANCEL,
    MessageType.WORLD_SURPRISE,   # always immediate — max priority 1 bypasses gating
    MessageType.EMOTION_CHANGE,
    MessageType.BEHAVIOR_CHANGE,
    MessageType.SKILL_LEARN,
    MessageType.SELF_REFLECT,
    MessageType.MEMORY_WRITE,
    # V6.0 — all self-evolution messages bypass GWT buffer
    MessageType.WHEELED_ROTATE,
    MessageType.NEURO_SYNTHESIS,
    MessageType.CODE_VALIDATED,
    MessageType.REFLEX_READY,
    # V7.0 — embodied real-time signals bypass GWT buffer
    MessageType.SPATIAL_POINT,
    MessageType.VLA_CONTROL_TOKEN,
    MessageType.THINKING_BUDGET,
    MessageType.RE_PLAN,
    # V8.0 — task execution signals bypass GWT buffer
    MessageType.TASK_EXECUTE,
    MessageType.TASK_OUTCOME,
    MessageType.DRIVER_SYNTHESIS,
    MessageType.OUTCOME_ANALYSIS,
    # V10.0 — all new agent signals bypass GWT buffer
    MessageType.PROPRIOCEPTION_STATE,
    MessageType.BINDING_UPDATE,
    MessageType.GOAL_PUSH,
    MessageType.GOAL_COMPLETE,
})

# Message routing table: message type → list of target agent names
ROUTING: dict[MessageType, list[str]] = {
    MessageType.PERCEPTION_SPEECH:  ["language_agent", "behavior_agent", "emotion_engine",
                                      "curiosity_agent", "speech_agent", "vision_agent",
                                      "planner_agent", "attention_agent",
                                      "theory_of_mind_agent", "temporal_reasoning_agent",
                                      "imagination_agent",
                                      # V10.0
                                      "goal_stack_agent", "learning_agent",
                                      "binding_agent"],
    MessageType.PERCEPTION_VISION:  ["behavior_agent", "cognition_agent"],
    MessageType.PERCEPTION_SENSOR:  ["behavior_agent", "sensory_agent", "reasoning_agent",
                                      "interoception_agent",
                                      # V10.0 — distance sensor feeds proprioception
                                      "proprioception_agent"],

    MessageType.COGNITION_INTENT:   ["cognition_agent", "behavior_agent", "attention_agent", "planner_agent"],
    # metacognition_agent MUST come before verifier_agent so it scores the
    # response and sets _confidence_override before verifier publishes ACTION_SPEAK
    MessageType.COGNITION_RESPONSE: ["metacognition_agent", "verifier_agent",
                                      "emotion_engine", "planner_agent",
                                      "behavior_agent", "theory_of_mind_agent",
                                      "intrinsic_motivation_agent",
                                      # V10.0
                                      "learning_agent"],
    MessageType.EMOTION_CHANGE:     ["display_agent", "motor_agent", "cognition_agent", "speech_agent",
                                      "attention_agent",
                                      # V10.0
                                      "binding_agent"],
    MessageType.ACTION_SPEAK:       ["speech_agent", "display_agent", "cognition_agent", "auditory_agent"],
    MessageType.ACTION_DISPLAY:     ["display_agent", "auditory_agent"],
    MessageType.ACTION_MOVE:        ["logic_agent", "motor_agent",
                                      # V10.0 — efference copy for body-schema tracking
                                      "proprioception_agent"],

    MessageType.BEHAVIOR_CHANGE:    ["emotion_engine", "curiosity_agent", "speech_agent",
                                      "cognition_agent", "vision_agent",
                                      "attention_agent", "intrinsic_motivation_agent"],
    MessageType.CURIOSITY_TRIGGER:  ["curiosity_agent", "motor_agent", "emotion_engine",
                                      "ideation_agent",
                                      # V10.0
                                      "learning_agent"],
    MessageType.SYSTEM_ERROR:       ["emotion_engine", "display_agent"],
    MessageType.PRIVACY_MODE:       ["vision_agent", "auditory_agent", "display_agent"],
    MessageType.DREAM_START:        ["dream_agent"],
    MessageType.DREAM_DONE:         ["display_agent", "intrinsic_motivation_agent"],
    MessageType.MEMORY_WRITE:       ["cognition_agent"],
    MessageType.SELF_REFLECT:       ["cognition_agent"],
    MessageType.SKILL_LEARN:        ["cognition_agent"],
    MessageType.PLAN_CANCEL:        ["planner_agent"],
    # ── Higher-order cognition routing ───────────────────────────────────────
    MessageType.ATTENTION_FOCUS:    ["cognition_agent", "behavior_agent"],
    MessageType.ATTENTION_SHIFT:    ["cognition_agent", "behavior_agent", "emotion_engine"],
    MessageType.IDEATION_REQUEST:   ["ideation_agent"],
    MessageType.IDEATION_RESULT:    ["cognition_agent"],
    # imagination_agent listed first so it handles external IMAGINATION_SIMULATE requests;
    # self-route guard in _route() prevents ImaginationAgent from processing its own results
    # vision_agent receives prediction to use in SurpriseHeuristic comparison
    MessageType.IMAGINATION_SIMULATE: ["imagination_agent", "cognition_agent", "planner_agent", "vision_agent"],
    MessageType.USER_MODEL_UPDATE:  ["cognition_agent"],
    MessageType.METACOG_CONFIDENCE: ["verifier_agent"],
    MessageType.MOTIVATION_DRIVE:   ["behavior_agent", "curiosity_agent",
                                      # V10.0
                                      "goal_stack_agent"],
    MessageType.INTERO_STATE:       ["emotion_engine", "behavior_agent",
                                      "intrinsic_motivation_agent", "cognition_agent"],
    MessageType.WORLD_SURPRISE:     ["cognition_agent", "attention_agent", "curiosity_agent",
                                      # V10.0
                                      "binding_agent",
                                      # V10.0 — WORLD_SURPRISE triggers SURPRISED emotion state
                                      "emotion_engine",
                                      # V10.0 — generate investigate goal
                                      "goal_stack_agent"],


    MessageType.TEMPORAL_INSIGHT:   ["cognition_agent"],
    MessageType.COGNITION_THOUGHT:  ["verifier_agent", "binding_agent"],

    # ── World Model / Conscious Perception routing ──────────────────────────
    MessageType.SCENE_CHANGE:       ["curiosity_agent", "cognition_agent", "behavior_agent"],
    MessageType.GAZE_DIRECT:        ["curiosity_agent", "emotion_engine"],
    MessageType.PERSON_ENTER:       ["curiosity_agent", "emotion_engine", "cognition_agent",
                                      "intrinsic_motivation_agent",
                                      # V10.0
                                      "goal_stack_agent", "binding_agent"],
    MessageType.PERSON_LEAVE:       ["curiosity_agent", "intrinsic_motivation_agent",
                                      # V10.0
                                      "goal_stack_agent", "binding_agent"],
    MessageType.AUDIO_EVENT:        ["curiosity_agent", "emotion_engine",
                                      # V10.0
                                      "binding_agent"],
    MessageType.VLM_SCAN_NOW:       ["vision_agent", "mirror_agent"],
    MessageType.WORLD_UPDATE:       ["cognition_agent", "curiosity_agent"],
    # V6.0 — Self-evolution routing
    MessageType.WHEELED_ROTATE:     ["motor_agent"],
    MessageType.NEURO_SYNTHESIS:    ["synthesis_agent"],
    MessageType.CODE_VALIDATED:     ["tester_agent"],
    MessageType.REFLEX_READY:       [],   # handled inline in _route()
    # V7.0 — Embodied Brain routing

    MessageType.SPATIAL_POINT:      ["motor_agent"],
    MessageType.SPATIAL_3D:         ["cognition_agent"],
    MessageType.VLA_CONTROL_TOKEN:  ["motor_agent"],
    MessageType.THINKING_BUDGET:    [],   # handled inline in _route()
    MessageType.RE_PLAN:            ["planner_agent"],
    MessageType.SUCCESS_ESTIMATE:   ["cognition_agent"],
    # V8.0 — Self-Healing Task Organism routing
    MessageType.TASK_EXECUTE:       ["planner_agent", "vision_agent"],
    MessageType.TASK_OUTCOME:       ["metacognition_agent", "cognition_agent"],
    MessageType.DRIVER_SYNTHESIS:   ["synthesis_agent"],
    MessageType.OUTCOME_ANALYSIS:   ["metacognition_agent"],
    # V9.0 — Neural Standardization routing
    MessageType.NEURO_REWEIGHT:     ["metacognition_agent"],
    # V10.0 — Sentient Embodied Autonomous Brain routing
    MessageType.GOAL_PUSH:           ["cognition_agent", "planner_agent",
                                       "learning_agent", "behavior_agent"],
    MessageType.GOAL_COMPLETE:       ["cognition_agent", "behavior_agent"],
    MessageType.PROPRIOCEPTION_STATE:["cognition_agent"],
    MessageType.BINDING_UPDATE:      ["cognition_agent", "reasoning_agent"],
}


class Orchestrator:
    def __init__(self, config: dict, hw: HardwareCapabilities):
        self._config = config
        self._hw = hw
        self._bus: asyncio.Queue[AgentMessage] = asyncio.Queue(maxsize=800)
        self._agents: dict[str, BaseAgent] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._watchdog = Watchdog()
        self._ws_subscribers: list[asyncio.Queue] = []
        self._running = False
        self._eco_mode: bool = False

        # V6.0 — Self-evolution state
        _syn_cfg = config.get("synthesis_agent", {})
        self._synthesis_crash_threshold: int = _syn_cfg.get("crash_threshold", 3)
        self._synthesis_confidence_threshold: float = _syn_cfg.get("confidence_threshold", 0.2)
        self._session_synthesis_limit: int = _syn_cfg.get("session_synthesis_limit", 5)
        self._error_registry: dict[str, int] = {}     # agent_name → crash count
        self._pulse_count: int = 0
        # Pattern registry — shared reference with DreamAgent for reflex distillation
        self._pattern_registry: dict[str, list[str]] = {}
        _reflex_cfg = config.get("reflex_registry", {})
        self._pattern_threshold: int = _reflex_cfg.get("pattern_threshold", 5)

        # V6.0 — Differential wheel drive
        _wd_cfg = config.get("wheel_drive", {})
        if hw.has_differential_drive:
            self._wheel_driver: WheelDriver | None = WheelDriver(
                left_pins=hw.wheel_left_pins,
                right_pins=hw.wheel_right_pins,
                speed_default=_wd_cfg.get("speed_default", 50),
                rotate_speed=_wd_cfg.get("rotate_speed", 35),
                ramp_ms=_wd_cfg.get("ramp_ms", 50),
                mock=False,
            )
            self._wheel_driver.init()
        else:
            self._wheel_driver = None

        # V8.0 — Task failure watchdog (3× same task → DNA rollback)
        _te_cfg = config.get("task_executor", {})
        self._max_task_failures: int = _te_cfg.get("max_task_failures", 3)
        self._task_failure_registry: dict[str, int] = {}

        # V7.0 — Dynamic thinking budget (adjusted by InteroceptionAgent thermal signals)
        _gr_cfg = config.get("gemini_robotics", {})
        self._thinking_budget: int = _gr_cfg.get("thinking_budget_default", 1024)
        self._thinking_budget_eco: int = _gr_cfg.get("thinking_budget_eco", 256)
        self._plan_active: bool = False    # True while deliberative ER planning is running

        # Cache v5.1 feature flags from config to avoid repeated dict lookups
        _ce = config.get("cognition_extensions", {})
        self._brain_fog_enabled: bool = _ce.get("brain_fog", {}).get("enabled", True)
        self._affective_bias_cfg: dict = _ce.get("affective_bias", {})
        self._internal_monologue_cfg: dict = _ce.get("internal_monologue", {})
        self._neuroplasticity_cfg: dict = _ce.get("neuroplasticity", {})
        self._emotional_pregate_cfg: dict = _ce.get("emotional_pregate", {})

        # GWT v5.0 — Global Workspace state
        gwt_cfg = config.get("cognition_extensions", {}).get("attention", {})
        self._pulse_hz: float = config.get("brain", {}).get("pulse_hz", 10.0)
        self._spotlight_size: int = gwt_cfg.get("spotlight_size", 3)
        self._salience_threshold: float = gwt_cfg.get("salience_threshold", 0.4)
        # Spotlight: list of {salience, type, source, ts} for the /workspace API
        self._spotlight: list[dict] = []
        # Workspace buffer: non-reflexive messages wait here for the Pulse cycle
        self._workspace_pending: list[tuple[float, AgentMessage]] = []  # (salience, msg)
        self._workspace_lock = asyncio.Lock()

        # Build shared resources
        self._working = WorkingMemory(config.get("memory", {}).get("working_ttl_seconds", 3600))
        self._episodic = EpisodicMemory(
            config["memory"]["episodic_db"],
            pregate_enabled=self._emotional_pregate_cfg.get("enabled", True),
            pregate_threshold=self._emotional_pregate_cfg.get("importance_threshold", 0.2),
        )
        self._semantic = SemanticMemory(config["memory"]["semantic_db"])
        self._gallery = GalleryManager(config.get("memory", {}).get("gallery_dir", "data/gallery"))
        self._soul = SoulManager(
            config["memory"]["soul_file"],
            config["memory"]["user_file"],
            config["memory"]["world_file"],
            config["memory"]["skills_file"],
            user_json_path=config["memory"].get("user_json_file", "data/USER.json"),
        )
        self._soul.load_all()
        # NOTE: PersonalityCore removed — OCEAN traits now live in USER.json
        # and are surfaced via SoulManager.get_system_prompt() / soul_manager.py.

        llm_cfg = config.get("llm", {})
        self._llm = LLMRouter(
            ollama_host=llm_cfg.get("ollama_host", "http://localhost:11434"),
            simple_model=llm_cfg.get("default_model", "phi3:mini"),
            balanced_model=llm_cfg.get("balanced_model", "llama3.2:3b"),
            cloud_model=llm_cfg.get("cloud_model", "claude-haiku-4-5-20251001"),
            embed_model=llm_cfg.get("embed_model", "nomic-embed-text"),
            anthropic_api_key=llm_cfg.get("anthropic_api_key", ""),
            openrouter_api_key=llm_cfg.get("openrouter_api_key", ""),
            openrouter_model=llm_cfg.get("openrouter_model", "meta-llama/llama-3.2-3b-instruct:free"),
            google_ai_api_key=llm_cfg.get("google_ai_api_key", ""),
            network_available=hw.network_available,
            ollama_enabled=llm_cfg.get("ollama_enabled", True),
            openrouter_enabled=llm_cfg.get("openrouter_enabled", True),
            ollama_cloud_api_key=llm_cfg.get("ollama_cloud_api_key", ""),
            ollama_cloud_model=llm_cfg.get("ollama_cloud_model", "llama3.3:70b-cloud"),
            ollama_cloud_enabled=llm_cfg.get("ollama_cloud_enabled", False),
            er_enabled=llm_cfg.get("er_enabled", True),
            google_ai_enabled=llm_cfg.get("google_ai_enabled", True),
        )

        # Hardware drivers
        # NOTE: mock= reflects whether the hardware is actually absent,
        # NOT whether we are in mock_mode. This lets real laptop mic/camera/speakers
        # work even when --mock is set (which only mocks Pi peripherals).
        _audio_out = config.get("audio", {}).get("output", {})
        self._audio = AudioDriver(
            speaker_type=hw.speaker_type,
            bt_mac=hw.bluetooth_speaker_mac,
            tts_engine=_audio_out.get("tts_engine", "auto"),
            edge_voice=_audio_out.get("edge_voice", "en-US-JennyNeural"),
            elevenlabs_api_key=_audio_out.get("elevenlabs_api_key", ""),
            elevenlabs_voice_name=_audio_out.get("elevenlabs_voice_name", "Anika"),
            elevenlabs_voice_id=_audio_out.get("elevenlabs_voice_id", ""),
            mock=not hw.speaker_available,
        )
        self._display_drv = DisplayDriver(
            display_type=hw.display_type if hw.display_available else "web",
            width=config.get("display", {}).get("width", 240),
            height=config.get("display", {}).get("height", 320),
            mock=not hw.display_available,
        )
        self._motor_drv = MotorDriver(
            motor_type=hw.motor_type,
            mock=not hw.motor_available,
        )
        self._gpio_drv = GPIODriver(mock=not hw.gpio_available)

        # Moondream vision processor — semantic pre-processing before LLM
        _vision_cfg = config.get("vision", {}).get("moondream", {})
        _moondream_mode = "mock" if not hw.camera_available else _vision_cfg.get("mode", "auto")
        self._vision_proc = VisionProcessor(
            mode=_moondream_mode,
            api_key=_vision_cfg.get("api_key", ""),
            model_size=_vision_cfg.get("model_size", "2b"),
            caption_interval_s=_vision_cfg.get("caption_interval_s", 3.0),
            bus=self._bus,
        )
        if hw.camera_available:
            self._vision_proc.init()

        # Face recognizer — identify returning users
        _face_cfg = config.get("vision", {}).get("face_recognition", {})
        _face_mock = not hw.camera_available or _face_cfg.get("mock", False)
        self._face_rec = FaceRecognizer(mock=_face_mock)
        self._face_rec.init()

        # WorldModel — shared perceptual ground truth (written by VisionAgent/AuditoryAgent,
        # read by CognitionAgent before every LLM call)
        self._world_model = WorldModel()

        # SmolVLM2 — continuous video understanding (Intel OpenVINO + HuggingFace fallback)
        _smolvlm2_cfg = config.get("vision", {}).get("smolvlm2", {})
        _smolvlm2_enabled = _smolvlm2_cfg.get("enabled", False) and hw.camera_available
        if _smolvlm2_enabled:
            self._smolvlm2 = SmolVLM2Processor(_smolvlm2_cfg)
            log.info("SmolVLM2Processor created — will load on VisionAgent.start()")
        else:
            self._smolvlm2 = None
            if not hw.camera_available:
                log.info("SmolVLM2: disabled (no camera)")
            elif not _smolvlm2_cfg.get("enabled", False):
                log.info("SmolVLM2: disabled in config (vision.smolvlm2.enabled=false)")

        # Store hw summary for LLM context
        self._working.set("hw_summary", self._build_hw_summary())

        # V9.0 — MCP Tool Registry: expose hardware capabilities as LLM function-calling schemas
        self._mcp_tools = MCPToolRegistry(
            motor_drv=self._motor_drv,
            gpio_drv=self._gpio_drv,
            audio_drv=self._audio,
            hw=hw,
        )
        log.info(
            "MCPToolRegistry: %d tools advertised: %s",
            len(self._mcp_tools.get_tools()),
            ", ".join(self._mcp_tools.get_tool_names()),
        )

    def _build_hw_summary(self) -> str:
        parts = []
        if self._hw.camera_available:
            parts.append(f"camera({self._hw.camera_source})")
        if self._hw.mic_available:
            parts.append("microphone")
        if self._hw.speaker_available:
            parts.append(f"speaker({self._hw.speaker_type})")
        if self._hw.display_available:
            parts.append(f"display({self._hw.display_type})")
        if self._hw.motor_available:
            parts.append(f"motors({self._hw.motor_type})")
        return ", ".join(parts) if parts else "no hardware detected"

    def _agent_enabled(self, name: str) -> bool:
        """Return True if the agent is enabled in agents.yaml (default: True)."""
        return self._config.get("agents", {}).get(name, {}).get("enabled", True)

    def _agent_features(self, name: str) -> dict:
        """Return the feature-flag dict for an agent (default: all enabled)."""
        return self._config.get("agents", {}).get(name, {}).get("features", {})

    def _build_agents(self) -> None:
        bus = self._bus
        _e = self._agent_enabled   # shorthand

        # Candidates — only add if enabled in agents.yaml
        candidates: dict[str, Any] = {}

        _vis_cfg = self._config.get("vision", {})
        if _e("vision_agent") and _vis_cfg.get("enabled", True):
            candidates["vision_agent"] = VisionAgent(
                bus, self._hw, self._vision_proc, self._face_rec,
                smolvlm2=self._smolvlm2,
                world_model=self._world_model,
                llm=self._llm if _vis_cfg.get("scan_enabled", True) else None,
                scan_interval_s=float(_vis_cfg.get("scan_interval", 3.0)),
                semantic=self._semantic,
                gallery=self._gallery,
                soul=self._soul,
            )

        _aud_in_cfg = self._config.get("audio", {}).get("input", {})
        if _e("auditory_agent") and _aud_in_cfg.get("enabled", True):
            candidates["auditory_agent"] = AuditoryAgent(
                bus, self._hw, self._audio,
                world_model=self._world_model,
            )
        if _e("sensory_agent"):
            candidates["sensory_agent"] = SensoryAgent(bus, self._hw, self._gpio_drv)
        if _e("language_agent"):
            candidates["language_agent"] = LanguageAgent(bus)
        if _e("cognition_agent"):
            candidates["cognition_agent"] = CognitionAgent(
                bus, self._working, self._episodic, self._semantic, self._soul, self._llm,
                world_model=self._world_model,
                gallery=self._gallery,
                vision_proc=self._vision_proc,
            )
        if _e("reasoning_agent"):
            _im = self._internal_monologue_cfg
            _reflex_sim = self._config.get("reflex_registry", {}).get("similarity_threshold", 0.92)
            candidates["reasoning_agent"] = ReasoningAgent(
                bus, self._llm, self._episodic,
                internal_monologue_enabled=_im.get("enabled", True),
                think_tokens=_im.get("think_tokens", 80),
                min_response_tokens=_im.get("min_response_tokens", 60),
                brevity_on_frustrated=self._affective_bias_cfg.get("brevity_on_frustrated", True),
                soul=self._soul,
                reflex_similarity_threshold=_reflex_sim,
                frame_getter=self._get_frame_b64,
            )
        if _e("logic_agent"):
            candidates["logic_agent"] = LogicAgent(bus)
        if _e("emotion_engine"):
            candidates["emotion_engine"] = EmotionEngine(bus)
        if _e("motor_agent"):
            _mot = self._config.get("motor", {})
            _wd = self._config.get("wheel_drive", {})
            candidates["motor_agent"] = MotorAgent(
                bus, self._hw, self._motor_drv,
                saccade_smoothing=_mot.get("saccade_smoothing", True),
                saccade_steps=_mot.get("saccade_steps", 8),
                saccade_duration_ms=_mot.get("saccade_duration_ms", 200),
                wheel_driver=self._wheel_driver,
                gaze_rotate_threshold=float(_wd.get("gaze_rotate_threshold", 45.0)),
                gaze_center_threshold=float(_wd.get("gaze_center_threshold", 15.0)),
            )
        if _e("display_agent"):
            candidates["display_agent"] = DisplayAgent(bus, self._display_drv)
        if _e("speech_agent") and self._config.get("audio", {}).get("output", {}).get("enabled", True):
            candidates["speech_agent"] = SpeechAgent(bus, self._audio)
        if _e("behavior_agent"):
            candidates["behavior_agent"] = BehaviorAgent(bus, self._config.get("brain", {}).get("idle_timeout_seconds", 300))
        if _e("curiosity_agent"):
            candidates["curiosity_agent"] = CuriosityAgent(bus, self._hw, self._llm, self._episodic, self._soul, semantic=self._semantic)
        if _e("verifier_agent"):
            candidates["verifier_agent"] = VerifierAgent(bus)
        if _e("dream_agent"):
            _reflex_cfg = self._config.get("reflex_registry", {})
            candidates["dream_agent"] = DreamAgent(
                bus, self._episodic, self._semantic, self._soul, self._llm,
                pattern_threshold=_reflex_cfg.get("pattern_threshold", 5),
                prune_after_days=_reflex_cfg.get("prune_after_days", 7),
                promote_after_invocations=_reflex_cfg.get("promote_after_invocations", 100),
            )
        if _e("planner_agent"):
            candidates["planner_agent"] = PlannerAgent(
                bus,
                llm=self._llm,
                world_model=self._world_model,
            )

        # ── Higher-order cognitive agents ─────────────────────────────────────
        if _e("attention_agent"):
            _ab = self._affective_bias_cfg
            candidates["attention_agent"] = AttentionAgent(
                bus,
                affective_bias_enabled=_ab.get("enabled", True),
                affective_multiplier=_ab.get("salience_multiplier", 1.3),
            )
        if _e("interoception_agent"):
            _thermal_thresh = self._config.get("brain", {}).get("thermal_throttle_temp", 75.0)
            _gr = self._config.get("gemini_robotics", {})
            candidates["interoception_agent"] = InteroceptionAgent(
                bus,
                thermal_eco_threshold=float(_thermal_thresh),
                thinking_budget_default=_gr.get("thinking_budget_default", 1024),
                thinking_budget_eco=_gr.get("thinking_budget_eco", 256),
            )
        if _e("metacognition_agent"):
            candidates["metacognition_agent"] = MetacognitionAgent(
                bus,
                soul=self._soul,
                persist_knowledge_gaps=self._neuroplasticity_cfg.get("persist_knowledge_gaps", True),
            )
        if _e("theory_of_mind_agent"):
            candidates["theory_of_mind_agent"] = TheoryOfMindAgent(bus, self._llm, self._soul)
        if _e("temporal_reasoning_agent"):
            candidates["temporal_reasoning_agent"] = TemporalReasoningAgent(bus, self._llm, self._episodic)
        if _e("intrinsic_motivation_agent"):
            candidates["intrinsic_motivation_agent"] = IntrinsicMotivationAgent(bus, self._llm, self._episodic, self._semantic)
        if _e("ideation_agent"):
            candidates["ideation_agent"] = IdeationAgent(bus, self._llm, self._semantic, self._episodic)
        if _e("imagination_agent"):
            candidates["imagination_agent"] = ImaginationAgent(bus, self._llm, self._episodic, self._working)

        # ── V6.0 Self-Evolution agents ────────────────────────────────────────
        if _e("synthesis_agent") and self._config.get("synthesis_agent", {}).get("enabled", True):
            from pathlib import Path
            candidates["synthesis_agent"] = SynthesisAgent(
                bus, self._llm, self._soul,
                sandbox_dir=Path("brain/sandbox"),
                code_health_path=Path("data/CODE_HEALTH.md"),
                crash_threshold=self._synthesis_crash_threshold,
                confidence_threshold=self._synthesis_confidence_threshold,
                orchestrator_ref=self,
            )
        if _e("tester_agent") and self._config.get("tester_agent", {}).get("enabled", True):
            from pathlib import Path
            candidates["tester_agent"] = TesterAgent(
                bus,
                sandbox_dir=Path("brain/sandbox"),
                timeout_seconds=self._config.get("tester_agent", {}).get("timeout_seconds", 10),
            )
        if _e("mirror_agent") and self._config.get("mirror_agent", {}).get("enabled", False):
            candidates["mirror_agent"] = MirrorAgent(
                bus, self._llm, self._soul,
                smolvlm2=self._smolvlm2,
            )

        # V10.0 — Sentient Embodied Autonomous Brain agents
        if _e("goal_stack_agent"):
            candidates["goal_stack_agent"] = GoalStackAgent(bus)
        if _e("proprioception_agent"):
            candidates["proprioception_agent"] = ProprioceptionAgent(
                bus, world_model=self._world_model,
            )
        if _e("learning_agent"):
            candidates["learning_agent"] = LearningAgent(
                bus, self._llm, self._semantic, mcp=self._mcp_tools,
            )
        if _e("binding_agent"):
            candidates["binding_agent"] = BindingAgent(
                bus, world_model=self._world_model,
            )

        self._agents = candidates

        # Inject shared pattern_registry into DreamAgent for reflex distillation
        dream = candidates.get("dream_agent")
        if dream and hasattr(dream, "set_pattern_registry"):
            dream.set_pattern_registry(self._pattern_registry)

        # Wire HW capability flags into CognitionAgent (Stream 4 proprioception)
        cog = candidates.get("cognition_agent")
        if cog and hasattr(cog, "_hw_caps"):
            cog._hw_caps = {
                "has_camera":            getattr(self._hw, "has_camera", False),
                "has_microphone":        getattr(self._hw, "has_microphone", False),
                "has_speaker":           getattr(self._hw, "has_speaker", False),
                "has_pan_tilt":          getattr(self._hw, "has_pan_tilt", False),
                "has_differential_drive": getattr(self._hw, "has_differential_drive", False),
                "has_arduino":           getattr(self._hw, "has_arduino", False),
                "arduino_port":          getattr(self._hw, "arduino_port", ""),
                "has_display":           getattr(self._hw, "has_display", False),
            }

        # Inject SemanticMemory into MetacognitionAgent for NEURO_REWEIGHT corrections
        metacog = candidates.get("metacognition_agent")
        if metacog and hasattr(metacog, "set_semantic"):
            metacog.set_semantic(self._semantic)

        # V10.0 — Wire SemanticMemory into MCPToolRegistry for semantic_lookup epistemic tool
        if hasattr(self._mcp_tools, "_semantic"):
            self._mcp_tools._semantic = self._semantic

        disabled = [name for name in (
            "vision_agent", "auditory_agent", "sensory_agent", "language_agent",
            "cognition_agent", "reasoning_agent", "logic_agent", "emotion_engine",
            "motor_agent", "display_agent", "speech_agent", "behavior_agent",
            "curiosity_agent", "verifier_agent", "dream_agent", "planner_agent",
            "attention_agent", "interoception_agent", "metacognition_agent",
            "theory_of_mind_agent", "temporal_reasoning_agent", "intrinsic_motivation_agent",
            "ideation_agent", "imagination_agent",
            "synthesis_agent", "tester_agent", "mirror_agent",
            # V10.0 new agents
            "goal_stack_agent", "proprioception_agent", "learning_agent", "binding_agent",
        ) if not _e(name)]
        if disabled:
            log.info("Agents disabled via agents.yaml: %s", ", ".join(disabled))

    async def start(self) -> None:
        self._running = True
        self._build_agents()

        # Inject watchdog so agents can call _beat() on every publish
        for agent in self._agents.values():
            agent.set_watchdog(self._watchdog)

        for name, agent in self._agents.items():
            task = asyncio.create_task(agent.start(), name=name)
            self._tasks[name] = task
            self._watchdog.register(name, agent.start)
            self._watchdog.set_task(name, task)

        asyncio.create_task(self._message_loop(), name="orchestrator-loop")
        asyncio.create_task(self._watchdog.start(), name="watchdog")
        asyncio.create_task(self._pulse_loop(), name="gwt-pulse")
        asyncio.create_task(self._deliberative_loop(), name="deliberative-loop")

        # V10.0 — WorldModel persistence: load last snapshot on boot
        from pathlib import Path as _Path
        _world_snapshot = _Path(self._config.get("memory", {}).get(
            "world_snapshot", "data/world_snapshot.json"
        ))
        try:
            await self._world_model.load(_world_snapshot)
        except Exception as _we:
            log.warning("Orchestrator: WorldModel load failed (fresh start): %s", _we)
        # V10.0 — WorldModel autosave every 5 minutes
        asyncio.create_task(
            self._world_model._autosave_loop(_world_snapshot, interval_s=300.0),
            name="worldmodel-autosave",
        )

        # Restore last session context so Brain remembers recent conversations
        if self._config.get("memory", {}).get("restore_on_startup", True):
            await self._restore_last_session()

        log.info(f"Orchestrator started — {len(self._agents)} agents running")

    async def _restore_last_session(self) -> None:
        """Load recent episodic events into working memory on boot."""
        try:
            recent = self._episodic.get_recent(n=40)
            if not recent:
                return

            # Restore last compact summary if one exists
            compact = next(
                (e for e in reversed(recent) if e.get("event_type") == "context_compact"),
                None,
            )
            if compact:
                summary_text = compact["content"].replace("Context compacted: ", "", 1)
                self._working.replace_with_summary(
                    f"[Previous session] {summary_text}", keep_last=0
                )
                log.info("Restored last session compact summary into working memory")

            # Restore last N user/brain exchanges verbatim
            n = self._config.get("memory", {}).get("restore_exchanges", 6)
            exchanges = [
                e for e in recent
                if e.get("event_type") in ("speech", "response")
            ][-n:]
            for e in exchanges:
                role = "user" if e.get("actor") == "user" else "assistant"
                self._working.add_to_context(role, e.get("content", ""))

            if exchanges:
                log.info(f"Restored {len(exchanges)} recent exchanges into working memory")
        except Exception as ex:
            log.warning(f"Session restore failed (non-fatal): {ex}")

    async def _message_loop(self) -> None:
        while self._running:
            try:
                msg = await asyncio.wait_for(self._bus.get(), timeout=1.0)

                # GWT v5.0 — Two-path routing:
                # 1. Reflexive: hardware/safety/lifecycle → route immediately
                # 2. Cognitive: sensory/perception/intent → buffer in workspace for Pulse
                if msg.type in _REFLEXIVE_TYPES or (msg.target and msg.target != "orchestrator"):
                    await self._route(msg)
                else:
                    # COGNITIVE PATH (Thalamocortical Loop)
                    # Messages compete for the Spotlight in the next Pulse cycle.
                    # AttentionAgent scores only — routing happens exclusively via Pulse.
                    attn = self._agents.get("attention_agent")
                    salience = (
                        attn._score_salience(msg)
                        if attn and hasattr(attn, "_score_salience")
                        else 0.5
                    )
                    if salience >= self._salience_threshold:
                        async with self._workspace_lock:
                            # Inverted salience: lower value = higher priority (PriorityQueue convention)
                            self._workspace_pending.append((1.0 - salience, msg))

                # Broadcast to WebSocket subscribers regardless of path
                for sub_queue in list(self._ws_subscribers):
                    try:
                        sub_queue.put_nowait(msg)
                    except asyncio.QueueFull:
                        pass
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Orchestrator loop error: {e}")

    def get_agent(self, name: str):
        """Return a live agent instance by name (used by SynthesisAgent for context)."""
        return self._agents.get(name)

    async def _trigger_synthesis(self, agent_name: str, error_text: str) -> None:
        await self._bus.put(AgentMessage(
            type=MessageType.NEURO_SYNTHESIS,
            source="orchestrator",
            data={"agent": agent_name, "error": error_text},
            priority=1,
        ))
        log.info("Orchestrator: NEURO_SYNTHESIS triggered for %s", agent_name)

    def _get_frame_b64(self) -> str:
        """Return the latest camera frame as a base64-encoded JPEG string.

        Called by ReasoningAgent before each LLM inference so Gemini Robotics ER
        can see the live camera feed inline. Returns '' when no frame is available.
        """
        import base64
        import io

        va = self._agents.get("vision_agent")
        if va is None:
            return ""
        frame = getattr(va, "_current_frame", None)
        if frame is None:
            return ""
        try:
            import cv2
            from PIL import Image
            # OpenCV uses BGR; PIL's fromarray() expects RGB — convert first
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Resize to max 512px — Gemini handles smaller images faster
            w, h = img.size
            max_dim = 512
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            log.debug("Orchestrator: frame encode failed: %s", e)
            return ""

    async def _hot_load_module(self, data: dict) -> None:
        """Hot-load a validated sandbox module onto a live agent via importlib."""
        import importlib.util, types
        from pathlib import Path
        _PROTECTED = frozenset({"orchestrator", "tester_agent", "synthesis_agent"})
        target_name = data.get("target_agent", "")
        fix_type = data.get("fix_type", "method_fix")

        # Reflexes are stored in SKILLS.json — no module patching needed
        if fix_type == "reflex":
            reflex = {
                "id": data.get("reflex_id", ""),
                "trigger_phrase": data.get("intent", ""),
                "trigger_embedding": data.get("trigger_embedding", []),
                "python_code": data.get("python_code", ""),
                "bypass_llm": True,
                "invocations": 0,
                "created": __import__("datetime").datetime.utcnow().isoformat(),
                "source_pattern_count": data.get("source_pattern_count", 0),
            }
            self._soul.add_reflex(reflex)
            log.info("Orchestrator: reflex '%s' stored in SKILLS.json", reflex["id"])
            return

        if target_name in _PROTECTED:
            log.warning("Orchestrator: hot-load blocked — %s is a protected agent", target_name)
            return

        agent = self._agents.get(target_name)
        file_path = data.get("file_path", "")
        if not file_path or not Path(file_path).exists():
            log.error("Orchestrator: hot-load file not found: %s", file_path)
            return

        # Pause agent during patching to prevent race conditions
        was_running = getattr(agent, "_running", False)
        if agent:
            agent._running = False
            await asyncio.sleep(0.1)

        try:
            # Snapshot DNA before any change
            synth = self._agents.get("synthesis_agent")
            if synth and hasattr(synth, "_snapshot_dna"):
                synth._snapshot_dna()

            spec = importlib.util.spec_from_file_location("_hotfix", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            method_name = data.get("method_name", "")
            if agent and method_name and hasattr(module, method_name):
                new_fn = getattr(module, method_name)
                setattr(agent, method_name, types.MethodType(new_fn, agent))
                log.info("Orchestrator: hot-loaded %s.%s", target_name, method_name)
            else:
                log.warning("Orchestrator: hot-load — method '%s' not found in module", method_name)
        except Exception as e:
            log.error("Orchestrator: hot-load failed, rolling back DNA: %s", e)
            self._soul.rollback_dna()
        finally:
            if agent and was_running:
                agent._running = True

    async def _check_synthesis_triggers(self) -> None:
        """Every 10 pulses: check MetacognitionAgent confidence for synthesis trigger."""
        metacog = self._agents.get("metacognition_agent")
        if metacog and hasattr(metacog, "_last_confidence"):
            if metacog._last_confidence < self._synthesis_confidence_threshold:
                log.info("Orchestrator: low metacog confidence=%.2f → NEURO_SYNTHESIS",
                         metacog._last_confidence)
                await self._trigger_synthesis("metacognition_agent", "low_confidence")

    async def _run_external_script(self, data: dict) -> None:
        """V8.0 — Execute a validated external script in a subprocess."""
        import subprocess as _sp
        from pathlib import Path as _Path

        file_path = _Path(data["file_path"])
        script_type = data.get("script_type", "python")
        task = data.get("task", "")

        if script_type == "python":
            cmd = ["python", str(file_path)]
        elif script_type == "arduino":
            port = getattr(self._hw, "arduino_port", "")
            cli = self._config.get("arduino", {}).get("cli_path", "arduino-cli")
            cmd = [cli, "compile", "--upload", "-p", port, str(file_path)]
        elif script_type == "html":
            import shutil
            target_path = _Path("api/static/display.html")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, target_path)
            log.info("Orchestrator: hot-loaded HTML UI update → %s", target_path)
            cmd = ["python", "-c", "print('HTML updated')"]
        else:
            log.warning("Orchestrator: unknown script_type '%s'", script_type)
            return

        timeout = self._config.get("task_executor", {}).get("script_timeout_seconds", 30)

        loop = asyncio.get_event_loop()
        stdout_text = ""
        stderr_text = ""
        try:
            result = await loop.run_in_executor(
                None,
                lambda: _sp.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env={},
                                cwd=str(file_path.parent)),
            )
            status = "success" if result.returncode == 0 else "failure"
            stdout_text = result.stdout
            stderr_text = result.stderr
        except Exception as e:
            status = "failure"
            stderr_text = str(e)

        log.info("Orchestrator: external script %s → %s", file_path.name, status)

        await self._bus.put(AgentMessage(
            type=MessageType.TASK_OUTCOME,
            source="orchestrator",
            data={"status": status, "stdout": stdout_text, "stderr": stderr_text,
                  "script": str(file_path), "task": task},
            priority=7,
        ))

        if status == "failure":
            await self._bus.put(AgentMessage(
                type=MessageType.DRIVER_SYNTHESIS,
                source="orchestrator",
                data={"task": task, "error": stderr_text,
                      "script_type": script_type, "previous_script": str(file_path)},
                priority=6,
            ))

    async def _route(self, msg: AgentMessage) -> None:
        # V6.0 — REFLEX_READY: handled inline (no agent dispatch)
        if msg.type == MessageType.REFLEX_READY:
            fix_type = msg.data.get("fix_type", "method_fix")
            if fix_type == "external_script":
                await self._run_external_script(msg.data)
            else:
                await self._hot_load_module(msg.data)
            return

        # V8.0 — TASK_OUTCOME: watchdog for repeated failures → DNA rollback
        if msg.type == MessageType.TASK_OUTCOME:
            if msg.data.get("status") == "failure":
                task = msg.data.get("task", "unknown")
                self._task_failure_registry[task] = self._task_failure_registry.get(task, 0) + 1
                if self._task_failure_registry[task] >= self._max_task_failures:
                    log.error("Orchestrator: task '%s' failed %d× — rolling back DNA",
                              task, self._max_task_failures)
                    self._soul.rollback_dna()
                    self._task_failure_registry[task] = 0
            else:
                task = msg.data.get("task", "")
                if task in self._task_failure_registry:
                    self._task_failure_registry[task] = 0

        # Update CognitionAgent active goal (Stream 4 injection) from planner signals
        if msg.type == MessageType.TASK_EXECUTE:
            cog = self._agents.get("cognition_agent")
            if cog and hasattr(cog, "_active_goal"):
                cog._active_goal = msg.data.get("goal", msg.data.get("text", ""))
        if msg.type == MessageType.TASK_OUTCOME and msg.data.get("status") == "success":
            cog = self._agents.get("cognition_agent")
            if cog and hasattr(cog, "_active_goal"):
                cog._active_goal = ""

        # V6.0 — SYSTEM_ERROR accumulation → synthesis trigger
        if msg.type == MessageType.SYSTEM_ERROR:
            _PROTECTED = frozenset({"orchestrator", "tester_agent", "synthesis_agent"})
            err_agent = msg.data.get("agent", msg.source)
            if err_agent not in _PROTECTED:
                self._error_registry[err_agent] = self._error_registry.get(err_agent, 0) + 1
                synth = self._agents.get("synthesis_agent")
                session_count = (getattr(synth, "_synthesis_count", {}) or {}).get(err_agent, 0)
                if (self._error_registry[err_agent] >= self._synthesis_crash_threshold
                        and session_count < self._session_synthesis_limit):
                    await self._trigger_synthesis(err_agent, msg.data.get("error", ""))
                    self._error_registry[err_agent] = 0
                elif session_count >= self._session_synthesis_limit:
                    log.warning("Orchestrator: synthesis session limit reached for %s", err_agent)

        # V6.0 — COGNITION_RESPONSE pattern tracking for reflex distillation
        if msg.type == MessageType.COGNITION_RESPONSE:
            intent = msg.data.get("intent", "general")
            response = msg.data.get("response", "")
            if intent and response and not msg.data.get("via_reflex"):
                self._pattern_registry.setdefault(intent, []).append(response)
                self._pattern_registry[intent] = self._pattern_registry[intent][-20:]

        # V9.8 — [EXECUTE:] Action Token Interceptor
        # Strip code-action tokens before text reaches SpeechAgent and fire synthesis.
        if msg.type == MessageType.COGNITION_RESPONSE:
            import re as _re
            raw = msg.data.get("response", "") or msg.data.get("text", "")
            _exec_match = _re.search(
                r"\[EXECUTE:\s*([A-Z_]+)(?:\s*\|\s*PATH:\s*([^\]]+))?\]", raw, _re.IGNORECASE
            )
            if _exec_match:
                action    = _exec_match.group(1).upper()
                path_hint = (_exec_match.group(2) or "").strip()
                # Extract [CONFIRM: ...] spoken text or everything after the token
                confirm_match = _re.search(r"\[CONFIRM:\s*([^\]]+)\]", raw, _re.IGNORECASE)
                spoken = confirm_match.group(1).strip() if confirm_match else \
                         _re.sub(r"\[[^\]]+\]", "", raw).strip()
                # Replace raw response with clean spoken text only
                msg.data["response"] = spoken
                msg.data["text"]     = spoken
                from brain.utils.repo_map import resolve_target_file
                target_file = path_hint or resolve_target_file(raw)
                log.info("Orchestrator: [EXECUTE:%s] → synthesis target=%s", action, target_file)
                asyncio.create_task(self._bus.put(AgentMessage(
                    type=MessageType.NEURO_SYNTHESIS,
                    source="orchestrator",
                    data={"trigger": action, "intent_text": raw[:200], "target_file": target_file},
                    priority=8,
                )))

        # V9.0 — MCP Tool Dispatch: if LLM issued tool calls, dispatch them now
        if msg.type == MessageType.COGNITION_RESPONSE:
            raw_response = msg.data.get("response", "")
            clean_text, tool_calls = self._llm.parse_tool_calls(raw_response)
            if tool_calls:
                log.info("Orchestrator: MCP tool calls detected (%d calls)", len(tool_calls))
                async def _dispatch_tool_calls():
                    for tc in tool_calls:
                        name = tc.get("name", "")
                        args = tc.get("args", tc.get("arguments", {}))
                        result = await self._mcp_tools.dispatch(name, args)
                        log.info("MCP dispatch result: tool=%s result=%s", name, result)
                asyncio.create_task(_dispatch_tool_calls())

        # V7.0 — THINKING_BUDGET: update ER token budget from InteroceptionAgent
        if msg.type == MessageType.THINKING_BUDGET:
            new_budget = int(msg.data.get("budget", self._thinking_budget))
            if new_budget != self._thinking_budget:
                log.info("Orchestrator: thinking_budget %d → %d (trigger=%s)",
                         self._thinking_budget, new_budget, msg.data.get("trigger", "thermal"))
                self._thinking_budget = new_budget
            return

        # V7.0 — RE_PLAN: mark plan as active so deliberative loop fires
        if msg.type == MessageType.RE_PLAN:
            self._plan_active = True

        # GWT body-state wiring: keep WorldModel in sync with intero state
        if msg.type == MessageType.INTERO_STATE and self._world_model:
            await self._world_model.update_intero(msg.data)

        # N3c: Store internal monologue thought in WorldModel for /world endpoint
        if msg.type == MessageType.COGNITION_THOUGHT and self._world_model:
            await self._world_model.update_thought(msg.data.get("thought", ""))

        # Brain Fog: track ECO mode — suppresses non-essential agents under hardware stress
        if msg.type == MessageType.BEHAVIOR_CHANGE and "eco" in msg.data:
            new_eco = bool(msg.data["eco"])
            if new_eco != self._eco_mode:
                self._eco_mode = new_eco
                log.info("Orchestrator: Brain Fog → ECO=%s (trigger=%s)",
                         new_eco, msg.data.get("trigger", "unknown"))

        # Directed message: deliver only to the named target
        if msg.target and msg.target != "orchestrator":
            agent = self._agents.get(msg.target)
            if agent and msg.target != msg.source:
                try:
                    await agent.handle(msg)
                except Exception as e:
                    log.error(f"Agent {msg.target} handle error: {e}")
            await self._log_event(msg)
            return

        # Broadcast: deliver to all routing table entries
        targets = ROUTING.get(msg.type, [])
        for target_name in targets:
            if target_name == msg.source:
                continue
            # Brain Fog: suppress imagination/ideation during hardware overload (configurable)
            if self._brain_fog_enabled and self._eco_mode and target_name in _ECO_BLOCKED_AGENTS:
                continue
            agent = self._agents.get(target_name)
            if agent:
                try:
                    result = await agent.handle(msg)
                    if target_name == "logic_agent" and result is None:
                        break
                except Exception as e:
                    log.error(f"Agent {target_name} handle error: {e}")

        await self._log_event(msg)

    async def _log_event(self, msg: AgentMessage) -> None:
        from brain.memory.episodic_memory import Episode
        loggable_types = {
            MessageType.PERCEPTION_SPEECH,
            MessageType.ACTION_SPEAK,
            MessageType.COGNITION_RESPONSE,
            MessageType.EMOTION_CHANGE,
        }
        if msg.type not in loggable_types:
            return
        emotion = self._agents.get("emotion_engine")
        current_emotion = emotion.current_emotion if emotion else "NEUTRAL"

        content = ""
        actor = "robot"
        if msg.type == MessageType.PERCEPTION_SPEECH:
            content = f"User said: {msg.data.get('text', '')}"
            actor = "user"
        elif msg.type == MessageType.ACTION_SPEAK:
            content = f"Brain said: {msg.data.get('text', '')}"
        elif msg.type == MessageType.EMOTION_CHANGE:
            content = f"Emotion changed: {msg.data.get('from')} → {msg.data.get('to')}"
            actor = "environment"

        if content:
            ep = Episode(
                actor=actor,
                event_type=msg.type.value,
                content=content,
                emotion=current_emotion,
                outcome="success" if msg.data.get("success") else "neutral",
            )
            self._episodic.log_event(ep)

    async def _deliberative_loop(self) -> None:
        """V7.0 Deliberative Pulse — 2Hz loop active only when ER planning is running.

        Polls PlannerAgent._active_task; when a plan is active it emits a
        WORLD_UPDATE so CognitionAgent gets fresh scene context between motor steps.
        Idles at 2s intervals when no plan is active to avoid wasting CPU.
        """
        while self._running:
            planner = self._agents.get("planner_agent")
            self._plan_active = bool(
                planner and getattr(planner, "_active_task", "") and
                getattr(planner, "_executing", False)
            )
            if self._plan_active and self._world_model:
                try:
                    snapshot = await self._world_model.snapshot()
                    await self._bus.put(AgentMessage(
                        type=MessageType.WORLD_UPDATE,
                        source="orchestrator",
                        data=snapshot,
                        priority=8,
                    ))
                except Exception as e:
                    log.debug("deliberative_loop: world_update failed: %s", e)
                await asyncio.sleep(0.5)   # 2Hz during active planning
            else:
                await asyncio.sleep(2.0)   # idle — no plan running

    async def _pulse_loop(self) -> None:
        """GWT Pulse — 10Hz heartbeat.

        Each tick:
          1. Drains the workspace buffer, sorts by salience (highest first).
          2. Routes the top-N (spotlight_size) messages to their cognitive targets.
          3. Discards the rest — they lost the competition for conscious access.
          4. Updates the spotlight state for the /workspace API.
        """
        interval = 1.0 / max(self._pulse_hz, 1.0)
        while self._running:
            await asyncio.sleep(interval)
            await self._broadcast_spotlight()

    async def _broadcast_spotlight(self) -> None:
        """Drain workspace buffer, elect top-N winners, route them."""
        self._pulse_count += 1
        if self._pulse_count % 10 == 0:
            await self._check_synthesis_triggers()

        async with self._workspace_lock:
            pending = list(self._workspace_pending)
            self._workspace_pending.clear()

        if not pending:
            return

        # Sort ascending by inverted salience — lowest value (= highest salience) wins
        pending.sort(key=lambda t: t[0])
        winners = pending[: self._spotlight_size]
        dropped = len(pending) - len(winners)
        if dropped > 0:
            log.debug("GWT Pulse: %d messages gated out (below spotlight cutoff)", dropped)

        # Update spotlight state for API
        attn = self._agents.get("attention_agent")
        if attn and hasattr(attn, "get_spotlight"):
            self._spotlight = attn.get_spotlight(self._spotlight_size)

        # Route spotlight winners to their cognitive targets
        for inv_salience, msg in winners:
            try:
                await self._route(msg)
                log.debug("GWT Spotlight: routed %s (salience=%.2f) from %s",
                          msg.type.value, 1.0 - inv_salience, msg.source)
            except Exception as exc:
                log.error("GWT Pulse routing error: %s", exc)

    def get_spotlight(self) -> list[dict]:
        """Return current Spotlight (top-N salient messages) for the /workspace API."""
        return list(self._spotlight)

    def subscribe_ws(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._ws_subscribers.append(q)
        return q

    def unsubscribe_ws(self, q: asyncio.Queue) -> None:
        try:
            self._ws_subscribers.remove(q)
        except ValueError:
            pass

    async def inject(self, msg: AgentMessage) -> None:
        await self._bus.put(msg)

    async def stop(self) -> None:
        self._running = False
        self._watchdog.stop()

        # V10.0 — Save WorldModel snapshot before shutdown
        from pathlib import Path as _Path
        _world_snapshot = _Path(self._config.get("memory", {}).get(
            "world_snapshot", "data/world_snapshot.json"
        ))
        try:
            await self._world_model.save(_world_snapshot)
            log.info("Orchestrator: WorldModel snapshot saved to %s", _world_snapshot)
        except Exception as _we:
            log.warning("Orchestrator: WorldModel save on shutdown failed: %s", _we)

        # Stop all agents in parallel with a 3 s timeout each so a stuck camera
        # or audio driver cannot block the entire shutdown sequence.
        stop_coros = [
            asyncio.wait_for(agent.stop(), timeout=3.0)
            for agent in self._agents.values()
        ]
        await asyncio.gather(*stop_coros, return_exceptions=True)
        for task in self._tasks.values():
            task.cancel()
        await self._llm.close()
        self._episodic.close()
        self._semantic.close()
        log.info("Orchestrator stopped")

