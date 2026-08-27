# Project-ABC — Changelog

---

## V7.0 — Natively Embodied Autonomous Brain (2026-04-25)

### Dual-Brain Framework

| Component | File | Description |
|---|---|---|
| `er_reason()` | `brain/llm/llm_router.py` | High-Level Brain — Gemini ER structured plan + success_estimate dict |
| `vla_act()` | `brain/llm/llm_router.py` | Low-Level Brain — Gemini VLA motor token; falls back to regex parse |
| `VLAMapper` | `brain/hardware/vla_mapper.py` | Stateless ACTION_MAP + `parse()` + `fallback_parse()` |
| `_gaze_snap()` | `brain/agents/motor_agent.py` | Instant SPATIAL_POINT (x,y) → pan/tilt; bypasses saccade smoothing |
| `_handle_vla_token()` | `brain/agents/motor_agent.py` | VLAMapper dispatch → WheelDriver / MotorDriver |

### Spatial Intelligence Layer

| Component | File | Description |
|---|---|---|
| `spatial_point(query, frame)` | `brain/hardware/vision_processor.py` | Normalised (x,y) centre of detected object via detect() bbox |
| `detect_3d(frame)` | `brain/hardware/vision_processor.py` | Monocular depth heuristic → `{label, x, y, z_est, w, h, d_est}` list |
| `update_spatial(objects)` | `brain/memory/world_model.py` | Store 3D object list from VisionProcessor |
| `get_nearest(label)` | `brain/memory/world_model.py` | Return closest matching object by z_est |
| `_visual_servo(target_label)` | `brain/agents/vision_agent.py` | 5s loop emitting SPATIAL_POINT until `|x-0.5| < 0.05` |
| "look at X" speech trigger | `brain/agents/vision_agent.py` | Regex match in PERCEPTION_SPEECH → `_visual_servo()` task |

### Long-Horizon Planning

| Component | File | Description |
|---|---|---|
| `_decompose_with_er()` | `brain/agents/planner_agent.py` | ER-powered task decomposition; emits RE_PLAN if success_estimate < 0.4 |
| `_estimate_success()` | `brain/agents/planner_agent.py` | Per-step success probability from ER; up to 3 RE_PLAN retries |
| COGNITION_INTENT routing | `brain/agents/planner_agent.py` | Intercepts `intent=task/plan/execute` for ER before pattern matching |
| RE_PLAN handler | `brain/agents/planner_agent.py` | Triggers fresh ER decomposition when success drops below threshold |
| `_deliberative_loop()` | `brain/orchestrator.py` | 2Hz loop during active plan; WorldModel snapshot → CognitionAgent |

### Dynamic Thinking Budget

| Component | File | Description |
|---|---|---|
| `_last_confidence` | `brain/agents/metacognition_agent.py` | Field polled by Orchestrator synthesis trigger |
| THINKING_BUDGET emission | `brain/agents/interoception_agent.py` | Emits 1024→256 token budget on thermal ECO transition |
| THINKING_BUDGET handler | `brain/orchestrator.py` | Inline update of `_thinking_budget`; routed before agent dispatch |
| `thinking_budget_default/eco` | `config/config.yaml` | Configurable under `gemini_robotics:` block |

### New MessageTypes (V7.0)

```python
SPATIAL_POINT       = "spatial.point"           # normalised (x,y) for gaze snap
SPATIAL_3D          = "spatial.3d"              # 3D bounding box list
RE_PLAN             = "plan.replan"             # success_estimate < threshold → re-plan
VLA_CONTROL_TOKEN   = "vla.control_token"       # Gemini VLA action token
THINKING_BUDGET     = "cognition.thinking_budget"  # ER token budget
SUCCESS_ESTIMATE    = "plan.success_estimate"   # per-step success probability
```

### New Config Block

```yaml
gemini_robotics:
  er_model: "gemini-2.0-flash"
  vla_model: "gemini-2.0-flash"
  thinking_budget_default: 1024
  thinking_budget_eco: 256
  success_threshold: 0.4
  vla_confidence_threshold: 0.3
  servo_timeout_s: 5.0
  servo_center_tolerance: 0.05
```

### Bug Fixes

- `MirrorAgent._mirror_action()` — replaced nonexistent `smolvlm2.describe()` call with correct `push_frame()` + `scan_async()` API

---

## V6.0 — Extreme Self-Evolution (2026-04-24)

### Phase 1 — Differential Wheel Drive

| Component | File | Description |
|---|---|---|
| `WheelDriver` | `brain/hardware/wheel_driver.py` | L298N H-Bridge driver; linear PWM ramp-up (50ms) prevents mechanical shock |
| `HardwareCapabilities.has_differential_drive` | `brain/hardware/hw_detector.py` | Auto-detected from `wheel_drive.left_pins/right_pins` config |
| Hysteresis gaze-nav | `brain/agents/motor_agent.py` | Trigger rotation at pan >45°; only clear when pan <15° — prevents jitter |
| `_gaze_rotating` flag | `brain/agents/motor_agent.py` | Prevents re-triggering while rotation is in progress |
| `WHEELED_ROTATE` MessageType | `brain/agents/base_agent.py` | Differential rotation command |

### Phase 2 — Neocortical Synthesis

| Component | File | Description |
|---|---|---|
| `SynthesisAgent` | `brain/agents/synthesis_agent.py` | Rewrites failing methods via Gemini; AST guardrail (FORBIDDEN_IMPORTS); attr-aware prompt |
| `TesterAgent` | `brain/agents/tester_agent.py` | Subprocess sandbox validator; empty `env={}`; never uses exec/eval |
| `_guardrail_check()` | `brain/agents/synthesis_agent.py` | `ast.parse()` + `ast.walk()` blocks forbidden imports before any file write |
| `_snapshot_dna()` | `brain/agents/synthesis_agent.py` | SHA256 of all `brain/*.py` (excluding sandbox) → `data/protected/dna.json` |
| Hot-load with agent pause | `brain/orchestrator.py` | `_running=False` → 100ms sleep → importlib patch → `finally` resume |
| Brainstem protection | `brain/orchestrator.py` | `orchestrator`, `tester_agent`, `synthesis_agent` can never be hot-loaded |
| Session synthesis limit | `brain/agents/synthesis_agent.py` | Max 5 re-syntheses per agent; 6th attempt is blocked and logged |
| Error registry | `brain/orchestrator.py` | `_error_registry[agent]` accumulates SYSTEM_ERRORs; resets after synthesis trigger |
| `CODE_HEALTH.md` | `data/CODE_HEALTH.md` | Markdown table: agent, crash count, error, outcome, blacklist status |

### Phase 2 — Native I2C Device Discovery (V6.0 → V7.0 bridge)

| Component | File | Description |
|---|---|---|
| `_discover_device(i2c_addr)` | `brain/agents/synthesis_agent.py` | Matches 11 known I2C device addresses (ADS1115, MPU6050, BMP280, VL53L0X …) |
| `_synthesize_driver(device_info)` | `brain/agents/synthesis_agent.py` | Generates smbus2 driver class via er_reason(); emits CODE_VALIDATED |
| `device_discovery` mode | `brain/agents/synthesis_agent.py` | NEURO_SYNTHESIS with `mode="device_discovery"` + `i2c_addr` int |

### Phase 3 — Semantic Muscle Memory

| Component | File | Description |
|---|---|---|
| SKILLS.json v2.0 | `data/SKILLS.json` | Schema: `{skills, reflexes, version, last_updated}` |
| `add_reflex()` | `brain/memory/soul_manager.py` | Appends to `_skills["reflexes"]`; enforces max 50; saves JSON |
| `_check_reflexes()` | `brain/agents/reasoning_agent.py` | Embeds query; cosine similarity ≥0.92 → exec reflex; bypasses LLM |
| `_compile_reflexes_from_pattern()` | `brain/agents/synthesis_agent.py` | Generates trigger_pattern + result code; stores embedding |
| `_distill_reflexes()` | `brain/agents/dream_agent.py` | Emits NEURO_SYNTHESIS for intents with ≥5 responses; clears registry |
| `_maintain_reflexes()` | `brain/agents/dream_agent.py` | Prunes zero-invocation after 7 days; promotes ≥100 to `brain/reflexes/` |
| Pattern registry | `brain/orchestrator.py` | `_pattern_registry` tracks COGNITION_RESPONSE intents; injected into DreamAgent |

### Phase 4 — MirrorAgent

| Component | File | Description |
|---|---|---|
| `MirrorAgent` | `brain/agents/mirror_agent.py` | Grounded observational learning via SmolVLM2 |
| `get_primitive_skills()` | `brain/memory/soul_manager.py` | Returns canonical motor primitive signatures |
| Grounded prompt | `brain/agents/mirror_agent.py` | Constrains SmolVLM2 to only known primitive names; parses PLAN_STEPs |

### New MessageTypes (V6.0)

```python
WHEELED_ROTATE  = "motor.wheeled_rotate"
NEURO_SYNTHESIS = "neuro.synthesis"
CODE_VALIDATED  = "neuro.code_validated"
REFLEX_READY    = "neuro.reflex_ready"
```

---

## V5.1 — Cognitive Upgrades (2026-04-22)

- Internal Monologue (two-stage LLM: think → respond)
- Brain Fog: ECO mode suppresses imagination/ideation agents
- Affective Bias: salience ×1.3 when SURPRISED/FRUSTRATED; brevity suffix when FRUSTRATED
- Neuroplasticity: correction detection, scenario preferences, predicted needs, knowledge gap persistence
- Emotional pre-gate: skips routine low-emotion low-importance episodes
- Global Workspace Theory (10Hz Pulse, spotlight top-3, salience threshold 0.4)
- SurpriseHeuristic: ImaginationAgent prediction delta >30% → WORLD_SURPRISE
- SmolVLM2-500M via OpenVINO (Intel Iris Xe) with 6-frame RingBuffer

---

## V5.0 — Global Workspace Theory (2026-04-17)

- AttentionAgent salience gating replacing naive broadcast
- WorldModel shared ground truth (scene, presence, gaze, audio, emotion, intero)
- SmolVLM2 continuous video loop (2.5s scan interval)
- InteroceptionAgent CPU/RAM/temp → Affective States + auto-ECO
- MetacognitionAgent confidence scoring vs WORLD.md facts
- Working memory TTL reduced from 60 min to 10 min; sensory RingBuffer 3s TTL
- Episodic retention reduced to 7 days (was 30 days)
