# Project-ABC — Robotic Brain v7.0

A self-aware, self-modifying, embodied multi-agent cognitive brain. Runs on Raspberry Pi 3B+ (production) or Intel i5-H / 16GB RAM / Intel Iris Xe (dev/WSL2).

**Cognitive Model:** Global Workspace Theory + Predictive Processing + Neuroplasticity + Dual-Brain Embodied Control

---

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo> project-abc && cd project-abc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API keys
cp config/.env.example config/.env
nano config/.env   # GOOGLE_AI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY

# 4. Download emotion GIFs
python scripts/download_emotions.py

# 5. Validate hardware
python scripts/hw_test.py

# 6. Start the brain
python -m brain

# Or with mock hardware (dev without camera/mic)
python -m brain --mock
```

---

## Architecture

### Global Workspace Theory (GWT) — 10Hz Pulse

All sensory inputs compete for a **Spotlight** of attention — only the top-3 most salient signals reach the LLM. Messages below salience 0.4 are discarded.

```
Perception → Blackboard → AttentionAgent (salience gate) → Spotlight (top-3)
           → CognitionAgent (GWT prompt: INTERO + ATTENTION + WORLD + MEMORIES + SOUL)
           → ReasoningAgent [Pass 1: think | Pass 2: respond]
           → MetacognitionAgent → VerifierAgent → SpeechAgent
```

### V7.0 Dual-Brain Embodied Control

```
High-Level Brain (ER)           Low-Level Brain (VLA)
─────────────────────           ──────────────────────
PlannerAgent._decompose_with_er()  MotorAgent._handle_vla_token()
→ er_reason() → plan steps     → vla_act() → VLAMapper.parse()
→ _estimate_success()          → WheelDriver / MotorDriver
→ RE_PLAN if prob < 0.4        → visual_servo() until |x-0.5| < 0.05
```

### V7.0 Deliberative Loop (2Hz when plan active)

```
Orchestrator._deliberative_loop() fires at 2Hz during active ER planning
→ WorldModel.snapshot() → WORLD_UPDATE → CognitionAgent gets fresh scene context
```

---

## File Structure

```
Project-ABC/
├── brain/
│   ├── agents/          # 28 specialized cognitive agents
│   │   ├── synthesis_agent.py   # V6.0 Neocortical Synthesis + device discovery
│   │   ├── tester_agent.py      # V6.0 Sandbox subprocess validator
│   │   └── mirror_agent.py      # V6.0 Grounded observational learning
│   ├── hardware/        # Hardware drivers
│   │   ├── wheel_driver.py      # V6.0 L298N H-Bridge + linear PWM ramp
│   │   └── vla_mapper.py        # V7.0 VLA action token → motor primitive map
│   ├── sandbox/         # AI-generated code zone (never imported directly)
│   ├── reflexes/        # Promoted high-invocation reflexes (permanent hardcode)
│   ├── llm/             # LLM router (Ollama → Google AI → OpenRouter → Claude)
│   ├── memory/          # Working, episodic (SQLite FTS5), semantic, soul
│   ├── utils/           # config, logger, watchdog
│   ├── orchestrator.py  # asyncio.Queue Blackboard + GWT Pulse + deliberative loop
│   └── __main__.py      # Boot sequence
├── api/
│   ├── rest_api.py      # FastAPI REST endpoints
│   └── ws_api.py        # WebSocket real-time event stream
├── config/
│   ├── config.yaml      # Runtime configuration (V6.0 + V7.0 sections added)
│   └── .env.example     # API key template
└── data/
    ├── SOUL.md          # Core personality + homeostatic drives
    ├── USER.json        # Structured user profile (OCEAN, habits, scenario prefs)
    ├── WORLD.md         # Environment facts
    ├── SKILLS.json      # Skills v2.0 — {skills, reflexes} with trigger embeddings
    ├── CODE_HEALTH.md   # Agent crash log + synthesis outcomes table
    ├── DREAM_LOG.md     # Nightly consolidation journal
    └── protected/
        └── dna.json     # SHA256 brainstem checksum (write-protected)
```

---

## API

- REST: `http://localhost:8080`
- WebSocket: `ws://localhost:8080/ws`
- Docs: `http://localhost:8080/docs`

| Method | Path | Description |
|---|---|---|
| GET | `/status` | Brain health, emotion, behavior, hardware |
| GET | `/workspace` | GWT Spotlight — top-3 salient messages |
| GET | `/drives` | Homeostatic drive levels |
| GET | `/world` | Live WorldModel snapshot (scene, presence, spatial objects, thought) |
| GET | `/memory/episodes` | Recent episodic events |
| GET | `/memory/search?q=` | FTS5 full-text memory search |
| POST | `/speak` | Inject text as ACTION_SPEAK |
| POST | `/ask` | Send a user message to the brain |
| POST | `/move` | Motor command (pan/tilt/scan/center/look_at) |
| POST | `/dream` | Trigger nightly dream cycle manually |
| GET | `/soul` | Current SOUL.md content |
| GET | `/snapshot` | Latest camera frame as JPEG |

---

## Brain Agents (28)

| Agent | Brain Region | Key Responsibility |
|---|---|---|
| VisionAgent | Visual Cortex | SmolVLM2-500M (OpenVINO), 6-frame RingBuffer, visual servo, SPATIAL_POINT/3D |
| AuditoryAgent | Auditory Cortex | Vosk STT, VAD, voice tone/RMS, back-channel acks |
| SensoryAgent | Somatosensory | GPIO sensors, touch/proximity |
| LanguageAgent | Wernicke's/Broca's | Intent classification (16 intents), entity extraction |
| CognitionAgent | Hippocampus + PFC | GWT prompt assembly, correction detection, scenario prefs, predicted needs |
| ReasoningAgent | Lateral PFC | Two-stage LLM (think→respond), semantic reflex short-circuit, emotion-aware brevity |
| LogicAgent | Basal Ganglia | Safety rules, command validation |
| VerifierAgent | Anterior Cingulate | Hallucination check, world-contradiction hedge, thought-consistency enforcement |
| EmotionEngine | Limbic System | 10-state FSM, voice tone empathy, auto-transitions |
| BehaviorAgent | Cingulate Cortex | 5-state FSM (IDLE/ATTENTIVE/FOCUSED/EXPLORING/ECO) |
| MotorAgent | Motor Cortex | Pan-tilt + WheelDriver, gaze-snap (SPATIAL_POINT), VLA token dispatch, hysteresis gaze-nav |
| DisplayAgent | Visual Output | Emotion GIF compositor, HUD overlay |
| SpeechAgent | Broca's Output | Kokoro TTS, emotion-speed mapping, volume mirroring, interrupt ack |
| CuriosityAgent | Default Mode Network | Proactive check-ins, WORLD_SURPRISE reactions, knowledge-gap questions |
| DreamAgent | Sleep Consolidation | Reflex distillation, reflex prune/promote, personality evolution, habit detection |
| PlannerAgent | Prefrontal Planning | Delayed task queue + ER-powered long-horizon decomposition + RE_PLAN |
| AttentionAgent | Posterior Parietal | GWT Gater — salience scoring, emotion-biased ×1.3 |
| InteroceptionAgent | Insula | CPU/RAM/temp → Affective States, THINKING_BUDGET emission, auto-ECO trigger |
| MetacognitionAgent | Anterior PFC | Self-correction vs WORLD.md, knowledge-gap trigger, `_last_confidence` |
| TheoryOfMindAgent | Medial PFC | User confusion/expertise model |
| TemporalReasoningAgent | Basal Ganglia | Hour-of-day patterns, causal chain analysis |
| IntrinsicMotivationAgent | OFC + Striatum | Drive hierarchy (learn/help/explore/rest) |
| IdeationAgent | DLPFC | Cross-domain synthesis, semantic neighbour retrieval |
| ImaginationAgent | Hippocampus + PFC | Predictive world-state, prediction delta for WORLD_SURPRISE |
| **SynthesisAgent** | **Neuroplasticity** | **V6.0 — rewrites failing methods, compiles reflexes, discovers I2C devices, stores Digital DNA** |
| **TesterAgent** | **Safety Circuit** | **V6.0 — subprocess sandbox validator (empty env, no exec/eval)** |
| **MirrorAgent** | **Mirror Neurons** | **V6.0 — grounded observational learning via SmolVLM2 + motor primitives** |
| BehaviorAgent | Cingulate | State machine, mood arc processing |

---

## LLM Routing

| Tier | Model | Condition |
|---|---|---|
| 1 — Simple | Ollama `llama3.2:1b` (local) | < 150 tokens, offline |
| 2 — Balanced | Ollama `llama3.2:3b` (local) | < 800 tokens, offline |
| 3 — Primary cloud | Google AI Gemini 2.0 Flash | Free tier |
| 4 — Secondary cloud | OpenRouter free | Fallback |
| 5 — Paid fallback | Claude haiku-4-5 | Requires ANTHROPIC_API_KEY |
| 6 — Offline fallback | Ollama `llama3.2:3b` | When all cloud unavailable |
| ECO mode | `llama3.2:1b` only | Auto-triggered by InteroceptionAgent |
| **ER reasoning** | **`er_reason()`** | **V7.0 — Gemini ER structured plan + success_estimate** |
| **VLA control** | **`vla_act()`** | **V7.0 — Gemini VLA motor token with VLAMapper fallback** |

Semantic cache: cosine similarity ≥ 0.97 → return cached, skip LLM (5min TTL, 200-entry deque).

---

## Memory Architecture

| Layer | Storage | TTL | Brain Equivalent |
|---|---|---|---|
| Working | Python dict (session) | 60 min | Prefrontal Cortex |
| Episodic | SQLite FTS5 (`data/episodes.db`) | 7 days | Hippocampus |
| Semantic | sqlite-vec (`data/chroma`) | Permanent | Neocortex |
| Soul | Markdown + JSON (`data/`) | Permanent | DNA / Personality |
| **Reflexes** | **SKILLS.json v2.0** | **Permanent (pruned/promoted)** | **Spinal Cord / Fast Reflex Arc** |
| **Digital DNA** | **`data/protected/dna.json`** | **Permanent (write-protected)** | **Brainstem integrity** |
| **Spatial** | **WorldModel._spatial_objects** | **Session (3D object list)** | **Parietal Cortex** |

---

## V6.0 — Self-Evolution

The brain can rewrite its own failing code at runtime.

```
SYSTEM_ERROR (×3) → SynthesisAgent → AST guardrail → sandbox .py
                  → TesterAgent (subprocess, empty env) → REFLEX_READY
                  → Orchestrator hot-load (agent pause → importlib patch → resume)
                  → DNA rollback on failure
```

| Feature | Description |
|---|---|
| **Neocortical Synthesis** | Rewrites failing agent methods via Gemini; forbidden import AST check; attr-aware prompt |
| **Semantic Muscle Memory** | Repeated LLM response patterns → cosine similarity reflex (≥0.92 → bypasses LLM) |
| **Digital DNA** | SHA256 of all `brain/*.py`; stored in `data/protected/dna.json`; auto-rollback on bad hot-load |
| **Brainstem protection** | `orchestrator`, `tester_agent`, `synthesis_agent` can never be hot-loaded |
| **Agent pause** | `_running=False` + 100ms sleep before importlib patch; resumed in `finally` |
| **Session synthesis limit** | Max 5 re-syntheses per agent per session; 6th attempt logs warning only |
| **Differential wheel drive** | L298N H-Bridge via `WheelDriver`; linear PWM ramp-up (50ms) prevents mechanical stress |
| **Hysteresis gaze-nav** | Trigger rotation at pan >45°, clear only when pan <15° — prevents jitter oscillation |
| **Reflex lifecycle** | Prune zero-invocation after 7 days; promote ≥100 invocations to `brain/reflexes/` as hardcode |
| **MirrorAgent** | Grounded observational learning — constrains SmolVLM2 prompts to known motor primitives |
| **Native I2C discovery** | SynthesisAgent._discover_device() matches 11 known I2C addresses; synthesizes smbus2 driver |

---

## V7.0 — Natively Embodied Autonomous Brain

| Feature | Description |
|---|---|
| **Spatial Intelligence** | `VisionProcessor.spatial_point()` → normalised (x,y) gaze target; `detect_3d()` → monocular depth estimate |
| **Gaze snap** | `MotorAgent._gaze_snap()` converts SPATIAL_POINT (x,y)→(pan,tilt) instantly, bypasses saccade smoothing |
| **Visual servoing** | `VisionAgent._visual_servo()` — 5s loop emitting SPATIAL_POINT until target within 5% of centre |
| **VLA token mapping** | `VLAMapper.parse()` maps Gemini VLA action dicts to WheelDriver/MotorDriver calls |
| **Long-horizon planning** | `PlannerAgent._decompose_with_er()` → Gemini ER structured plan; `_estimate_success()` after each step |
| **RE_PLAN loop** | success_estimate < 0.4 → RE_PLAN emitted → PlannerAgent re-decomposes (max 3 retries) |
| **Deliberative loop** | `Orchestrator._deliberative_loop()` at 2Hz during active planning; WorldModel snapshot → CognitionAgent |
| **Dynamic thinking budget** | InteroceptionAgent emits THINKING_BUDGET (1024→256 tokens) on thermal throttle |
| **WorldModel spatial** | `update_spatial()`, `get_nearest(label)` for 3D object proximity queries |
| **Cross-embodiment** | MirrorAgent maps VLA DOF tokens to motor primitives via `get_primitive_skills()` |

---

## Neuroplasticity (V5.1+)

| Mechanism | How it works |
|---|---|
| **Correction detection** | Detects "that's wrong" → stores high-importance episode (0.9) |
| **Nightly consolidation** | DreamAgent extracts style insight → stored in USER.json |
| **Scenario preferences** | "In the morning I like coffee" → USER.json `scenario_preferences` |
| **Predicted Needs** | CognitionAgent injects `## Predicted Needs` from time-of-day + habit patterns |
| **Semantic reflexes** | Repeated intents → embedding-based reflex (≥0.92 cosine) → bypasses LLM entirely |
| **Knowledge gaps** | MetacognitionAgent persists low-confidence topics across restarts in USER.json |

---

## Autonomous Behaviours

| Trigger | Action |
|---|---|
| 5 min idle, Curiosity Drive high | Pan-tilt scan + LLM-generated curious thought |
| Person present, 3 min silence | Tone-aware proactive check-in |
| New knowledge gap detected | CuriosityAgent generates targeted clarifying question |
| `WORLD_SURPRISE` fired | CuriosityAgent generates "What just changed?" |
| CPU > 85% or RAM > 85% | Auto-ECO + THINKING_BUDGET reduced to 256 tokens |
| CPU temp > 75°C | THINKING_BUDGET kill-switch; ECO mode |
| SYSTEM_ERROR ×3 for same agent | SynthesisAgent rewrites failing method |
| Reflex invocations ≥ 100 | DreamAgent promotes reflex to `brain/reflexes/` as permanent code |
| "look at X" speech command | VisionAgent._visual_servo() tracks X until centred |
| VLA action token received | VLAMapper dispatches to WheelDriver / MotorDriver |
| COGNITION_INTENT with `task` intent | PlannerAgent uses ER to decompose into motor steps |
| 2AM systemd timer | Dream cycle: reflex distillation, pruning, promotion, personality evolution |

---

## Message Types

### V6.0 (self-evolution)
- `motor.wheeled_rotate` — differential rotation to re-centre gaze
- `neuro.synthesis` — trigger SynthesisAgent
- `neuro.code_validated` — TesterAgent sandbox request
- `neuro.reflex_ready` — validated module ready for hot-load

### V7.0 (embodied)
- `spatial.point` — normalised (x,y) gaze target for MotorAgent snap
- `spatial.3d` — 3D bounding box list from VisionProcessor
- `plan.replan` — success probability < threshold → re-plan
- `vla.control_token` — Gemini VLA action token → motor dispatch
- `cognition.thinking_budget` — ER token budget from InteroceptionAgent
- `plan.success_estimate` — per-step success probability from ER

---

## Requirements

- Python 3.11+
- Ollama with `llama3.2:1b` + `llama3.2:3b` + `nomic-embed-text` models
- Optional: Google AI API key (free tier), Anthropic API key, OpenRouter API key
- `psutil` for InteroceptionAgent hardware sensing
- `sklearn` for cosine similarity in semantic reflex matching
- `smbus2` for I2C device driver synthesis (Raspberry Pi)
- `sentence-transformers` (optional offline embedding fallback)
- Intel OpenVINO Toolkit (WSL2/Linux, optional — SmolVLM2 GPU backend)
