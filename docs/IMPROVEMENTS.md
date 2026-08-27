# Project-ABC — Improvements & Compliance Status

---

## V7.0 PRD Compliance — 100%

| Area | Status |
|---|---|
| All 28 cognitive agents | ✅ Complete |
| Hardware drivers (camera, audio, display, motor, GPIO, face rec, SmolVLM2, wheel) | ✅ Complete |
| Memory tiers (working, episodic FTS5, semantic, soul, reflexes, DNA) | ✅ Complete |
| LLM router (Ollama → Google AI → OpenRouter → Claude, 7-tier + ER/VLA) | ✅ Complete |
| REST + WebSocket API | ✅ Complete |
| Boot sequence, watchdog, systemd services | ✅ Complete |
| F1 — Emotion-aware TTS speed | ✅ `SpeechAgent` EMOTION_SPEED → Kokoro `speed` |
| F2 — Voice tone / back-channel acks | ✅ `AuditoryAgent` RMS energy + "Mm-hmm" |
| F3 — User name in responses | ✅ `SoulManager.get_system_prompt()` injects name |
| F4 — Time-of-day awareness | ✅ `CognitionAgent` time_label + Predicted Needs |
| F5 — Barge-in acknowledgement | ✅ `SpeechAgent` "Go ahead" on TTS interrupt |
| F6 — Personality evolution | ✅ `DreamAgent._evolve_personality()` OCEAN ±0.01 nightly |
| F7 — Memory decay | ✅ `DreamAgent` Ebbinghaus ×0.85 weekly |
| F8 — Gaze tracking | ✅ `VisionAgent` publishes `look_at` offset → `MotorAgent` |
| F9 — Confidence expression | ✅ `VerifierAgent` semantic consistency hedge prefix |
| F10 — Volume matching | ✅ `SpeechAgent` voice_rms EMA → Kokoro buffer scale |
| F11 — Habit detection | ✅ `DreamAgent._detect_habits()` → USER.json |
| F12 — Proactive conversation | ✅ `CuriosityAgent` WORLD_SURPRISE + 3-min silence check-in |
| N1 — Correction awareness | ✅ `CognitionAgent` detects corrections → `DreamAgent` style insight |
| N2 — Scenario preferences + Predicted Needs | ✅ `CognitionAgent` + `SoulManager.upsert_scenario_preference()` |
| N3 — Internal monologue (two-stage LLM) | ✅ `ReasoningAgent` Pass 1 (think) → Pass 2 (respond) |
| N4 — Persistent knowledge gaps | ✅ `MetacognitionAgent` → USER.json via `SoulManager.update_knowledge_gaps()` |
| Brain Fog (ECO suppresses imagination/ideation) | ✅ `Orchestrator._eco_mode` + `_ECO_BLOCKED_AGENTS` |
| Motor saccade smoothing | ✅ Sigmoid smoothstep 8-step interpolation in `MotorAgent._smooth_look_at()` |
| Attention gating (GWT salience) | ✅ `AttentionAgent` top-3 spotlight |
| Affective bias | ✅ `AttentionAgent` ×1.3; `ReasoningAgent` brevity suffix |
| Auto-ECO via InteroceptionAgent | ✅ BEHAVIOR_CHANGE(eco=True/False) |
| Knowledge-gap curiosity trigger | ✅ `MetacognitionAgent` → `CURIOSITY_TRIGGER` |
| Emotional salience memory pre-gate | ✅ `EpisodicMemory` skips routine low-emotion episodes |
| **V6 — Neocortical Synthesis** | ✅ `SynthesisAgent` AST guardrail + Gemini repair + hot-load |
| **V6 — Tester Sandbox** | ✅ `TesterAgent` subprocess isolation, empty env |
| **V6 — Digital DNA** | ✅ SHA256 brainstem checksum + protected rollback |
| **V6 — Brainstem protection** | ✅ Protected agent frozenset; hot-load blocked |
| **V6 — Agent pause during hot-load** | ✅ `_running=False` → 100ms → patch → resume |
| **V6 — Session synthesis limit** | ✅ Max 5 per agent; blacklist on breach |
| **V6 — Differential wheel drive** | ✅ `WheelDriver` L298N + linear PWM ramp |
| **V6 — Hysteresis gaze-nav** | ✅ Trigger >45°, clear <15° |
| **V6 — Semantic muscle memory** | ✅ Embedding-based reflexes ≥0.92 bypass LLM |
| **V6 — Reflex distillation** | ✅ `DreamAgent._distill_reflexes()` nightly |
| **V6 — Reflex prune/promote** | ✅ Prune 0-invocation after 7d; promote ≥100 to `brain/reflexes/` |
| **V6 — MirrorAgent** | ✅ Grounded motor primitives from SmolVLM2 |
| **V7 — Spatial point / 3D detection** | ✅ `VisionProcessor.spatial_point()` + `detect_3d()` |
| **V7 — Gaze snap** | ✅ `MotorAgent._gaze_snap()` normalised→degrees |
| **V7 — Visual servoing** | ✅ `VisionAgent._visual_servo()` 5s tracking loop |
| **V7 — VLAMapper** | ✅ `brain/hardware/vla_mapper.py` ACTION_MAP + fallback_parse |
| **V7 — VLA token dispatch** | ✅ `MotorAgent._handle_vla_token()` |
| **V7 — ER long-horizon planning** | ✅ `PlannerAgent._decompose_with_er()` + `_estimate_success()` |
| **V7 — RE_PLAN loop** | ✅ success_estimate < 0.4 → RE_PLAN (max 3 retries) |
| **V7 — Deliberative loop** | ✅ `Orchestrator._deliberative_loop()` 2Hz during active plan |
| **V7 — Dynamic thinking budget** | ✅ `InteroceptionAgent` THINKING_BUDGET emission 1024→256 |
| **V7 — WorldModel spatial** | ✅ `update_spatial()` + `get_nearest()` |
| **V7 — I2C device discovery** | ✅ `SynthesisAgent._discover_device()` 11 known addresses + `_synthesize_driver()` |
| **V7 — gemini_robotics config** | ✅ `config/config.yaml` block |

---

## Human Brain vs Project-ABC — V7.0

| Brain Region | Human Function | Project-ABC v7.0 | Gap |
|---|---|---|---|
| Visual Cortex | Multi-layer vision: edges → objects → faces | VisionAgent + SmolVLM2-500M + 6-frame RingBuffer + spatial_point/detect_3d | No stereo depth |
| Auditory Cortex | Speech, tone, ambient sounds | AuditoryAgent + Vosk STT + VAD + RMS energy | ✅ Voice tone detected |
| Prefrontal Cortex | Planning, reasoning, decision-making | CognitionAgent + ReasoningAgent + PlannerAgent (ER long-horizon) | ✅ Long-horizon decomposition |
| Limbic / Amygdala | Emotions, fear response, memory tagging | EmotionEngine 10-state FSM + affective bias | ✅ Emotion drives salience |
| Hippocampus | Memory formation, episodic → semantic | EpisodicMemory FTS5 + DreamAgent nightly consolidation | ✅ Well-modelled |
| Cerebellum | Motor coordination, learned patterns | MotorAgent (saccade smoothing + VLA tokens + gaze snap) | ✅ Saccade smoothing complete |
| Default Mode Network | Mind-wandering, curiosity, creativity | CuriosityAgent + IdeationAgent + ImaginationAgent | ✅ Active via drives |
| Mirror Neurons | Empathy, imitation | MirrorAgent (grounded observational learning) + EmotionEngine | ✅ Motor primitive imitation |
| Broca's / Wernicke's | Language production/comprehension | LanguageAgent + LLM | ✅ Strong via LLM |
| Thalamus | Sensory routing, attention gating | Orchestrator + AttentionAgent GWT Spotlight | ✅ Spotlight top-3 |
| Reticular Formation | Arousal, sleep/wake, fatigue | BehaviorAgent FSM + InteroceptionAgent + Brain Fog | ✅ ECO on CPU/RAM/temp |
| Insula | Interoception, body awareness | InteroceptionAgent → THINKING_BUDGET + ECO + INTERO_STATE | ✅ Thermal kill-switch |
| Cerebral Cortex | Consciousness, identity | SoulManager OCEAN + DreamAgent nightly evolution | ✅ OCEAN evolves |
| Anterior PFC | Self-reflection, metacognition | MetacognitionAgent + VerifierAgent + `_last_confidence` | ✅ Full pipeline |
| Spinal Cord | Fast reflexes, hardwired responses | ReasoningAgent semantic reflex arc (≥0.92 cosine bypass) | ✅ Sub-LLM latency |
| Premotor Cortex | Motor planning, action prediction | PlannerAgent ER decomposition + SUCCESS_ESTIMATE | ✅ Long-horizon motor plan |
| Neuroplasticity | Synaptic strengthening, rewiring | SynthesisAgent hot-load + reflex promotion + DreamAgent | ✅ Self-modifying code |
| Parietal Cortex | Spatial awareness, 3D world model | WorldModel spatial + VisionProcessor.detect_3d() | ✅ Monocular depth estimate |

---

## The Honest Ceiling

Project-ABC v7.0 reaches **~100% of what is achievable with current hardware and software**.

| Limit | Why it cannot be closed |
|---|---|
| **True understanding** | LLMs simulate comprehension via statistical pattern — they do not have it |
| **Genuine emotion** | The 10-state FSM mimics affect — it does not feel |
| **Continuous consciousness** | Brain resets between sessions; humans maintain identity through sleep |
| **Stereo depth** | `detect_3d()` uses monocular heuristics — no true parallax without a stereo camera |
| **Infinite planning horizon** | ER re-plan loop capped at 3 retries per step — bounded rationality |

Everything else — memory, personality, adaptation, social cues, gaze, tone, timing, curiosity, self-correction, motor learning, spatial grounding, and self-rewriting code — is implemented and running.
