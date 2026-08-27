# Project-ABC v5.1 — Configuration Reference

All runtime configuration lives in two files:

| File | Purpose |
|---|---|
| `config/config.yaml` | All brain behaviour, LLM, audio, vision, memory, agents |
| `config/hardware.yaml` | Hardware enable/disable overrides (runs after auto-detection) |
| `config/.env` | Secret API keys (never commit this file) |

---

## config/config.yaml

### `brain` — Core settings

```yaml
brain:
  name: "Brain"
  wake_word: "hey brain"
  language: "en"
  idle_timeout_seconds: 60
  thermal_throttle_temp: 75
  data_dir: "data"
  assets_dir: "assets"
  log_level: "INFO"
  log_file: "data/brain.log"
  pulse_hz: 10
```

| Key | Default | Description |
|---|---|---|
| `name` | `"Brain"` | Robot's self-identifier used in prompts |
| `wake_word` | `"hey brain"` | Phrase used for vosk_text fallback matching |
| `language` | `"en"` | ISO language code for STT and TTS |
| `idle_timeout_seconds` | `60` | Seconds of silence before BehaviorAgent enters IDLE |
| `thermal_throttle_temp` | `75` | °C — InteroceptionAgent triggers ECO mode above this |
| `data_dir` | `"data"` | Root directory for all runtime data files |
| `assets_dir` | `"assets"` | Root directory for GIFs, models, sounds |
| `log_level` | `"INFO"` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `log_file` | `"data/brain.log"` | Log output file path |
| `pulse_hz` | `10` | GWT Spotlight heartbeat frequency (Hz) |

---

### `llm` — LLM Routing

```yaml
llm:
  default_model: "llama3.2:1b"
  balanced_model: "llama3.2:3b"
  cloud_model: "gemini-2.0-flash"
  embed_model: "nomic-embed-text"
  ollama_host: "http://localhost:11434"
  simple_token_threshold: 150
  complex_token_threshold: 800
  cache_similarity_threshold: 0.92
  max_retries: 3
  timeout_seconds: 30
  openrouter_model: "meta-llama/llama-3.2-3b-instruct:free"
```

| Key | Default | Description |
|---|---|---|
| `default_model` | `llama3.2:1b` | Tier 1 — fast local model for short queries |
| `balanced_model` | `llama3.2:3b` | Tier 2 — local model for standard queries |
| `cloud_model` | `gemini-2.0-flash` | Tier 3/4 — primary cloud model |
| `embed_model` | `nomic-embed-text` | Ollama embedding model for semantic memory |
| `ollama_host` | `http://localhost:11434` | Ollama server URL |
| `simple_token_threshold` | `150` | Queries below this token count use Tier 1 |
| `complex_token_threshold` | `800` | Queries above this use cloud (Tier 4) |
| `cache_similarity_threshold` | `0.92` | Cosine similarity ≥ this → return cached response |
| `max_retries` | `3` | Retry attempts before escalating to next tier |
| `timeout_seconds` | `30` | Per-request LLM timeout |
| `openrouter_model` | `llama-3.2-3b-instruct:free` | Free cloud fallback model via OpenRouter |

**Free OpenRouter alternatives** (set via `openrouter_model`):
```
meta-llama/llama-3.1-8b-instruct:free   # smarter, slower
google/gemma-3-27b-it:free              # very capable, rate-limited
mistralai/mistral-7b-instruct:free      # balanced
qwen/qwen-2.5-7b-instruct:free          # multilingual
```

**API keys** (set in `config/.env`, not here):
```
GOOGLE_AI_API_KEY      → enables Tier 3/4 (Gemini)
ANTHROPIC_API_KEY      → enables Tier 6 (Claude Haiku)
OPENROUTER_API_KEY     → enables Tier 5 (free cloud models)
```

---

### `memory` — Memory Architecture

```yaml
memory:
  working_ttl_seconds: 600
  sensory_ttl_seconds: 3
  episodic_db: "data/episodes.db"
  semantic_db: "data/chroma"
  episodic_retention_days: 7
  soul_file: "data/SOUL.md"
  user_file: "data/USER.md"
  user_json_file: "data/USER.json"
  world_file: "data/WORLD.md"
  skills_file: "data/SKILLS.json"
  dream_journal: "data/DREAM_LOG.md"
  restore_on_startup: true
  restore_exchanges: 6
```

| Key | Default | Description |
|---|---|---|
| `working_ttl_seconds` | `600` | Working memory TTL (10 min) — per-session context |
| `sensory_ttl_seconds` | `3` | RingBuffer frame TTL — frames older than this are dropped |
| `episodic_db` | `data/episodes.db` | SQLite FTS5 episodic memory path |
| `semantic_db` | `data/chroma` | ChromaDB vector store path |
| `episodic_retention_days` | `7` | Episodes older than this are purged by DreamAgent |
| `soul_file` | `data/SOUL.md` | Core personality markdown |
| `user_file` | `data/USER.md` | Legacy user profile markdown |
| `user_json_file` | `data/USER.json` | Structured user profile (OCEAN, habits, preferences) |
| `world_file` | `data/WORLD.md` | Environment facts for MetacognitionAgent validation |
| `skills_file` | `data/SKILLS.json` | Learned conditional skills |
| `dream_journal` | `data/DREAM_LOG.md` | DreamAgent nightly consolidation log |
| `restore_on_startup` | `true` | Reload last session context on boot |
| `restore_exchanges` | `6` | Number of recent turns to restore into working memory |

---

### `display` — Display Driver

```yaml
display:
  enabled: true
  type: "ili9341"
  width: 240
  height: 320
  spi_device: 0
  spi_port: 0
  dc_pin: 24
  rst_pin: 25
  backlight_pin: 18
  fps: 15
  default_emotion: "neutral"
  hud_enabled: true
```

| Key | Default | Options / Notes |
|---|---|---|
| `enabled` | `true` | Set `false` to disable all display output |
| `type` | `ili9341` | `ili9341` / `st7789` / `ssd1306` / `mock` |
| `width` / `height` | `240` / `320` | Pixel dimensions of the display |
| `spi_device` / `spi_port` | `0` / `0` | SPI bus (Raspberry Pi) |
| `dc_pin` / `rst_pin` / `backlight_pin` | `24` / `25` / `18` | GPIO pin numbers (BCM) |
| `fps` | `15` | GIF animation frame rate |
| `default_emotion` | `neutral` | Emotion shown at startup |
| `hud_enabled` | `true` | Overlay text HUD (emotion label, state) on display |

---

### `audio` — Microphone, TTS, Wake Word

```yaml
audio:
  input:
    sample_rate: 16000
    channels: 1
    chunk_size: 1024
    vad_threshold: 0.5
    silence_timeout_ms: 1500
  output:
    preferred: "bluetooth"
    volume: 80
    tts_engine: "auto"
    edge_voice: "en-US-JennyNeural"
    elevenlabs_voice_name: "Anika"
    elevenlabs_voice_id: ""
    espeak_voice: "en+f3"
    espeak_speed: 165
  wake_word:
    enabled: false
    engine: "openwakeword"
    model: "hey_jarvis"
    threshold: 0.5
    vad_enabled: true
    fallback: "vosk_text"
```

**Input:**

| Key | Default | Description |
|---|---|---|
| `sample_rate` | `16000` | Microphone sample rate (Hz) — Vosk requires 16000 |
| `channels` | `1` | Mono microphone input |
| `chunk_size` | `1024` | PyAudio buffer chunk size |
| `vad_threshold` | `0.5` | Voice activity detection threshold (0-1) |
| `silence_timeout_ms` | `1500` | ms of silence to mark end of utterance |

**Output:**

| Key | Default | Description |
|---|---|---|
| `preferred` | `bluetooth` | `bluetooth` / `usb` / `aux` — speaker connection type |
| `volume` | `80` | Master volume (0-100) |
| `tts_engine` | `auto` | `auto` / `kokoro` / `elevenlabs` / `edge` / `espeak` |
| `edge_voice` | `en-US-JennyNeural` | Microsoft Edge TTS voice name |
| `elevenlabs_voice_name` | `Anika` | ElevenLabs voice name (looked up by API) |
| `elevenlabs_voice_id` | `""` | ElevenLabs voice ID — set to skip name lookup |
| `espeak_voice` | `en+f3` | espeak-ng voice variant |
| `espeak_speed` | `165` | espeak-ng words per minute |

`tts_engine: auto` priority: ElevenLabs (if key) → Kokoro → Edge → espeak.

**Wake Word:**

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable wake-word gating (disable = all speech goes to Brain) |
| `engine` | `openwakeword` | `openwakeword` / `vosk_text` / `none` |
| `model` | `hey_jarvis` | Built-in openwakeword model or path to `.tflite` |
| `threshold` | `0.5` | Detection confidence (lower = more sensitive) |
| `vad_enabled` | `true` | Skip wake-word when VAD detects silence |
| `fallback` | `vosk_text` | Fallback engine if openwakeword fails |

Built-in openwakeword models: `hey_jarvis`, `alexa`, `hey_mycroft`, `timer`, `weather`.

---

### `vision` — Camera and Vision Models

```yaml
vision:
  enabled: true
  source: "auto"
  capture_interval_ms: 500
  detection_confidence: 0.5
  scene_description: true
  moondream:
    mode: "auto"
    model_size: "2b"
    caption_interval_s: 3.0
  face_recognition:
    mock: false
    dist_threshold: 0.55
    lbph_threshold: 70.0
  smolvlm2:
    enabled: true
    backend: "auto"
    llamacpp_url: "http://localhost:8090"
    scan_interval: 2.5
    ring_buffer_frames: 6
    surprise_threshold: 0.30
    model_id: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    openvino_device: "GPU"
    mock: false
```

**Top level:**

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Master camera enable |
| `source` | `auto` | `auto` / `usb` / `csi` — camera source |
| `capture_interval_ms` | `500` | Frame capture interval (ms) |
| `detection_confidence` | `0.5` | Minimum confidence for object/face detection |
| `scene_description` | `true` | Enable VLM scene captioning |

**SmolVLM2 (primary vision model):**

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable SmolVLM2 continuous video understanding |
| `backend` | `auto` | `auto` / `llamacpp` / `openvino` / `huggingface` / `mock` |
| `llamacpp_url` | `http://localhost:8090` | llama-server endpoint for GGUF backend |
| `scan_interval` | `2.5` | Seconds between VLM inference calls |
| `ring_buffer_frames` | `6` | Frames held in RingBuffer (3s window at 2fps) |
| `surprise_threshold` | `0.30` | Scene-change delta > this fires `WORLD_SURPRISE` |
| `model_id` | `SmolVLM2-500M-Video-Instruct` | HuggingFace model ID (huggingface backend) |
| `openvino_device` | `GPU` | `GPU` (Intel Iris Xe) or `CPU` |
| `mock` | `false` | Return placeholder captions without running model |

Backend priority (when `backend: auto`): llamacpp → openvino → huggingface → mock.

**Face recognition:**

| Key | Default | Description |
|---|---|---|
| `mock` | `false` | Disable face recognition (no dlib required) |
| `dist_threshold` | `0.55` | face_recognition library match threshold (lower = stricter) |
| `lbph_threshold` | `70.0` | OpenCV LBPH fallback threshold (lower = stricter) |

---

### `motor` — Pan-Tilt / Wheel Control

```yaml
motor:
  enabled: false
  type: "pantilt"
  pan_channel: 0
  tilt_channel: 1
  pan_range: [-90, 90]
  tilt_range: [-45, 45]
  speed_default: 50
  saccade_smoothing: true
  saccade_steps: 8
  saccade_duration_ms: 200
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Auto-enabled when motor hardware detected |
| `type` | `pantilt` | `pantilt` / `wheels` / `arm` / `none` |
| `pan_channel` | `0` | PCA9685 servo channel for pan |
| `tilt_channel` | `1` | PCA9685 servo channel for tilt |
| `pan_range` | `[-90, 90]` | Degrees of travel for pan axis |
| `tilt_range` | `[-45, 45]` | Degrees of travel for tilt axis |
| `speed_default` | `50` | Default movement speed (0-100) |
| `saccade_smoothing` | `true` | Sigmoid interpolation on `look_at` (biological saccade) |
| `saccade_steps` | `8` | Interpolation steps — more = smoother |
| `saccade_duration_ms` | `200` | Total `look_at` movement duration (ms) |

---

### `api` — REST / WebSocket Server

```yaml
api:
  enabled: true
  host: "0.0.0.0"
  port: 8080
  cors_origins: ["*"]
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Start FastAPI server on boot |
| `host` | `0.0.0.0` | Bind address (`127.0.0.1` to restrict to localhost) |
| `port` | `8080` | HTTP port |
| `cors_origins` | `["*"]` | Allowed CORS origins — restrict in production |

---

### `dream` — Nightly Memory Consolidation

```yaml
dream:
  enabled: true
  schedule: "02:00"
  min_episodes: 5
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable DreamAgent nightly cycle |
| `schedule` | `02:00` | Wall-clock time to run (systemd timer on Pi) |
| `min_episodes` | `5` | Skip cycle if fewer episodes recorded today |

Trigger manually via API: `POST /dream`

---

### `privacy` — Privacy Mode

```yaml
privacy:
  gpio_button_pin: 17
  enabled: true
```

| Key | Default | Description |
|---|---|---|
| `gpio_button_pin` | `17` | BCM GPIO pin number for hardware privacy button |
| `enabled` | `true` | Enable privacy mode feature |

When active: camera, microphone, and display are all paused.

---

### `cognition_extensions` — Higher-Order Cognitive Agents

#### `attention`

```yaml
cognition_extensions:
  attention:
    enabled: true
    focus_decay_seconds: 30
    salience_threshold: 0.4
    shift_threshold: 0.8
    spotlight_size: 3
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable AttentionAgent (GWT gating) |
| `focus_decay_seconds` | `30` | Seconds before top-down focus expires |
| `salience_threshold` | `0.4` | Messages below this are discarded (not queued) |
| `shift_threshold` | `0.8` | Salience above this fires bottom-up `ATTENTION_SHIFT` |
| `spotlight_size` | `3` | Top-N messages broadcast to cognitive agents each Pulse |

#### `interoception`

```yaml
  interoception:
    enabled: true
    poll_interval_seconds: 10
    cpu_fatigue_threshold: 85
    cpu_busy_threshold: 60
    temp_discomfort_threshold: 70
    ram_overwhelm_threshold: 85
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable hardware vitals polling |
| `poll_interval_seconds` | `10` | How often to read CPU/RAM/temp |
| `cpu_fatigue_threshold` | `85` | CPU% → `"overwhelmed"` label + ECO trigger |
| `cpu_busy_threshold` | `60` | CPU% → `"busy"` label |
| `temp_discomfort_threshold` | `70` | °C → `"hot"` label |
| `ram_overwhelm_threshold` | `85` | RAM% → `"memory_pressure"` label + ECO trigger |

ECO also triggers when `temp_c ≥ brain.thermal_throttle_temp` (75°C default), independent of labels.

#### `metacognition`

```yaml
  metacognition:
    enabled: true
    confidence_threshold: 0.3
    low_confidence_prefix: true
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable MetacognitionAgent confidence scoring |
| `confidence_threshold` | `0.3` | Below this → flag uncertain response to VerifierAgent |
| `low_confidence_prefix` | `true` | Prepend uncertainty prefix when confidence is low |

#### `theory_of_mind`

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable user confusion/expertise modelling |
| `update_interval_interactions` | `5` | Update user model every N interactions |

#### `temporal_reasoning`

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable hour-of-day pattern + causal chain analysis |
| `pattern_update_interval_seconds` | `60` | How often to recompute temporal patterns |
| `causal_detection` | `true` | Detect causal phrases ("because", "after") |

#### `intrinsic_motivation`

```yaml
  intrinsic_motivation:
    enabled: true
    recompute_interval_seconds: 300
    drives:
      learn: 0.40
      help: 0.30
      explore: 0.20
      rest: 0.10
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable drive-based motivation system |
| `recompute_interval_seconds` | `300` | Drive recomputation interval |
| `drives.learn` | `0.40` | Weight for learning/knowledge drive |
| `drives.help` | `0.30` | Weight for helping/social drive |
| `drives.explore` | `0.20` | Weight for exploration drive |
| `drives.rest` | `0.10` | Weight for rest/conservation drive |

Drive weights must sum to 1.0.

#### `ideation`

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable cross-domain idea synthesis |
| `cooldown_seconds` | `60` | Minimum seconds between ideation runs |
| `semantic_neighbors_k` | `5` | Number of semantic neighbours to retrieve |
| `curiosity_trigger_probability` | `0.30` | Probability of triggering curiosity from an idea |

#### `imagination`

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable world-state prediction (30s lookahead) |
| `cooldown_seconds` | `120` | Minimum seconds between imagination runs |
| `pre_action_check` | `false` | Predict user reaction before speaking (adds ~1s latency) |

---

### `cognition_extensions` — v5.1 Feature Flags

#### `brain_fog`

```yaml
  brain_fog:
    enabled: true
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Suppress `imagination_agent` + `ideation_agent` routing during ECO mode |

When disabled, all agents receive messages even under hardware stress.

#### `internal_monologue`

```yaml
  internal_monologue:
    enabled: true
    think_tokens: 80
    min_response_tokens: 60
    thought_consistency: true
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Two-stage LLM: private think pass before responding |
| `think_tokens` | `80` | Token budget for the thinking pass |
| `min_response_tokens` | `60` | Skip monologue when affective response budget ≤ this |
| `thought_consistency` | `true` | VerifierAgent enforces brevity/empathy from stated thought |

#### `affective_bias`

```yaml
  affective_bias:
    enabled: true
    salience_multiplier: 1.3
    brevity_on_frustrated: true
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Emotion-aware salience scoring and response style |
| `salience_multiplier` | `1.3` | PERCEPTION salience multiplied by this when SURPRISED/FRUSTRATED |
| `brevity_on_frustrated` | `true` | Append brevity instruction to system prompt when FRUSTRATED |

#### `neuroplasticity`

```yaml
  neuroplasticity:
    correction_detection: true
    scenario_preferences: true
    predicted_needs: true
    persist_knowledge_gaps: true
```

| Key | Default | Description |
|---|---|---|
| `correction_detection` | `true` | Detect "that's wrong" → store high-importance correction episode |
| `scenario_preferences` | `true` | Learn time-contextual preferences ("morning → coffee") |
| `predicted_needs` | `true` | Inject `## Predicted Needs` section into LLM system prompt |
| `persist_knowledge_gaps` | `true` | Save MetacognitionAgent gap list to USER.json across restarts |

#### `emotional_pregate`

```yaml
  emotional_pregate:
    enabled: true
    importance_threshold: 0.2
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Skip routine perception episodes with no useful emotion |
| `importance_threshold` | `0.2` | Episodes below this importance AND no useful emotion are discarded |

Affected event types: `perception_vision`, `audio_event`, `world_update`.
Useful emotions: EXCITED, FRUSTRATED, SURPRISED, HAPPY, CURIOUS, CONFUSED, SAD, ANGRY, FEARFUL.

---

## config/hardware.yaml

Hardware auto-detection runs first; this file overrides results.

```yaml
camera:
  enabled: auto        # auto | true | false
  prefer_csi: true
  usb_device_index: 0

microphone:
  enabled: auto
  device_name: "auto"

display:
  enabled: auto
  mock_mode: false

bluetooth:
  enabled: auto
  speaker_mac: ""

motors:
  enabled: auto
  mock_mode: false

gpio:
  enabled: auto
  mock_mode: false
```

| Section | `enabled` | `mock_mode` | Notes |
|---|---|---|---|
| `camera` | `auto` | — | `prefer_csi: true` prefers Pi CSI over USB |
| `microphone` | `auto` | — | `device_name: "auto"` uses first detected input |
| `display` | `auto` | `false` | `mock_mode: true` logs display ops, no hardware |
| `bluetooth` | `auto` | — | Set `speaker_mac` to auto-connect on boot |
| `motors` | `auto` | `false` | `mock_mode: true` logs servo commands |
| `gpio` | `auto` | `false` | `mock_mode: true` safe for non-Pi environments |

`enabled: auto` — detected via OS device enumeration.
`enabled: true` — force-enable (use for testing specific hardware).
`enabled: false` — force-disable even if hardware is detected.

---

## config/.env — API Keys

```env
# Google AI (free — primary cloud tier)
GOOGLE_AI_API_KEY=

# Anthropic Claude (optional paid fallback)
ANTHROPIC_API_KEY=

# OpenRouter (optional free cloud models)
OPENROUTER_API_KEY=

# ElevenLabs TTS (optional — falls back to Kokoro/Edge)
ELEVENLABS_API_KEY=

# Dev flags
MOCK_HARDWARE=false    # true = force all hardware to mock
MOCK_LLM=false         # true = skip LLM calls, return canned responses
```

Copy from template: `cp config/.env.example config/.env`

---

## agents — Per-Agent Enable / Disable

Individual agents can be disabled via `config/config.yaml` under an `agents:` section (auto-created if missing). This is separate from `cognition_extensions`:

```yaml
agents:
  imagination_agent:
    enabled: false     # disable without affecting other agents
  ideation_agent:
    enabled: true
  dream_agent:
    enabled: true
  theory_of_mind_agent:
    enabled: false     # disable on low-memory hardware
```

Any agent not listed defaults to `enabled: true`.

---

## Common Tuning Scenarios

### Low-memory device (4 GB RAM)

```yaml
cognition_extensions:
  imagination:
    enabled: false
  ideation:
    enabled: false
  temporal_reasoning:
    enabled: false
  internal_monologue:
    enabled: false

agents:
  theory_of_mind_agent:
    enabled: false
```

### Fully offline (no internet)

```yaml
llm:
  cloud_model: ""          # disables cloud tiers
  # Keep ollama_host and local models
```

Leave `GOOGLE_AI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` empty in `.env`.

### Faster responses (reduce latency)

```yaml
cognition_extensions:
  internal_monologue:
    enabled: false         # saves one LLM call per response
  imagination:
    cooldown_seconds: 300  # run less often
  ideation:
    cooldown_seconds: 300
```

### Maximum memory efficiency

```yaml
memory:
  working_ttl_seconds: 300       # 5 min working memory
  episodic_retention_days: 3     # keep only 3 days

cognition_extensions:
  emotional_pregate:
    enabled: true
    importance_threshold: 0.4   # stricter filtering
```

### Development / no wake word

```yaml
audio:
  wake_word:
    enabled: false              # every utterance processed immediately
```

---

## Detailed Behavior & Examples

This section explains what actually happens at runtime when each setting is changed, with concrete examples.

---

### `brain.thermal_throttle_temp`

**What it does:** When CPU temperature reaches this value, `InteroceptionAgent` immediately publishes a `BEHAVIOR_CHANGE` with `eco: true`, regardless of CPU % or RAM %. The LLM router switches to the fastest local model (`default_model`) and token budgets shrink.

**Example:**
```yaml
brain:
  thermal_throttle_temp: 65   # aggressive — throttle earlier on Pi 3B+
```
Pi 3B+ running face recognition + LLM simultaneously often hits 72-78 °C. Setting this to 65 keeps the chip cool at the cost of slower, shorter responses.

---

### `brain.pulse_hz`

**What it does:** The GWT Spotlight heartbeat rate. Every tick, `AttentionAgent` scores all buffered messages and broadcasts the top `spotlight_size` to cognitive agents. Higher = more responsive, more CPU.

**Example:**
```yaml
brain:
  pulse_hz: 5   # halves CPU load from GWT loop; acceptable on Pi 3B+
```
At 10 Hz the loop fires every 100 ms. At 5 Hz it fires every 200 ms — perceptibly slower reactions to sudden events but meaningfully lighter on the CPU.

---

### `llm` tiers (default_model → cloud_model)

**What it does:** The router selects a model based on query token count and ECO mode:

| Condition | Model used |
|---|---|
| ECO mode active | `default_model` (always Tier 1) |
| tokens < `simple_token_threshold` | `default_model` |
| tokens < `complex_token_threshold` | `balanced_model` |
| tokens ≥ `complex_token_threshold` | `cloud_model` (Gemini) |
| Ollama unreachable | `cloud_model` then `openrouter_model` |

**Example — force all queries local:**
```yaml
llm:
  simple_token_threshold: 99999
  complex_token_threshold: 99999
  cloud_model: ""
```
Every query goes to `default_model`. No internet required. Slower and less accurate for complex questions.

**Example — prefer cloud:**
```yaml
llm:
  simple_token_threshold: 0
  complex_token_threshold: 0
```
Every query immediately jumps to Tier 3 (Gemini). Fastest and most accurate but uses API quota for all requests.

---

### `llm.cache_similarity_threshold`

**What it does:** Before calling any model, the router embeds the query and checks ChromaDB for a semantically similar previous response. If cosine similarity ≥ threshold, the cached answer is returned instantly — zero LLM cost.

**Example:**
```yaml
llm:
  cache_similarity_threshold: 0.85   # more cache hits, occasional stale replies
  cache_similarity_threshold: 0.97   # near-exact only, always fresh responses
```
At 0.92 (default), "What time is it?" and "Can you tell me the time?" return the same cached answer. At 0.97, they would be treated as different queries.

---

### `memory.restore_on_startup` / `restore_exchanges`

**What it does:** On boot, `CognitionAgent` reloads the last N conversation turns into working memory so the brain remembers what it was discussing before it was restarted.

**Example:**
```yaml
memory:
  restore_on_startup: true
  restore_exchanges: 3    # only restore 3 turns — faster boot
```
Without restore: "What were we talking about?" → "I don't have that context." With restore: brain picks up the thread immediately.

**Set to false for a clean slate on each boot:**
```yaml
memory:
  restore_on_startup: false
```

---

### `memory.episodic_retention_days`

**What it does:** DreamAgent prunes episodes older than this many days every night. Shorter retention keeps the SQLite database small and semantic search fast.

**Example:**
```yaml
memory:
  episodic_retention_days: 3    # Pi 3B+ with limited SD card — aggressive pruning
  episodic_retention_days: 30   # desktop/server — long-term relationship memory
```

---

### `audio.wake_word.enabled`

**What it does:**
- `enabled: false` — every microphone utterance is immediately transcribed and sent to the brain. Best for development and always-on assistants.
- `enabled: true` — the brain listens silently until the wake word is detected, then opens a 10-second attention window. All other speech is discarded.

**Example — enable with custom phrase:**
```yaml
audio:
  wake_word:
    enabled: true
    engine: "vosk_text"
    model: "hey_jarvis"   # or path to custom .tflite
    threshold: 0.4        # lower = more sensitive (more false triggers)
```
Lower `threshold` catches quieter or mumbled wake words but may trigger on similar-sounding words. Raise to 0.7+ in a noisy room.

---

### `audio.output.tts_engine`

**What it does:** Controls which TTS engine produces speech. `auto` tries each in priority order until one succeeds.

| Value | Quality | Latency | Requires |
|---|---|---|---|
| `elevenlabs` | Best — natural voice | 300-600 ms (network) | `ELEVENLABS_API_KEY` |
| `kokoro` | High — local ONNX | 150-400 ms | `assets/models/kokoro/` |
| `edge` | Good — Microsoft cloud | 200-500 ms (network) | Internet |
| `espeak` | Robotic — fully offline | < 50 ms | `espeak-ng` package |

**Example — force local-only:**
```yaml
audio:
  output:
    tts_engine: "kokoro"
```
No network calls for TTS. Falls back to espeak automatically if Kokoro model is missing.

---

### `vision.smolvlm2.scan_interval`

**What it does:** How often SmolVLM2 analyzes the camera buffer. Lower = more frequent scene updates, higher CPU usage.

**Example:**
```yaml
vision:
  smolvlm2:
    scan_interval: 1.0   # high-activity environment — fast scene changes
    scan_interval: 5.0   # static room — save CPU, still detects people entering
```
On Pi 3B+ with `scan_interval: 2.5`, VLM inference takes ~1.2-1.8s and runs every 2.5s — leaving ~0.7-1.3s of idle time between calls.

---

### `vision.smolvlm2.surprise_threshold`

**What it does:** Between consecutive VLM captions, the system computes a text-change delta. If delta > threshold, a `WORLD_SURPRISE` event fires — which routes to `CuriosityAgent` and `AttentionAgent`, causing the brain to say something like "Who just came in?" or turn toward the new stimulus.

**Example:**
```yaml
vision:
  smolvlm2:
    surprise_threshold: 0.15   # sensitive — fires on small changes (person shifting)
    surprise_threshold: 0.50   # fires only on large changes (person entering/leaving)
```

---

### `motor.saccade_smoothing`

**What it does:** When `look_at` command is received, instead of snapping the servo directly to the target angle, the motion is split into `saccade_steps` steps using a sigmoid (smoothstep) curve. This mimics biological eye saccades — fast at first, decelerating into the target.

**Example:**
```yaml
motor:
  saccade_smoothing: true
  saccade_steps: 12          # very smooth — good for slow, deliberate tracking
  saccade_duration_ms: 300
```

```yaml
motor:
  saccade_smoothing: false   # instant snap — lowest latency, mechanical feel
```

With smoothing on, a 90° pan takes 200 ms with 8 incremental servo commands. Without smoothing, it's a single command that arrives at the target in the servo's natural speed (~80 ms) but with a visible jerk.

---

### `cognition_extensions.brain_fog.enabled`

**What it does:** When the system enters ECO mode (CPU > 85%, RAM > 85%, or temperature > 75°C), this flag tells the Orchestrator to stop routing messages to `imagination_agent` and `ideation_agent`. These two agents run LLM calls to predict future world states and synthesize creative ideas — they are not needed for basic conversation.

**When enabled (default):**
- Brain under stress → imagination + ideation silenced → response latency drops by 30-50%
- Basic conversation, vision, speech, emotion all continue normally

**When disabled:**
- All agents receive messages regardless of hardware load
- Risk: on Pi 3B+ under thermal stress, imagination LLM calls compete with response generation, causing > 5s response times

**Example — disable for desktop/server where heat is not a concern:**
```yaml
cognition_extensions:
  brain_fog:
    enabled: false
```

---

### `cognition_extensions.internal_monologue`

**What it does:** Every user utterance triggers two sequential LLM calls instead of one:

1. **Pass 1 — Think (private):** `"You are thinking privately. What is the user really asking? What tone and length should the response be?"` — capped at `think_tokens` (80).
2. **Pass 2 — Respond:** The thought is prepended as `## Internal Thought` in the system prompt. The response is grounded in the stated reasoning.

`thought_consistency: true` activates VerifierAgent enforcement: if Pass 1 says "keep it brief" but Pass 2 produces > 200 chars, the response is trimmed. If Pass 1 says "empathetic tone" but no softeners are present, "I understand..." is prepended.

**When disabled:** Single LLM pass. Saves 0.5-1.5 s per response.

**Example — reduce think budget on slow hardware:**
```yaml
cognition_extensions:
  internal_monologue:
    enabled: true
    think_tokens: 40          # halves Pass 1 latency
    thought_consistency: false  # skip VerifierAgent post-check
```

---

### `cognition_extensions.affective_bias`

**What it does:** Connects the emotion engine to attention scoring and response style.

- **`salience_multiplier: 1.3`** — When current emotion is SURPRISED or FRUSTRATED, all incoming PERCEPTION messages (camera frames, audio events) have their salience score multiplied by 1.3. This makes the brain more alert and reactive during emotional arousal, mimicking the amygdala hijack effect.
- **`brevity_on_frustrated: true`** — When FRUSTRATED, appends `"Keep response brief and direct. Avoid lengthy explanations."` to the LLM system prompt.

**Example flow:**
1. User says "That's wrong, I've told you three times!" → emotion → FRUSTRATED
2. AttentionAgent multiplies salience of next camera motion by 1.3 (hypervigilance)
3. ReasoningAgent receives brevity directive → produces 1-2 sentence response

**To disable brevity but keep alertness:**
```yaml
cognition_extensions:
  affective_bias:
    enabled: true
    salience_multiplier: 1.3
    brevity_on_frustrated: false
```

---

### `cognition_extensions.neuroplasticity`

**`correction_detection: true`**

Detects correction phrases in user speech: "no that's wrong", "you misunderstood", "not what I meant", "that's incorrect". When matched:
- Stores the exchange as an `Episode` with `importance=0.9` and tag `["correction"]`
- DreamAgent's nightly `_consolidate_corrections()` queries these episodes and asks the LLM: "What communication mistake does this represent?" → writes insights to `communication_style_history` in `USER.json`
- Next session, `CognitionAgent` reads the updated style and adjusts

**`scenario_preferences: true`**

Learns time-contextual preferences from conversation. Example: "In the morning I usually have coffee" → stored as `{"morning": "coffee"}` in `USER.json`. At 8 AM next day, CognitionAgent injects `"Predicted Needs: user may want coffee"` into the prompt.

**`predicted_needs: true`**

Requires `scenario_preferences` to have data. Injects a `## Predicted Needs` section into every system prompt based on current `time_label` (morning / afternoon / evening / night) and learned habits.

**`persist_knowledge_gaps: true`**

When MetacognitionAgent detects an uncertain response (confidence < `confidence_threshold`), it saves the topic to `USER.json → knowledge_gaps`. On the next restart, the list is restored so CuriosityAgent can generate targeted follow-up questions about unresolved topics across sessions.

**Example — disable for a stateless kiosk:**
```yaml
cognition_extensions:
  neuroplasticity:
    correction_detection: false
    scenario_preferences: false
    predicted_needs: false
    persist_knowledge_gaps: false
```

---

### `cognition_extensions.emotional_pregate`

**What it does:** Guards the `log_event()` call in `EpisodicMemory`. Before writing an episode to SQLite, checks two conditions:
1. Is the emotion a "useful" signal? (EXCITED, FRUSTRATED, SURPRISED, HAPPY, CURIOUS, CONFUSED, SAD, ANGRY, FEARFUL)
2. Is `importance > importance_threshold`?

If **both** fail (neutral/no emotion AND low importance AND routine event type), the episode is discarded without being written.

Affected event types: `perception_vision`, `audio_event`, `world_update`.
Not affected: `conversation`, `correction`, `dream`, `behavior_change` — these are always stored.

**Impact on SQLite growth:**
```
Without pregate:  ~600-900 routine perception rows/day (Pi running 16h)
With pregate:     ~30-80 rows/day — only emotionally or contextually relevant moments
```

**Example — stricter filtering for low-storage SD card:**
```yaml
cognition_extensions:
  emotional_pregate:
    enabled: true
    importance_threshold: 0.4   # discard anything below 0.4 importance with no emotion
```

---

### `cognition_extensions.attention`

**`salience_threshold: 0.4`** — Messages scored below this by `AttentionAgent` are silently dropped and never enter the GWT workspace. Set lower (0.2) to let more background events through; higher (0.6) to only process salient events.

**`shift_threshold: 0.8`** — When a message scores above this, `AttentionAgent` publishes a bottom-up `ATTENTION_SHIFT` that interrupts the current focus. This is what causes the brain to suddenly react to a loud sound or unexpected movement.

**`spotlight_size: 3`** — Each Pulse cycle, the top 3 scored messages are broadcast to cognitive agents. Increase to 5 for richer multi-stream awareness; decrease to 1 for maximum focus/minimum distraction.

**Example — highly focused, ignore background:**
```yaml
cognition_extensions:
  attention:
    salience_threshold: 0.6    # only strong signals
    shift_threshold: 0.9       # very hard to interrupt
    spotlight_size: 2          # only top 2 messages per cycle
    focus_decay_seconds: 60    # hold focus for 1 full minute
```

---

### `config/hardware.yaml` — enabled vs mock_mode

**`enabled: auto`** — The system probes OS device files to detect hardware. If a Pi CSI camera is present, `camera.enabled` auto-resolves to `true`. If not on a Pi, it auto-resolves to `false`.

**`enabled: true`** — Force-enable. Use when auto-detection fails (e.g., USB camera on an unusual port).

**`enabled: false`** — Force-disable. Useful for running only specific subsystems during testing.

**`mock_mode: true`** — Hardware calls are replaced with log statements. The agent continues to function (publishes messages, responds to events) but nothing moves and nothing is recorded.

**Example — test motor logic without physical servos:**
```yaml
motors:
  enabled: true
  mock_mode: true
```
MotorAgent logs every `look_at` and `pan` command with timestamp and angle, but no pigpio calls are made. Safe to run on WSL2.

**Example — force camera off even when plugged in:**
```yaml
camera:
  enabled: false
```
VisionAgent enters passive mode. SmolVLM2 and face recognition are not called. All vision-related messages stop.

---

### `config/.env` dev flags

**`MOCK_HARDWARE=true`** — Equivalent to setting `mock_mode: true` for all hardware sections simultaneously. Faster than editing `hardware.yaml` for quick dev sessions.

**`MOCK_LLM=true`** — All LLM calls return a canned response instantly. Useful for testing message routing, agent pipelines, and memory without burning API quota or waiting for inference.

**Example `.env` for pure pipeline testing:**
```env
MOCK_HARDWARE=true
MOCK_LLM=true
# Leave all API keys empty
```
The brain boots, all 28 agents start, the full GWT loop runs, but no hardware is accessed and no LLM is called. Total boot time: ~2 s.

---

## V6.0 Self-Evolution Configuration

### `wheel_drive:`

```yaml
wheel_drive:
  enabled: false                   # set true when L298N is wired
  left_pins: [17, 27]             # BCM GPIO: IN1, IN2
  right_pins: [22, 23]            # BCM GPIO: IN3, IN4
  speed_default: 50               # PWM duty cycle 0–100
  rotate_speed: 35                # slower speed for precision rotation
  ramp_ms: 50                     # linear ramp-up duration (ms) — prevents mechanical stress
  gaze_rotate_threshold: 45       # pan degrees that triggers body rotation
  gaze_center_threshold: 15       # pan degrees at which rotation is "done" (hysteresis)
```

`gaze_rotate_threshold` and `gaze_center_threshold` implement hysteresis: the robot only starts rotating when pan exceeds 45°, and only stops when it returns below 15°. This prevents oscillation when the gaze hovers near the threshold.

`ramp_ms` causes `WheelDriver._ramp()` to linearly increase PWM duty cycle from 0 → target_speed over the specified milliseconds. Set to 0 to disable ramping.

---

### `synthesis_agent:`

```yaml
synthesis_agent:
  enabled: true
  crash_threshold: 3              # SYSTEM_ERROR count before synthesis fires
  confidence_threshold: 0.2       # METACOG_CONFIDENCE below this also fires synthesis
  session_synthesis_limit: 5      # max re-syntheses per agent per session
```

`session_synthesis_limit` is a memory-leak guard: after 5 synthesis attempts for the same agent in one boot session, the agent is blacklisted from further attempts and a warning is logged. Restart the brain to reset counters.

---

### `tester_agent:`

```yaml
tester_agent:
  enabled: true
  timeout_seconds: 10
```

TesterAgent runs AI-generated code in a subprocess with an empty environment (`env={}`), preventing secret leakage. It never uses `exec()` or `eval()`. If the subprocess times out, a `SYSTEM_ERROR` is emitted back to `synthesis_agent`.

---

### `reflex_registry:`

```yaml
reflex_registry:
  enabled: true
  pattern_threshold: 5            # min pattern repetitions to trigger distillation
  max_reflexes: 50                # prune oldest when exceeded
  similarity_threshold: 0.92      # cosine similarity threshold for reflex hit
  prune_after_days: 7             # remove zero-invocation reflexes after N days
  promote_after_invocations: 100  # move to brain/reflexes/ as permanent hardcode
```

`similarity_threshold`: how closely a user query must match a stored reflex embedding (via `nomic-embed-text`) to skip the LLM entirely. 0.92 is strict — only near-identical phrasings hit. Lower to 0.85 for broader matching.

`promote_after_invocations`: when a reflex has been used 100 times it is written as a `.py` file in `brain/reflexes/` and treated as a permanent hardcoded response — no embedding lookup needed.

---

### `mirror_agent:`

```yaml
mirror_agent:
  enabled: false                  # enable when SmolVLM2 active and camera available
```

When enabled, MirrorAgent subscribes to `VLM_SCAN_NOW` frames and asks SmolVLM2 to decompose observed human actions into motor primitives. Only actions matching `get_primitive_skills()` are stored. Cooldown: 5 seconds between mirror attempts.

---

## V7.0 Embodied Brain Configuration

### `gemini_robotics:`

```yaml
gemini_robotics:
  er_model: "gemini-2.0-flash"          # High-Level Brain (ER reasoning + planning)
  vla_model: "gemini-2.0-flash"         # Low-Level Brain (VLA motor tokens)
  thinking_budget_default: 1024         # default ER max_tokens for er_reason()
  thinking_budget_eco: 256              # reduced budget when CPU temp > thermal_throttle_temp
  success_threshold: 0.4                # RE_PLAN when success_estimate < this
  vla_confidence_threshold: 0.3         # minimum confidence to execute a VLA token
  servo_timeout_s: 5.0                  # visual_servo() loop timeout in seconds
  servo_center_tolerance: 0.05          # |x - 0.5| < this → target considered centred
```

**`thinking_budget_default` / `thinking_budget_eco`**: `InteroceptionAgent` emits `THINKING_BUDGET` messages when CPU temperature crosses `thermal_throttle_temp`. The Orchestrator updates `_thinking_budget` inline and passes the reduced value to `er_reason()` calls. This prevents the brain from running expensive ER reasoning while the hardware is thermally stressed.

**`success_threshold`**: After each motor step, `PlannerAgent._estimate_success()` calls `er_reason()` to evaluate residual success probability. If the probability drops below this threshold, a `RE_PLAN` message is emitted. The planner will attempt re-decomposition up to 3 times per task.

**`vla_confidence_threshold`**: VLA action tokens below this confidence are silently dropped. The VLAMapper `fallback_parse()` always returns confidence 0.3 from its regex path — set this lower if you want regex fallbacks to execute.

**`servo_timeout_s`**: `VisionAgent._visual_servo()` loops at ~5Hz emitting `SPATIAL_POINT` until the target object is within `servo_center_tolerance` of the frame centre (x=0.5). If the timeout expires first, the servo loop exits gracefully without error.

---

### Visual Servoing Trigger Phrases

Any PERCEPTION_SPEECH containing these patterns triggers `_visual_servo()`:

```
look at <object>
find <object>
track <object>
locate <object>
focus on <object>
point at <object>
```

The object name is extracted from the phrase and passed to `VisionProcessor.spatial_point()`. The motor system snaps gaze toward the detection and continues adjusting until centred or timed out.

---

### VLA Action Tokens

VLA tokens are dicts produced by `LLMRouter.vla_act()` and mapped by `VLAMapper.parse()`:

```python
# Token format
{"action_type": "move_pan", "params": {"degrees": 30.0}, "confidence": 0.85}

# Canonical action_type values
move_pan        → MotorDriver.pan(degrees)
move_tilt       → MotorDriver.tilt(degrees)
rotate_wheels   → WheelDriver.rotate_right/left(degrees, speed)
stop            → WheelDriver.stop()
speak           → ACTION_SPEAK published
```

Unknown action types are silently ignored. Low-confidence tokens (below `vla_confidence_threshold`) are also dropped.

---

### I2C Device Discovery

`SynthesisAgent` can synthesize Python drivers for connected I2C devices. Trigger via REST or internal message:

```python
# Emit via bus
AgentMessage(
    type=MessageType.NEURO_SYNTHESIS,
    data={"mode": "device_discovery", "i2c_addr": 0x68},  # MPU6050
)
```

Known device registry (11 addresses):

| Address | Device | Type |
|---|---|---|
| 0x48 | ADS1115 | 16-bit ADC |
| 0x40 | INA219 | Power sensor |
| 0x77 | BMP280 | Env sensor |
| 0x76 | BME280 | Env sensor |
| 0x68 | MPU6050 | 6-axis IMU |
| 0x3C | SSD1306 | OLED display |
| 0x27 | PCF8574 | GPIO expander |
| 0x29 | VL53L0X | ToF distance |
| 0x10 | APDS9960 | Gesture sensor |

Unknown addresses are logged but no driver is generated.
