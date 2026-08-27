# Project-ABC — Developer Setup (Laptop / Windows)

> **Target audience:** Developers running the robotic brain on a Windows or macOS laptop without Pi hardware.  
> For Raspberry Pi production setup, see the main [README.md](README.md).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Clone & Environment](#clone--environment)
4. [Install Dependencies](#install-dependencies)
5. [Configure Environment Variables](#configure-environment-variables)
6. [Download Emotion GIFs](#download-emotion-gifs)
7. [Download STT Model (Vosk)](#download-stt-model-vosk)
8. [Wake Word Setup (openwakeword)](#wake-word-setup-openwakeword)
9. [Run in Mock Mode](#run-in-mock-mode)
10. [Run the API](#run-the-api)
11. [Test Hardware Simulation](#test-hardware-simulation)
12. [Project Structure](#project-structure)
13. [Emotion System](#emotion-system)
14. [LLM Backends](#llm-backends)
15. [Troubleshooting](#troubleshooting)

---

## Overview

Project-ABC is a multi-agent cognitive brain for robotics. On a laptop you run it in **mock mode**: all hardware (camera, GPIO, SPI display, motors) is simulated; the microphone and speakers use your real laptop hardware. The API, LLM routing, emotion engine, and memory system all work exactly as they do on the Pi.

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11 or 3.12 | [python.org](https://www.python.org/downloads/) |
| Git | any | [git-scm.com](https://git-scm.com/) |
| Ollama | latest | [ollama.com](https://ollama.com/) — for local LLM |
| PortAudio | system | Required by PyAudio (see below) |

### Install PortAudio (Windows)

PyAudio needs PortAudio. On Windows the easiest path is to install the pre-built wheel:

```powershell
pip install pipwin
pipwin install pyaudio
```

Or download the wheel from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and:

```powershell
pip install PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl
```

---

## Clone & Environment

```powershell
# 1. Clone
git clone <repo-url> project-abc
cd project-abc

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\Activate.ps1      # PowerShell
# or
.venv\Scripts\activate.bat      # CMD
```

---

## Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on `tflite-runtime`**: The `openwakeword` package requires `tflite-runtime`. On Windows x86-64 this installs automatically. On Apple Silicon (M1/M2) you may need:
> ```bash
> pip install tensorflow-macos  # instead of tflite-runtime
> ```

Install the English NLP model for spaCy:

```powershell
python -m spacy download en_core_web_sm
```

---

## Configure Environment Variables

```powershell
copy config\.env.example config\.env
notepad config\.env           # or open in VS Code
```

Minimum required for laptop dev:

```dotenv
# Anthropic API (optional — only needed for complex queries)
ANTHROPIC_API_KEY=sk-ant-...

# Mock hardware — disables GPIO, SPI, I2C, pi-camera
MOCK_HARDWARE=true

# Skip real LLM calls (pure offline testing)
MOCK_LLM=false
```

> `GIPHY_API_KEY` is **no longer required** — emotion GIFs are now downloaded directly from GitHub.

---

## Download Emotion GIFs

The project uses 6 animated robot face GIFs from the [otto-emoji-gif-component](https://github.com/txp666/otto-emoji-gif-component/tree/main/gifs) project, mapped to 10 emotion states.

```powershell
python scripts/download_emotions.py
```

This will create:

```
assets/
  emotions/
    _cache/              ← raw GIFs from GitHub (6 files)
    neutral.gif          ← staticstate.gif
    attentive.gif        ← staticstate.gif
    focused.gif          ← staticstate.gif
    happy.gif            ← happy.gif
    excited.gif          ← happy.gif
    curious.gif          ← buxue.gif
    confused.gif         ← buxue.gif
    bored.gif            ← sad.gif
    frustrated.gif       ← anger.gif
    surprised.gif        ← scare.gif
```

No API key needed. Re-run is safe (skips existing files).

---

## Download STT Model (Vosk)

The AuditoryAgent uses Vosk for offline speech-to-text.

```powershell
# Create model directory
mkdir assets\models

# Download Vosk small English model (~50 MB)
# From: https://alphacephei.com/vosk/models
# Direct link:
curl -L https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -o vosk-model.zip
Expand-Archive vosk-model.zip -DestinationPath assets\models\
del vosk-model.zip
```

Expected path: `assets/models/vosk-model-small-en-us-0.15/`

> If the folder is missing, the AuditoryAgent prints a warning and runs without STT. The rest of the brain still works.

---

## Wake Word Setup (openwakeword)

The brain listens for **"Hey Brain"** before processing speech. On laptop, `openwakeword` is used — a neural wake-word detector that runs entirely on-device with no cloud dependency.

### How it works

```
Microphone → 80ms chunks → openwakeword (hey_jarvis model) → score ≥ 0.5?
                                                                    │
                                                    Yes → Wake event fired
                                                          → Vosk STT activated
                                                          → Brain processes utterance
```

> **Note:** We use the built-in `hey_jarvis` model as a reasonable stand-in for "Hey Brain" while a custom model is trained. The detection phrase is close enough for development.

### Configuration (`config/config.yaml`)

```yaml
audio:
  wake_word:
    enabled: true
    engine: "openwakeword"     # openwakeword | vosk_text | none
    model: "hey_jarvis"        # built-in model name
    threshold: 0.5             # lower = more sensitive, higher = fewer false positives
    fallback: "vosk_text"      # if openwakeword fails, text-match "hey brain" in transcript
```

### Available built-in models

| Model name | Trigger phrase |
|------------|---------------|
| `hey_jarvis` | "Hey Jarvis" (best stand-in) |
| `alexa` | "Alexa" |
| `hey_mycroft` | "Hey Mycroft" |
| `timer` | "Set a timer" |
| `weather` | "What's the weather" |

### Training a custom "Hey Brain" model

For production use, train a custom model using [openWakeWord's training guide](https://github.com/dscripka/openWakeWord?tab=readme-ov-file#training-new-models). Record at least 1000 positive examples of "Hey Brain" + negative samples, then:

1. Place the trained `.tflite` file in `assets/wake_word/hey-brain.tflite`
2. Update `config.yaml`:
   ```yaml
   audio:
     wake_word:
       model: "assets/wake_word/hey-brain.tflite"
   ```

### Disabling wake word (testing)

```yaml
audio:
  wake_word:
    enabled: false    # every utterance goes straight to the brain
```

---

## Run in Mock Mode

```powershell
# Recommended for laptop dev — real mic/camera/speakers + mocked Pi hardware
python -m brain --mock --laptop

# Verbose logging to see all STT/wake decisions
python -m brain --mock --laptop --log-level DEBUG
```

### What `--mock` does (and doesn't) disable

| Hardware | `--mock` behaviour |
|---|---|
| 🎙️ Laptop microphone | ✅ **Used** — real PyAudio input |
| 🔊 Laptop speakers | ✅ **Used** — real pyttsx3 TTS output |
| 📷 USB webcam | ✅ **Used** — real OpenCV capture |
| 🖥️ Web display | ✅ **Used** — browser at `localhost:8080/display` |
| GPIO pins | ❌ Mocked (no Pi GPIO on laptop) |
| SPI/I2C display | ❌ Mocked → uses web display instead |
| Pan-tilt motors | ❌ Mocked → logged as text |
| Bluetooth speaker | ❌ Mocked → uses default audio output |

> **Before this fix**: `--mock` would disable the microphone and use `[MOCK SPEAK]` for TTS.  
> **Now**: `--mock` only mocks Pi-specific hardware. Your laptop's real AV hardware is always used if available.

---

## Run the API

The REST + WebSocket API starts automatically with the brain. Access it at:

| Endpoint | URL |
|----------|-----|
| REST API | http://localhost:8080 |
| API Docs (Swagger) | http://localhost:8080/docs |
| WebSocket | ws://localhost:8080/ws |

### Useful endpoints during development

```bash
# Check brain status
curl http://localhost:8080/status

# Send a text query directly (bypasses wake word)
curl -X POST http://localhost:8080/ask \
     -H "Content-Type: application/json" \
     -d '{"text": "What can you do?"}'

# Inject TTS
curl -X POST http://localhost:8080/speak \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, I am your robot brain."}'

# Trigger emotion
curl -X POST http://localhost:8080/emotion \
     -H "Content-Type: application/json" \
     -d '{"emotion": "happy"}'
```

---

## Test Hardware Simulation

```powershell
# Verify what the detector sees on your laptop
python scripts/hw_test.py
```

Expected output on laptop:
```
mic_available      : True   (laptop mic detected)
speaker_available  : True
camera_available   : True/False (depends on webcam)
gpio_available     : False  (no Pi GPIO)
spi_display        : False  (mock display used)
i2c_available      : False
```

---

## Project Structure

```
project-abc/
├── brain/
│   ├── agents/              # 17 specialized agents
│   │   ├── auditory_agent.py    ← microphone, VAD, openwakeword, Vosk STT
│   │   ├── display_agent.py     ← emotion GIF rendering
│   │   ├── emotion_engine.py    ← 10-state affective FSM
│   │   ├── speech_agent.py      ← TTS output
│   │   ├── vision_agent.py      ← camera + face detection
│   │   └── ...
│   ├── hardware/
│   │   ├── audio_driver.py      ← PyAudio + pyttsx3/espeak
│   │   ├── display_driver.py    ← SPI LCD / OLED / mock
│   │   └── hw_detector.py       ← auto-detects available hardware
│   ├── memory/              # L1/L2/L3 memory tiers
│   ├── llm/                 # Ollama + Claude router
│   └── identity/            # Personality + Soul manager
├── config/
│   ├── config.yaml          ← main config (LLM, audio, display, motor)
│   ├── emotions.yaml        ← emotion states + GIF mapping + FSM transitions
│   └── .env                 ← secrets (copy from .env.example)
├── assets/
│   ├── emotions/            ← Otto robot GIFs (download_emotions.py)
│   └── models/              ← Vosk STT model
├── api/                     ← FastAPI REST + WebSocket
├── scripts/
│   ├── download_emotions.py ← fetch GIFs from GitHub (no API key needed)
│   ├── hw_test.py           ← hardware detection check
│   └── setup.sh             ← Pi production setup
└── data/                    ← SQLite databases + soul files (auto-created)
```

---

## Emotion System

The brain has a 10-state affective FSM. Each state plays an Otto robot GIF on the display.

| Emotion | Otto GIF | Trigger |
|---------|----------|---------|
| `NEUTRAL` | `staticstate.gif` | Default / reset |
| `ATTENTIVE` | `staticstate.gif` | Voice / motion detected |
| `FOCUSED` | `staticstate.gif` | Processing complex query |
| `HAPPY` | `happy.gif` | Task success, positive input |
| `EXCITED` | `happy.gif` | Novel interesting discovery |
| `CURIOUS` | `buxue.gif` | Exploration mode |
| `CONFUSED` | `buxue.gif` | Ambiguous input |
| `BORED` | `sad.gif` | Idle > 5 minutes |
| `FRUSTRATED` | `anger.gif` | 3× inference failures |
| `SURPRISED` | `scare.gif` | Unexpected event |

FSM transitions are defined in `config/emotions.yaml`.

---

## LLM Backends

### Local (Ollama) — recommended for dev

```powershell
# 1. Install Ollama from https://ollama.com/
# 2. Pull models
ollama pull phi3:mini
ollama pull llama3.2:3b

# 3. Ollama runs automatically — brain connects to http://localhost:11434
```

### Cloud (Anthropic Claude) — optional

Set `ANTHROPIC_API_KEY` in `config/.env`. The brain auto-routes complex queries (>800 tokens) to Claude if network is available.

### Full offline mode

```yaml
# config/config.yaml
llm:
  default_model: "phi3:mini"
  balanced_model: "llama3.2:3b"
  cloud_model: ""             # leave empty to disable cloud
```

Or set `MOCK_LLM=true` in `.env` to skip all LLM calls (returns canned responses).

---

## Troubleshooting

### PyAudio fails to install

```powershell
pip install pipwin && pipwin install pyaudio
```

### `openwakeword` not detecting wake word

- Lower the threshold: `config.yaml → audio.wake_word.threshold: 0.3`
- Make sure your mic is set as the default recording device in Windows Sound Settings
- Run `python scripts/hw_test.py` to confirm `mic_available: True`
- Switch to text-match fallback: `engine: "vosk_text"` in config.yaml

### Vosk model missing

```
WARNING  Vosk model not found at assets/models/vosk-model-small-en-us-0.15
```

Re-run the download in the [STT model section](#download-stt-model-vosk).

### Emotion GIFs show 1×1 grey box

Run `python scripts/download_emotions.py` — it will re-download from GitHub.

### Ollama connection refused

Make sure Ollama is running:
```powershell
ollama serve
```

### `No module named spacy` warning

spaCy is optional — the brain continues without it using regex-only NLP. To install it:

```powershell
# Make sure your venv is active FIRST
.venv\Scripts\Activate.ps1
pip install spacy
python -m spacy download en_core_web_sm
```

> **Important:** If `python` resolves to `C:\Users\navee\.platformio\penv\...` or any system Python instead of your venv, the brain runs with the wrong interpreter. Always activate the venv before running:
> ```powershell
> .venv\Scripts\Activate.ps1
> python -m brain --mock --laptop
> ```

### `CancelledError` in log on Ctrl+C (shutdown)

This was a uvicorn cleanup artefact — **already fixed**. Clean shutdown now logs only:
```
Received signal 2 — shutting down...
Shutting down brain...
Orchestrator stopped
Brain offline. Goodbye.
```
No more `Exception in ASGI application` spam.

### STT spamming `'huh'` or `'uh'` constantly

Vosk transcribes microphone background hiss as filler words. This is now filtered:
- Single-word noise transcripts (`huh`, `uh`, `um`, `ah`, etc.) are silently dropped
- Transcripts ≤ 3 characters are ignored
- Identical repeated transcripts within 2 seconds are debounced
- Filtered noise is visible at `DEBUG` level only

If real speech is also being filtered, lower the debounce or expand `_NOISE_WORDS` in `brain/agents/auditory_agent.py`.

To see what's being filtered:
```powershell
python -m brain --mock --laptop --log-level DEBUG
```

### pyttsx3 TTS no sound

Open **Windows Sound Settings → App volume** and verify that Python's volume is not muted.

---

## Quick Dev Workflow

```powershell
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — start the brain in mock mode
cd project-abc
.venv\Scripts\Activate.ps1
python -m brain --mock --laptop

# Terminal 3 — interact via API
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d '{"text":"hello"}'
```

Say **"Hey Jarvis"** (or enable `wake_word.enabled: false` during dev) and speak your query.

---

*Last updated: April 2026*
