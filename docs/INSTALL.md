# Project-ABC v7.0 — Installation Guide

Two paths: **Automated** (one command) or **Manual** (step by step).

---

## Quick Install (Automated)

> **Platform:** Linux / macOS / WSL2 / Raspberry Pi OS only.
> Windows users: see [Windows (Native) Installation](#windows-native-installation) below, or install WSL2 and run this from inside it.

```bash
git clone <repo> project-abc && cd project-abc
bash scripts/install.sh
```

Flags:
| Flag | Effect |
|---|---|
| `--pi` | Force Raspberry Pi mode (SPI/I2C/systemd) |
| `--skip-models` | Skip Ollama + SmolVLM2 download (fast dev setup) |
| `--mock` | Skip hardware self-test at the end |

```bash
# Examples
bash scripts/install.sh --skip-models   # fast first run, add models later
bash scripts/install.sh --pi            # Raspberry Pi
bash scripts/install.sh --mock          # dev machine, no hardware
```

---

## Manual Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 works, 3.10 does not |
| OS | Ubuntu 22.04+ / macOS 13+ / Windows 10 22H2+ | WSL2 fully supported |
| Ollama | latest | local LLM inference |
| 8 GB RAM | — | 16 GB recommended |
| Disk space | ~5 GB | models + venv |

---

### Step 1 — System Packages

**Ubuntu / WSL2 / Debian:**

```bash
sudo apt-get update
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    portaudio19-dev libportaudio2 libasound2-dev \
    libsndfile1 ffmpeg espeak-ng \
    libopencv-dev libatlas-base-dev \
    libsqlite3-dev sqlite3 \
    git curl wget unzip fonts-liberation \
    pulseaudio
```

**Raspberry Pi (add these too):**

```bash
sudo apt-get install -y \
    pigpiod i2c-tools \
    pulseaudio-module-bluetooth bluez
```

**macOS:**

```bash
brew install portaudio ffmpeg espeak libsndfile python@3.11 wget
```

---

### Step 2 — Python Virtual Environment

```bash
# Create venv
python3.11 -m venv venv

# Activate (do this every session)
source venv/bin/activate          # Linux / macOS / WSL2
# venv\Scripts\activate           # Windows CMD (not recommended for this project)

# Upgrade pip
pip install --upgrade pip wheel setuptools
```

---

### Step 3 — Python Dependencies

```bash
pip install -r requirements.txt

# OpenVINO (Intel Iris Xe acceleration — skip on Pi or macOS)
pip install openvino openvino-dev

# Google AI SDK (free Gemini access — required for V7.0 ER + VLA reasoning)
pip install google-generativeai

# V6.0 — semantic reflex cosine similarity
pip install scikit-learn

# V6.0 — I2C device driver synthesis (Raspberry Pi only; skip on dev machines)
# pip install smbus2

# spaCy language model (must run after spacy installs)
python -m spacy download en_core_web_sm
```

**What gets installed and why:**

| Package | Why |
|---|---|
| `fastapi` + `uvicorn` | REST + WebSocket API server |
| `anthropic` | Claude fallback LLM tier |
| `ollama` | Local LLM SDK |
| `vosk` | Offline speech-to-text |
| `pyaudio` | Microphone capture |
| `edge-tts` | Free neural TTS (no API key) |
| `opencv-python-headless` | Camera capture + face detection |
| `chromadb` | Semantic vector memory |
| `sentence-transformers` | Offline embeddings fallback |
| `psutil` | CPU/RAM/temp sensing for InteroceptionAgent |
| `openwakeword` | "Hey Brain" wake-word detection |
| `scikit-learn` | Cosine similarity for semantic reflex matching (V6.0) |
| `smbus2` | I2C device driver synthesis — Raspberry Pi only (V6.0) |

---

### Step 4 — Ollama + Local LLM Models

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server (runs in background)
ollama serve &

# Pull models (total ~4 GB)
ollama pull llama3.2:1b          # fast local model — Tier 1
ollama pull llama3.2:3b          # balanced local model — Tier 2
ollama pull nomic-embed-text     # embedding model for semantic memory

# Optional (better quality, larger)
ollama pull phi3:mini            # ECO mode fallback
```

**LLM routing order** (configured in `config/config.yaml`):

| Tier | Model | When used |
|---|---|---|
| 1 | `llama3.2:1b` (Ollama) | < 150 tokens, offline |
| 2 | `llama3.2:3b` (Ollama) | < 800 tokens, offline |
| 3 | Google AI Gemini 2.0 Flash | Primary cloud (free tier) |
| 4 | OpenRouter free | Cloud fallback |
| 5 | Claude haiku-4-5 | Paid fallback (ANTHROPIC_API_KEY) |
| 6 | `llama3.2:3b` | All cloud unavailable |
| ECO | `llama3.2:1b` only | Auto-triggered by InteroceptionAgent |
| **ER** | **Gemini 2.0 Flash** | **V7.0 — long-horizon task decomposition** |
| **VLA** | **Gemini 2.0 Flash** | **V7.0 — motor token generation** |

---

### Step 5 — SmolVLM2 Vision Model

SmolVLM2 is the 6-frame video understanding model. Two backend options:

#### Option A — llama.cpp (Recommended, fastest)

```bash
mkdir -p assets/models/smolvlm2

# Download the GGUF model (≈600 MB)
wget "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM2-500M-Instruct-Q8_0.gguf" \
    -O assets/models/smolvlm2/SmolVLM2-500M-Instruct-Q8_0.gguf

# Download llama-server binary (x86_64 Ubuntu / WSL2)
wget "https://github.com/ggerganov/llama.cpp/releases/download/b4450/llama-b4450-bin-ubuntu-x64.zip" \
    -O /tmp/llama.zip
unzip /tmp/llama.zip llama-server -d assets/models/
chmod +x assets/models/llama-server
rm /tmp/llama.zip

# Start the server (run this before starting Brain)
assets/models/llama-server \
    --model assets/models/smolvlm2/SmolVLM2-500M-Instruct-Q8_0.gguf \
    --port 8090 -ngl 99
```

#### Option B — HuggingFace (No llama.cpp required)

```bash
# In config/config.yaml, set:
#   vision.smolvlm2.backend: "huggingface"
pip install transformers accelerate
# Model downloads automatically on first Brain start (~1.2 GB)
```

#### Option C — OpenVINO (Intel Iris Xe GPU)

```bash
# Run the export script first
python scripts/export_smolvlm2_openvino.py

# In config/config.yaml, set:
#   vision.smolvlm2.backend: "openvino"
#   vision.smolvlm2.openvino_device: "GPU"
```

---

### Step 6 — Vosk Speech-to-Text Model

```bash
mkdir -p assets/models

wget "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" \
    -O /tmp/vosk.zip
unzip /tmp/vosk.zip -d assets/models/
rm /tmp/vosk.zip

# Verify
ls assets/models/vosk-model-small-en-us-0.15/
```

For better accuracy (larger model, Pi may be slow):

```bash
# Full model — 1.8 GB, much better accuracy
wget "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip" \
    -O /tmp/vosk-large.zip
unzip /tmp/vosk-large.zip -d assets/models/
```

Update `config/config.yaml` if using a different model name:
```yaml
# (no explicit path — VisionAgent auto-detects first vosk-model-* in assets/models/)
```

---

### Step 7 — Kokoro TTS (Text-to-Speech)

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="hexgrad/Kokoro-82M",
    local_dir="assets/models/kokoro",
    ignore_patterns=["*.bin", "*.safetensors"],
)
EOF
```

If Kokoro fails to load, Brain automatically falls back to:
1. ElevenLabs (if `ELEVENLABS_API_KEY` set)
2. Edge TTS (Microsoft neural, free, no key)
3. espeak-ng (robotic but always works)

Set TTS preference in `config/config.yaml`:
```yaml
audio:
  output:
    tts_engine: "auto"    # auto | kokoro | elevenlabs | edge | espeak
```

---

### Step 8 — Emotion GIFs

```bash
python scripts/download_emotions.py
```

This downloads JoyPixels animated GIFs for the 10 emotion states into `assets/emotions/`.

Manual alternative — place any GIF files named after emotions:

```
assets/emotions/
├── neutral.gif
├── happy.gif
├── curious.gif
├── bored.gif
├── focused.gif
├── confused.gif
├── frustrated.gif
├── surprised.gif
├── excited.gif
└── loading.gif    ← boot screen (any spinner/hourglass GIF)
```

---

### Step 9 — API Keys

```bash
cp config/.env.example config/.env
nano config/.env     # or use any text editor
```

```env
# Free — required for cloud LLM tiers 3 and 4
GOOGLE_AI_API_KEY=your_key_here       # aistudio.google.com → Get API key (free)

# Optional — paid fallback
ANTHROPIC_API_KEY=your_key_here       # console.anthropic.com

# Optional — free cloud models via OpenRouter
OPENROUTER_API_KEY=your_key_here      # openrouter.ai (free tier available)

# Optional — ElevenLabs neural TTS
ELEVENLABS_API_KEY=your_key_here
```

**Without any API keys:** Brain runs fully offline using Ollama local models. Quality is lower but fully functional.

---

### Step 10 — Directory Structure

Ensure these directories exist:

```bash
mkdir -p data assets/emotions assets/sounds assets/fonts
mkdir -p data/protected    # V6.0 — Digital DNA brainstem checksum (write-protected on Pi)
mkdir -p brain/sandbox     # V6.0 — AI-generated code staging area
mkdir -p brain/reflexes    # V6.0 — promoted high-invocation reflexes (permanent hardcode)

# Initialise soul files if missing
[ ! -f data/SOUL.md ]     && echo "# Soul" > data/SOUL.md
[ ! -f data/USER.json ]   && echo '{}' > data/USER.json
[ ! -f data/WORLD.md ]    && echo "# World Model" > data/WORLD.md
[ ! -f data/SKILLS.json ] && echo '{"skills":[],"reflexes":[],"version":"2.0","last_updated":""}' > data/SKILLS.json

# Pi only — restrict DNA store from being overwritten by agents
# chmod 444 data/protected/dna.json   # run after first brain boot creates it
```

---

### Step 11 — Hardware Validation

```bash
# Activate venv first
source venv/bin/activate

# Run self-test (detects camera, mic, speaker, motors, GPU)
python scripts/hw_test.py
```

Expected output on WSL2 dev machine:
```
[OK]  microphone — device 0
[OK]  speaker — pulse
[--]  camera — not detected (use --mock)
[--]  motors — not detected (use --mock)
[OK]  Ollama — llama3.2:1b responding
[OK]  Google AI — connected
[OK]  SQLite FTS5 — ok
[OK]  ChromaDB — ok
```

---

### Step 12 — Raspberry Pi Only

#### Enable hardware interfaces

```bash
sudo raspi-config nonint do_spi 0    # SPI (display)
sudo raspi-config nonint do_i2c 0    # I2C (servo HAT)
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

#### Install systemd services

```bash
sudo cp services/brain.service  /etc/systemd/system/
sudo cp services/dream.timer    /etc/systemd/system/
sudo cp services/dream.service  /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable brain.service dream.timer

# Start immediately
sudo systemctl start brain
```

#### Check service logs

```bash
sudo journalctl -u brain -f
```

---

## Windows (Native) Installation

> WSL2 is the smoothest path on Windows — run `wsl --install` and follow the Linux steps above. Use native Windows only if WSL2 is unavailable.

Open **PowerShell as Administrator** for all steps below.

---

### Step 1 — Python 3.11

```powershell
winget install Python.Python.3.11
# Close and reopen PowerShell after installing
python --version    # should print 3.11.x
```

Or download from [python.org/downloads](https://python.org/downloads) — tick **"Add Python to PATH"** during setup.

---

### Step 2 — System Dependencies

```powershell
# FFmpeg (audio/video processing)
winget install Gyan.FFmpeg

# Git
winget install Git.Git

# Visual C++ Redistributable (required by several Python packages)
winget install Microsoft.VCRedist.2015+.x64
```

**espeak-ng (TTS offline fallback):**
Download `espeak-ng-X.XX-x64.msi` from [github.com/espeak-ng/espeak-ng/releases](https://github.com/espeak-ng/espeak-ng/releases), run it, then add `C:\Program Files\eSpeak NG\` to your System PATH.

---

### Step 3 — Ollama

```powershell
winget install Ollama.Ollama
# Restart PowerShell — Ollama runs as a background Windows service automatically
ollama --version
```

Or download `OllamaSetup.exe` from [ollama.com](https://ollama.com).

---

### Step 4 — Python Virtual Environment

```powershell
cd C:\path\to\project-abc

python -m venv venv

# Activate — PowerShell
.\venv\Scripts\Activate.ps1

# If blocked by execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# Activate — CMD (alternative)
# venv\Scripts\activate.bat

pip install --upgrade pip wheel setuptools
```

---

### Step 5 — Python Dependencies

PyAudio requires PortAudio. On Windows use the prebuilt wheel via `pipwin`:

```powershell
pip install pipwin
pipwin install pyaudio
```

> If `pipwin` fails, download the matching `.whl` from [lfd.uci.edu/~gohlke/pythonlibs/#pyaudio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install with:
> `pip install pyaudio-0.2.XX-cpXXX-win_amd64.whl`

Then install the rest:

```powershell
pip install -r requirements.txt
pip install google-generativeai
python -m spacy download en_core_web_sm
```

---

### Step 6 — Ollama Models

Ollama service is already running as a Windows service — no `ollama serve` needed.

```powershell
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

---

### Step 7 — SmolVLM2 Vision Model

```powershell
New-Item -ItemType Directory -Force -Path assets\models\smolvlm2

# Start-BitsTransfer is built into Windows, handles redirects, shows progress
Start-BitsTransfer `
    -Source "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf" `
    -Destination "assets\models\smolvlm2\SmolVLM-500M-Instruct-Q8_0.gguf"

# Download llama-server Windows binary (GitHub releases are fine with Invoke-WebRequest)
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest `
    -Uri "https://github.com/ggerganov/llama.cpp/releases/download/b4450/llama-b4450-bin-win-avx2-x64.zip" `
    -OutFile "llama-win.zip"
Expand-Archive -Path llama-win.zip -DestinationPath assets\models\ -Force
Remove-Item llama-win.zip
```

---

### Step 8 — Vosk STT Model

```powershell
New-Item -ItemType Directory -Force -Path assets\models

$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest `
    -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" `
    -OutFile "vosk.zip"
Expand-Archive -Path vosk.zip -DestinationPath assets\models\ -Force
Remove-Item vosk.zip
```

---

### Step 9 — Kokoro TTS

```powershell
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='hexgrad/Kokoro-82M',
    local_dir='assets/models/kokoro',
    ignore_patterns=['*.bin', '*.safetensors'],
)
"
```

---

### Step 10 — API Keys and Data Directories

```powershell
Copy-Item config\.env.example config\.env
notepad config\.env    # add GOOGLE_AI_API_KEY and any optional keys

# Runtime directories
New-Item -ItemType Directory -Force -Path data, assets\emotions, assets\sounds, assets\fonts

# Initialise soul files
if (-not (Test-Path data\SOUL.md))     { "# Soul"         | Out-File -Encoding utf8 data\SOUL.md }
if (-not (Test-Path data\USER.json))   { '{}'              | Out-File -Encoding utf8 data\USER.json }
if (-not (Test-Path data\WORLD.md))    { "# World Model"  | Out-File -Encoding utf8 data\WORLD.md }
if (-not (Test-Path data\SKILLS.json)) { '{"skills":[],"reflexes":[],"version":"2.0","last_updated":""}' | Out-File -Encoding utf8 data\SKILLS.json }
# V6.0 runtime directories
New-Item -ItemType Directory -Force -Path data\protected, brain\sandbox, brain\reflexes
```

---

### Step 11 — Hardware Validation

```powershell
.\venv\Scripts\Activate.ps1
python scripts/hw_test.py
```

---

### Step 12 — Start the Brain (Windows)

```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Terminal 1 — SmolVLM2 vision server
.\assets\models\llama-server.exe `
    --model assets\models\smolvlm2\SmolVLM2-500M-Instruct-Q8_0.gguf `
    --port 8090 -ngl 99

# Terminal 2 — Brain (normal)
python -m brain

# Brain in mock mode (no hardware required)
python -m brain --mock
```

---

### Windows Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1` is blocked | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pyaudio` install fails | `pip install pipwin` then `pipwin install pyaudio` |
| `ollama` not found after install | Restart PowerShell; check PATH includes `%LOCALAPPDATA%\Programs\Ollama` |
| `ffmpeg` not found | Restart PowerShell; or run `winget install Gyan.FFmpeg` again |
| `espeak` not found | Add `C:\Program Files\eSpeak NG\` to System PATH in Windows Settings |
| No microphone detected | Set default mic in Windows Sound → Recording tab |
| Camera shows black frame | Try `usb_device_index: 1` or `2` in `config/hardware.yaml` |
| `pigpiod` / `raspi-config` errors | Pi-only hardware — use `--mock` flag on Windows |
| ChromaDB install fails | `pip install chromadb --no-build-isolation` |
| `dlib` / face_recognition fails | Pre-built wheel: `pip install dlib‑XX.X‑cpXXX‑win_amd64.whl` from Gohlke; or set `face_recognition.mock: true` in `config/config.yaml` |

---

## Starting the Brain

### Development (WSL2 / laptop)

#### Quick start (recommended)

`scripts/start.sh` handles venv activation, Ollama server, and SmolVLM2 in a single command:

```bash
bash scripts/start.sh                              # normal start
bash scripts/start.sh --mock                       # mock mode — no hardware required
bash scripts/start.sh --no-ollama                  # skip Ollama server check/start
bash scripts/start.sh --no-vlm                     # skip SmolVLM2 server check/start
bash scripts/start.sh --mock --no-ollama --no-vlm  # fully offline dev mode
```

| Flag | Effect |
|---|---|
| _(none)_ | Activates venv, starts Ollama + SmolVLM2 if not running, then launches brain |
| `--mock` | Passes `--mock` to the brain; skips hardware init — safe on any laptop |
| `--no-ollama` | Skips Ollama start check; LLM falls back to Google AI / OpenRouter |
| `--no-vlm` | Skips SmolVLM2 llama-server start; vision uses HuggingFace fallback |
| `--mock --no-ollama --no-vlm` | Fully offline, no model servers needed — fastest dev cycle |

#### Manual start (multi-terminal)

```bash
source venv/bin/activate

# Terminal 1 — SmolVLM2 vision server
assets/models/llama-server \
    --model assets/models/smolvlm2/SmolVLM2-500M-Instruct-Q8_0.gguf \
    --port 8090 -ngl 99

# Terminal 2 — Brain
python -m brain

# Or without camera/mic (pure software test)
python -m brain --mock
```

### Production (Raspberry Pi)

```bash
sudo systemctl start brain
sudo journalctl -u brain -f
```

---

## API Access

Once running:

| Endpoint | Description |
|---|---|
| `http://localhost:8080/docs` | Interactive API docs (Swagger) |
| `http://localhost:8080/status` | Brain health, emotion, behavior |
| `http://localhost:8080/workspace` | GWT Spotlight — top-3 salient messages |
| `http://localhost:8080/world` | Live world snapshot + internal thought |
| `http://localhost:8080/drives` | Homeostatic drive levels |
| `ws://localhost:8080/ws` | Real-time event stream |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'brain'`
```bash
# You must run from the project root with venv active
source venv/bin/activate
python -m brain    # NOT python brain/__main__.py
```

### Ollama not responding
```bash
ollama serve &      # start in background
ollama list         # verify models
```

### No audio on WSL2
WSL2 doesn't have direct audio access. Options:
1. Set `tts_engine: edge` in config (streams via Edge browser API)
2. Set `tts_engine: espeak` (saves to file, auto-plays via Windows)
3. Use PulseAudio over TCP: see [WSL2 audio guide](https://github.com/microsoft/WSL/issues/237)

### SmolVLM2 server not found
```bash
# Check it's running on port 8090
curl http://localhost:8090/health

# If not, start it:
assets/models/llama-server --model assets/models/smolvlm2/SmolVLM2-500M-Instruct-Q8_0.gguf --port 8090
```

### `psutil` not installed
```bash
pip install psutil
# Required for InteroceptionAgent CPU/RAM/temp monitoring
```

### ChromaDB import error
```bash
pip install chromadb --upgrade
# If still failing: pip install chromadb==0.5.20
```

### `No module named 'sklearn'` (V6.0 semantic reflexes)
```bash
pip install scikit-learn
# Required by ReasoningAgent._check_reflexes() for cosine similarity
```

### `No module named 'smbus2'` (V6.0 I2C driver synthesis)
```bash
pip install smbus2
# Required for SynthesisAgent._synthesize_driver() on Raspberry Pi
# Safe to skip on dev machines — synthesis will fall back to mock driver
```

### V6.0 SynthesisAgent fires unexpectedly
The SynthesisAgent triggers when the same agent crashes 3+ times (configurable in `config/config.yaml` under `synthesis_agent.crash_threshold`). During development, increase the threshold or set `synthesis_agent.enabled: false` to disable auto-rewrite.

### V7.0 ER / VLA calls return empty plans
Ensure `GOOGLE_AI_API_KEY` is set in `config/.env`. The Gemini 2.0 Flash model is used for both ER reasoning and VLA motor token generation. Without a valid key, both fall back to plain `infer()` with `success_estimate=0.5`.

### `data/protected/dna.json` missing after boot
The DNA file is created on first boot by SynthesisAgent (`_snapshot_dna()`). If missing after boot, check that `synthesis_agent.enabled: true` in `config/config.yaml` and that the brain started without errors.

### Camera not detected
```bash
# WSL2: Camera pass-through is experimental. Use mock mode for dev:
python -m brain --mock

# Or enable USB camera in WSL2 via usbipd:
# https://learn.microsoft.com/en-us/windows/wsl/connect-usb
```

---

## Updating

```bash
git pull
source venv/bin/activate
pip install -r requirements.txt --upgrade -q
python -m spacy download en_core_web_sm
```

No database migrations needed — SQLite schema auto-updates on next start.

**V6.0 / V7.0 specific:**
- `data/SKILLS.json` schema upgraded to v2.0 (`reflexes` key added) — run the directory init commands in Step 10 to reinitialise if upgrading from V5.x
- `data/protected/dna.json` is created automatically on first boot after upgrading
- New Python dependencies: `scikit-learn` (all platforms), `smbus2` (Pi only)
- New config blocks in `config/config.yaml`: `wheel_drive`, `synthesis_agent`, `tester_agent`, `reflex_registry`, `mirror_agent`, `gemini_robotics` — already present if you cloned fresh; merge manually if upgrading
