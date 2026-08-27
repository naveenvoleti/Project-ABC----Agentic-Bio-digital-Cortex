#!/usr/bin/env bash
# =============================================================================
# Project-ABC v5.1 — Full Automated Installer
# Targets: WSL2 / Ubuntu 22.04+ / Raspberry Pi OS (64-bit)
#
# Usage:
#   bash scripts/install.sh              # auto-detect platform
#   bash scripts/install.sh --pi         # force Raspberry Pi mode
#   bash scripts/install.sh --skip-models  # skip Ollama + SmolVLM2 download
#   bash scripts/install.sh --mock       # dev install (no hardware deps)
# =============================================================================
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
info() { echo -e "${CYAN}  →${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail() { echo -e "${RED}  ✗ ERROR:${NC} $*"; exit 1; }
section() { echo -e "\n${BOLD}${CYAN}══ $* ══${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
FORCE_PI=false
SKIP_MODELS=false
MOCK_MODE=false
for arg in "$@"; do
    case $arg in
        --pi)           FORCE_PI=true ;;
        --skip-models)  SKIP_MODELS=true ;;
        --mock)         MOCK_MODE=true ;;
    esac
done

# ── Platform detection ────────────────────────────────────────────────────────
IS_WSL=false
IS_PI=false
IS_MAC=false

if $FORCE_PI; then
    IS_PI=true
elif grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
elif [[ "$(uname -m)" == "aarch64" ]] && grep -qi "raspberry" /proc/cpuinfo 2>/dev/null; then
    IS_PI=true
elif [[ "$(uname)" == "Darwin" ]]; then
    IS_MAC=true
fi

PLATFORM="Linux/Ubuntu"
$IS_WSL && PLATFORM="WSL2"
$IS_PI  && PLATFORM="Raspberry Pi"
$IS_MAC && PLATFORM="macOS"

echo -e "\n${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     Project-ABC v5.1 — Auto Installer    ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo -e "  Platform : ${CYAN}${PLATFORM}${NC}"
echo -e "  Mock mode: ${CYAN}${MOCK_MODE}${NC}"
echo -e "  Models   : ${CYAN}$( $SKIP_MODELS && echo 'SKIP' || echo 'download')${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
info "Working directory: $PROJECT_DIR"

# ── 1. System packages ────────────────────────────────────────────────────────
section "1 / 9  System Packages"

if $IS_MAC; then
    if ! command -v brew &>/dev/null; then
        fail "Homebrew not found. Install from https://brew.sh then re-run."
    fi
    brew install portaudio ffmpeg espeak libsndfile python@3.11 wget || true
    ok "Homebrew packages installed"
else
    sudo apt-get update -qq
    PKGS=(
        python3.11 python3.11-venv python3.11-dev python3-pip
        portaudio19-dev libportaudio2 libasound2-dev
        libsndfile1 ffmpeg espeak-ng
        libopencv-dev libatlas-base-dev
        libsqlite3-dev sqlite3
        git curl wget unzip
        fonts-liberation
    )
    # Pi-only hardware packages
    if $IS_PI; then
        PKGS+=(pigpiod i2c-tools)
    fi
    # WSL2 / desktop audio
    if $IS_WSL || ! $IS_PI; then
        PKGS+=(pulseaudio)
    fi
    # Pi full audio stack
    if $IS_PI; then
        PKGS+=(pulseaudio pulseaudio-module-bluetooth bluez)
    fi

    sudo apt-get install -y "${PKGS[@]}" -qq
    ok "System packages installed"
fi

# ── 2. Python virtual environment ─────────────────────────────────────────────
section "2 / 9  Python Virtual Environment"

if [ ! -d "venv" ]; then
    python3.11 -m venv venv
    ok "Created venv (Python 3.11)"
else
    ok "venv already exists — skipping creation"
fi

# Activate venv for the rest of the script
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip wheel setuptools -q
ok "pip upgraded"

# ── 3. Python dependencies ────────────────────────────────────────────────────
section "3 / 9  Python Dependencies"

info "Installing requirements.txt (this may take 3-5 minutes)..."
pip install -r requirements.txt -q
ok "Core packages installed"

# OpenVINO — Intel Iris Xe acceleration (WSL2 / x86_64 Linux)
if ! $IS_PI && ! $IS_MAC; then
    info "Installing OpenVINO runtime..."
    pip install openvino openvino-dev -q || warn "OpenVINO install failed — SmolVLM2 will fall back to HuggingFace"
    ok "OpenVINO installed"
fi

# Google AI SDK
pip install google-generativeai -q
ok "Google AI SDK installed"

# spaCy language model
info "Downloading spaCy en_core_web_sm..."
python -m spacy download en_core_web_sm -q
ok "spaCy model downloaded"

# ── 4. Ollama + LLM models ────────────────────────────────────────────────────
section "4 / 9  Ollama + LLM Models"

if $SKIP_MODELS; then
    warn "Skipping Ollama installation (--skip-models)"
else
    # Install Ollama if not present
    if ! command -v ollama &>/dev/null; then
        info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        ok "Ollama installed"
    else
        ok "Ollama already installed: $(ollama --version)"
    fi

    # Start Ollama server in background if not running
    if ! pgrep -x ollama &>/dev/null; then
        info "Starting Ollama server..."
        ollama serve &>/dev/null &
        sleep 3
    fi

    # Pull models
    MODELS=("llama3.2:1b" "llama3.2:3b" "nomic-embed-text")
    for model in "${MODELS[@]}"; do
        if ollama list 2>/dev/null | grep -q "^${model}"; then
            ok "Model already pulled: $model"
        else
            info "Pulling $model (this takes a while)..."
            ollama pull "$model"
            ok "Pulled: $model"
        fi
    done
fi

# ── 5. SmolVLM2 via llama.cpp ─────────────────────────────────────────────────
section "5 / 9  SmolVLM2 Vision Model"

if $SKIP_MODELS; then
    warn "Skipping SmolVLM2 download (--skip-models)"
else
    mkdir -p assets/models/smolvlm2

    GGUF_PATH="assets/models/smolvlm2/SmolVLM2-500M-Instruct-Q8_0.gguf"
    if [ ! -f "$GGUF_PATH" ]; then
        info "Downloading SmolVLM2-500M GGUF (≈600MB)..."
        wget -q --show-progress \
            "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM2-500M-Instruct-Q8_0.gguf" \
            -O "$GGUF_PATH" || warn "SmolVLM2 GGUF download failed — set backend=huggingface in config.yaml"
        ok "SmolVLM2 GGUF downloaded"
    else
        ok "SmolVLM2 GGUF already exists"
    fi

    # llama.cpp server binary
    LLAMA_BIN="assets/models/llama-server"
    if [ ! -f "$LLAMA_BIN" ]; then
        info "Downloading llama-server binary..."
        LLAMA_VER="b4450"
        if $IS_PI; then
            LLAMA_URL="https://github.com/ggerganov/llama.cpp/releases/download/${LLAMA_VER}/llama-${LLAMA_VER}-bin-ubuntu-arm64.zip"
        else
            LLAMA_URL="https://github.com/ggerganov/llama.cpp/releases/download/${LLAMA_VER}/llama-${LLAMA_VER}-bin-ubuntu-x64.zip"
        fi
        wget -q --show-progress "$LLAMA_URL" -O /tmp/llama.zip \
            && unzip -q /tmp/llama.zip llama-server -d assets/models/ \
            && chmod +x "$LLAMA_BIN" \
            && rm /tmp/llama.zip \
            || warn "llama-server download failed — SmolVLM2 will use HuggingFace backend"
        ok "llama-server downloaded"
    else
        ok "llama-server already exists"
    fi
fi

# ── 6. Vosk STT model ─────────────────────────────────────────────────────────
section "6 / 9  Vosk Speech-to-Text Model"

mkdir -p assets/models
VOSK_DIR="assets/models/vosk-model-small-en-us-0.15"
if [ ! -d "$VOSK_DIR" ]; then
    info "Downloading Vosk STT model (≈40MB)..."
    wget -q --show-progress \
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" \
        -O /tmp/vosk.zip
    unzip -q /tmp/vosk.zip -d assets/models/
    rm /tmp/vosk.zip
    ok "Vosk model downloaded"
else
    ok "Vosk model already exists"
fi

# ── 7. Kokoro TTS (ONNX) ──────────────────────────────────────────────────────
section "7 / 9  Kokoro TTS"

KOKORO_DIR="assets/models/kokoro"
if [ ! -d "$KOKORO_DIR" ]; then
    info "Downloading Kokoro TTS ONNX model (≈80MB)..."
    mkdir -p "$KOKORO_DIR"
    pip install huggingface_hub -q
    python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="hexgrad/Kokoro-82M",
    local_dir="assets/models/kokoro",
    ignore_patterns=["*.bin", "*.safetensors"],  # ONNX only
)
PYEOF
    ok "Kokoro TTS downloaded"
else
    ok "Kokoro TTS already exists"
fi

# ── 8. Emotion GIFs ───────────────────────────────────────────────────────────
section "8 / 9  Emotion GIFs"

mkdir -p assets/emotions
GIF_COUNT=$(find assets/emotions -name "*.gif" 2>/dev/null | wc -l)
if [ "$GIF_COUNT" -lt 5 ]; then
    info "Downloading emotion GIFs..."
    python scripts/download_emotions.py || warn "GIF download failed — add GIFs manually to assets/emotions/"
else
    ok "Emotion GIFs already present ($GIF_COUNT files)"
fi

# ── 9. Configuration ──────────────────────────────────────────────────────────
section "9 / 9  Configuration"

# Data directories
mkdir -p data assets/sounds assets/fonts
ok "Data directories created"

# .env file
if [ ! -f "config/.env" ]; then
    cp config/.env.example config/.env
    warn "Created config/.env — you MUST add your API keys before starting"
else
    ok "config/.env already exists"
fi

# Initialise empty soul files if missing
[ ! -f "data/SOUL.md" ]    && cp data/SOUL.md.example  data/SOUL.md    2>/dev/null || true
[ ! -f "data/USER.json" ]  && echo '{}' > data/USER.json
[ ! -f "data/WORLD.md" ]   && echo '# World Model' > data/WORLD.md
[ ! -f "data/SKILLS.json" ] && echo '{"skills":[]}' > data/SKILLS.json
ok "Soul files initialised"

# Pi: enable hardware interfaces + systemd services
if $IS_PI; then
    info "Enabling SPI and I2C..."
    sudo raspi-config nonint do_spi 0
    sudo raspi-config nonint do_i2c 0

    if [ -d "services" ]; then
        info "Installing systemd services..."
        sudo cp services/brain.service  /etc/systemd/system/
        sudo cp services/dream.timer    /etc/systemd/system/
        sudo cp services/dream.service  /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable brain.service dream.timer
        ok "systemd services installed"
    fi
fi

# Run hardware self-test (skip in mock mode)
if ! $MOCK_MODE; then
    info "Running hardware self-test..."
    python scripts/hw_test.py || warn "Hardware test reported issues — check output above"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║         Installation Complete! ✓         ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo -e "  1. Edit ${CYAN}config/.env${NC} and add your API keys"
echo -e "     • GOOGLE_AI_API_KEY  (free — aistudio.google.com)"
echo -e "     • ANTHROPIC_API_KEY  (optional — fallback tier)"
echo -e "     • OPENROUTER_API_KEY (optional — free models)"
echo ""
if ! $SKIP_MODELS; then
    echo -e "  2. Start SmolVLM2 server (in a separate terminal):"
    echo -e "     ${CYAN}assets/models/llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF --port 8090 -ngl 99${NC}"
    echo ""
    echo -e "  3. Start the brain:"
else
    echo -e "  2. Start the brain:"
fi
echo -e "     ${CYAN}source venv/bin/activate && python -m brain${NC}"
echo -e "     or in mock mode: ${CYAN}python -m brain --mock${NC}"
echo ""
echo -e "  API: ${CYAN}http://localhost:8080${NC}"
echo -e "  Docs: ${CYAN}http://localhost:8080/docs${NC}"
echo ""
