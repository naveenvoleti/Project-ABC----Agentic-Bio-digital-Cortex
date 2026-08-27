#!/usr/bin/env bash
# =============================================================================
# Project-ABC — Quick Start Script
# Skips installation; just boots the brain with optional flags.
#
# Usage:
#   bash scripts/start.sh              # normal start
#   bash scripts/start.sh --mock       # mock mode (no hardware required)
#   bash scripts/start.sh --no-ollama  # skip Ollama server check
#   bash scripts/start.sh --no-vlm    # skip SmolVLM2 server check
#   bash scripts/start.sh --mock --no-ollama --no-vlm  # fully offline dev mode
# =============================================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'; BOLD='\033[1m'
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
info() { echo -e "${CYAN}  →${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }

MOCK_MODE=false
START_OLLAMA=true
START_VLM=true
BRAIN_EXTRA_ARGS=()

for arg in "$@"; do
    case $arg in
        --mock)       MOCK_MODE=true;       BRAIN_EXTRA_ARGS+=("--mock") ;;
        --no-ollama)  START_OLLAMA=false ;;
        --no-vlm)     START_VLM=false ;;
        *)            BRAIN_EXTRA_ARGS+=("$arg") ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║       Project-ABC — Quick Start          ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${NC}"
echo -e "  Mock : ${CYAN}${MOCK_MODE}${NC}   Ollama: ${CYAN}${START_OLLAMA}${NC}   VLM: ${CYAN}${START_VLM}${NC}"
echo ""

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    warn "venv not found — run 'bash scripts/install.sh' first"
    exit 1
fi
source venv/bin/activate
ok "venv activated"

# ── Ollama server ──────────────────────────────────────────────────────────────
if $START_OLLAMA && ! $MOCK_MODE; then
    if command -v ollama &>/dev/null; then
        if ! pgrep -x ollama &>/dev/null; then
            info "Starting Ollama server in background..."
            ollama serve &>/dev/null &
            sleep 2
            ok "Ollama started (pid $!)"
        else
            ok "Ollama already running"
        fi
    else
        warn "Ollama not installed — LLM will fall back to Google AI / OpenRouter"
    fi
fi

# ── SmolVLM2 llama-server ──────────────────────────────────────────────────────
LLAMA_BIN="assets/models/llama-server"
GGUF_PATH="assets/models/smolvlm2/SmolVLM2-500M-Instruct-Q8_0.gguf"
if $START_VLM && ! $MOCK_MODE; then
    if [ -f "$LLAMA_BIN" ] && [ -f "$GGUF_PATH" ]; then
        if ! curl -sf http://localhost:8090/health &>/dev/null; then
            info "Starting SmolVLM2 server on :8090..."
            "$LLAMA_BIN" \
                --model "$GGUF_PATH" \
                --port 8090 \
                --ctx-size 2048 \
                --n-gpu-layers 99 \
                --host 127.0.0.1 \
                &>/tmp/llama-server.log &
            sleep 3
            if curl -sf http://localhost:8090/health &>/dev/null; then
                ok "SmolVLM2 server started (pid $!)"
            else
                warn "SmolVLM2 server may still be loading — check /tmp/llama-server.log"
            fi
        else
            ok "SmolVLM2 server already running on :8090"
        fi
    else
        warn "SmolVLM2 model/binary not found — vision will use HuggingFace fallback"
        warn "  Run 'bash scripts/install.sh' to download models"
    fi
fi

# ── Launch brain ───────────────────────────────────────────────────────────────
echo ""
info "Starting brain${BRAIN_EXTRA_ARGS:+ (${BRAIN_EXTRA_ARGS[*]})}..."
echo ""
exec python -m brain "${BRAIN_EXTRA_ARGS[@]}"
