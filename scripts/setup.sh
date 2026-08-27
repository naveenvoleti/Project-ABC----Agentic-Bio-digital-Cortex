#!/usr/bin/env bash
# Project-ABC Setup Script
# Run once on a fresh Raspberry Pi OS Lite (64-bit) image
set -e

echo "=== Project-ABC Setup ==="

# System packages
sudo apt-get update -y
sudo apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libopencv-dev python3-opencv \
    portaudio19-dev libportaudio2 \
    espeak-ng \
    pulseaudio pulseaudio-module-bluetooth bluez bluealsa \
    libasound2-dev \
    libatlas-base-dev \
    pigpiod \
    fonts-liberation \
    libsqlite3-dev \
    git curl wget

# Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Pull LLM models
echo "Pulling Ollama models (this may take a while)..."
ollama pull phi3:mini
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

# spaCy language model
python -m spacy download en_core_web_sm

# Download Vosk model for offline STT
mkdir -p assets/models
if [ ! -d "assets/models/vosk-model-small-en-us-0.15" ]; then
    echo "Downloading Vosk STT model..."
    wget -q https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -O /tmp/vosk.zip
    unzip -q /tmp/vosk.zip -d assets/models/
    rm /tmp/vosk.zip
fi

# Copy .env if not present
if [ ! -f config/.env ]; then
    cp config/.env.example config/.env
    echo "Created config/.env — please fill in your API keys"
fi

# Create data directories
mkdir -p data assets/emotions assets/sounds assets/fonts

# Install systemd services
echo "Installing systemd services..."
sudo cp services/brain.service /etc/systemd/system/
sudo cp services/dream.timer /etc/systemd/system/
sudo cp services/dream.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable brain.service
sudo systemctl enable dream.timer

# Enable SPI and I2C
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0

echo ""
echo "=== Setup Complete ==="
echo "1. Edit config/.env with your API keys"
echo "2. Run: python scripts/download_emotions.py"
echo "3. Run: python scripts/hw_test.py"
echo "4. Start: sudo systemctl start brain"
