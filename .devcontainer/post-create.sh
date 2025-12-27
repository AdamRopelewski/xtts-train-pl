#!/bin/bash
set -e


pip install --no-cache-dir torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

pip install --no-cache-dir faster-whisper --break-system-packages



# Clone XTTS if not already present (devcontainer copy may already bind-mount workspace)
if [ ! -d /workspace/coqui-tts ]; then
    git clone https://github.com/idiap/coqui-ai-TTS /workspace/coqui-tts
fi
cd /workspace/coqui-tts || exit 0
pip install -e . --break-system-packages
