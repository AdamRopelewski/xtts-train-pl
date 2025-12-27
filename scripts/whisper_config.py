"""Shared configuration for whisper scripts.

This module contains constants extracted from `whisper.py` so other helper scripts
(e.g., `whisper_verbal.py`) can import configuration without importing `whisper`.
"""

import os

# Directories
MODELS_DIR = os.path.join(os.getcwd(), "models")
# Default to a top-level 'whisper-models' directory for downloaded/managed Whisper models
WHISPER_MODELS_DIR = os.path.join(os.getcwd(), "whisper-models")

# Whisper settings
# WHISPER_MODEL = "large-v2" 
WHISPER_MODEL = "turbo"
# Prefer GPU by default when available
WHISPER_DEVICE = "cuda"  # 'cuda' | 'cpu' | 'auto'
WHISPER_COMPUTE_TYPE = "float16"  # prefer float16 on GPU for speed

# Transcription parameters
WHISPER_WORD_TIMESTAMPS = False
# Disable VAD by default — not needed for these clips
WHISPER_VAD_FILTER = False
WHISPER_VAD_MIN_SILENCE_DURATION_MS = 200
WHISPER_LANGUAGE = "pl"
WHISPER_TEMPERATURE = 0.0
WHISPER_BEAM_SIZE = 5
WHISPER_BEST_OF = 5
WHISPER_PATIENCE = 1.0
WHISPER_CONDITION_ON_PREVIOUS_TEXT = True
WHISPER_INITIAL_PROMPT = None

# By default we DO NOT write per-clip .txt files (use single JSONL output only)
WHISPER_WRITE_PER_CLIP = False

# Hallucination prevention
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
WHISPER_LOG_PROB_THRESHOLD = -1.0
WHISPER_NO_SPEECH_THRESHOLD = 0.6

# Mode-specific defaults (accurate)
WHISPER_MODE_DEFAULT = "safe"
WHISPER_NO_REPEAT_NGRAM_SIZE_ACCURATE = 0
WHISPER_REPETITION_PENALTY_ACCURATE = 1.0

# Mode-specific defaults (safe)
WHISPER_NO_REPEAT_NGRAM_SIZE_SAFE = 2
WHISPER_REPETITION_PENALTY_SAFE = 1.2


# Defaults
DATA_DIR = os.environ.get("XTTS_DATA_DIR", os.path.join(os.getcwd(), "train-data"))
DEFAULT_AUDIO_DIR = os.path.join(os.getcwd(), "train-data", "wavs")
DEFAULT_TRANSCRIPT_DIR = os.path.join(os.getcwd(), "train-data", "transcriptions")
# Default output file will be computed per audio-dir if not provided
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_TRANSCRIPT_DIR, "transcriptions.jsonl")
