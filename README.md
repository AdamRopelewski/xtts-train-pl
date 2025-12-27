# XTTS Polish Training Pipeline — End-to-End 📚🇵🇱

This repository provides a full pipeline to train an XTTS (XTTS v2) model for Polish from raw WAV files:

WAVs → Whisper transcription → Polish text normalization (ByT5) → LJSpeech-style metadata CSV → XTTS training (Coqui-TTS / local helper)

It includes helper scripts to run each stage and a devcontainer that sets up a GPU-enabled environment for easy reproducible runs.

---

## Contents
- `scripts/whisper.py` — transcribe WAVs using faster-whisper (outputs a JSONL with `whisper_raw` per clip)
- `scripts/prepare_metadata.py` — prepares dataset JSONL from raw Whisper output, optionally runs local ByT5 normalization and writes CSV (`basename|preserve|expanded`)
- `jsonl_to_ljspeech_metadata.py` — convert JSONL (with `variants.preserve|expanded`) to LJSpeech-style `metadata.csv`
- `train.py` — helper script to launch XTTS training (uses `xtts-base-model` files)
- `coqui-tts/` — cloned and installed Coqui-TTS (if you use the devcontainer)
- `train-data/wavs/` — place your WAVs here
- `train-data/transcriptions/` — transcription JSONL and CSV outputs
- `scripts/remove_optimalizer.py` — for removing the optimalizer from the train model (decresses the file size a lot)
---

## Quick End-to-End Guide ✅

### 0) Prepare environment
Recommended: open repository in VS Code and choose **Reopen in Container** if `.devcontainer/` exists. This will install CUDA PyTorch and required libs automatically.

Manual (non-devcontainer):
```bash
# Example for CUDA 13 / PyTorch (adjust to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install faster-whisper transformers tqdm pandas
```

---

### 1) Transcribe WAVs with Whisper
Ensure your audio files are in `train-data/wavs/`.

```bash
python scripts/whisper.py \
  --audio-dir train-data/wavs \
  --output-file train-data/transcriptions/wavs_transcriptions.jsonl
```

Options:
- `--max-files N` to test on N files.
- `scripts/whisper_config.py` contains defaults: `WHISPER_MODEL`, `WHISPER_DEVICE`, compute settings.

---

### 2) Prepare metadata and normalize (ByT5)
**Important:** You must manually place the ByT5 text-normalization model in `./byt5-small-pl-text-normalization/` (files required: `config.json`, model weights `model.safetensors`, and tokenizer files such as `tokenizer_config.json` and `config.json`). \
[huggingface](https://huggingface.co/Folx/byt5-small-pl-text-normalization)

- Only prepare CSV from raw Whisper (no normalization):
```bash
python scripts/prepare_metadata.py \
  -i train-data/transcriptions/wavs_transcriptions.jsonl \
  --output train-data/transcriptions/wavs_transcriptions.prepared.jsonl \
  --csv-output train-data/transcriptions/metadata_t5_from_raw.csv \
  --no-t5
```

- Run T5 normalization (recommended: GPU):
```bash
python scripts/prepare_metadata.py \
  -i train-data/transcriptions/wavs_transcriptions.jsonl \
  --output train-data/transcriptions/wavs_transcriptions.t5prepared.jsonl \
  --csv-output train-data/transcriptions/metadata_t5_from_raw.csv \
  --device cuda --num-beams 1 --overwrite
```
Options:
- `--no-t5`: prepare CSV without running the model
- `--limit N`: process only first N lines (useful for testing)
- `--dry-run`: show a few sample conversions

Cleaning rules applied before and after T5 (`clean_text`): remove leading quotes/dashes, drop `-` after punctuation, replace remaining dashes with commas, collapse whitespace.

**If the model fails to load**: ensure `config.json` and tokenizer files are present. The scripts will raise informative errors if files are missing.

---

### 3) Convert to LJSpeech `metadata.csv` (for training)
If your training recipe expects LJSpeech-style `metadata.csv`:

```bash
python jsonl_to_ljspeech_metadata.py \
  --input train-data/transcriptions/wavs_transcriptions.t5prepared.jsonl \
  --output train-data/csv/metadata.csv
```

---

### 4) XTTS base model (manual — required for training)
Download or place the XTTS base model files under `xtts-base-model/xttsv2_2.0.2/`:
- `model.pth` (model weights)
- `dvae.pth`
- `mel_stats.pth`
- `vocab.json`
- `config.json`
- `speakers_xtts.pth`

We do **not** auto-download these due to licensing and size.

---

### 5) Train XTTS (local helper or Coqui-TTS recipe)
- Quick start with the helper:
```bash
python train.py
```
- Or use Coqui-TTS recipes:
```bash
python coqui-tts/recipes/ljspeech/xtts_v2/train_gpt_xtts.py --dataset /path/to/your/dataset
```

Check `train.py` for default checkpoint paths and hyperparameters. Use GPU for practical training times.

---

### Generate sample audio (qualitative validation)
Synthesize a short audio file from a checkpoint for quick subjective checks.

```bash
# Example using the 'tts' CLI (equivalent defaults):
tts --text "Dzień dobry" \
  --model_path xtts-base-model/xttsv2_2.0.2 \
  --config_path xtts-base-model/xttsv2_2.0.2/config.json \
  --out_path run/samples/sample.wav \
  --language_idx pl \
  --speaker_wav train-data/speaker_reference/ref.wav \
  --use_cuda
```

Run the repository helper with the same defaults:
```bash
python scripts/generate_sample.py  # runs the same default command (text="Dzień dobry", language_idx=pl, speaker_wav=train-data/speaker_reference/ref.wav, uses CUDA if available)
```

Note:
`--model_path` may point to a model directory (Coqui will resolve its checkpoint/config) or a direct `model.pth` file if you prefer.


Speaker options:
- `--speaker_idx <id>` to pick a speaker from `speakers_xtts.pth` (multi-speaker models)
- `--speaker_wav <path>` to condition on an example speaker audio if supported

Check the sample output in `run/samples/` and compare to references as part of qualitative validation.


## Useful flags
- `--no-t5` (prepare CSV without running T5)
- `--limit N` (process only first N entries)
- `--dry-run` (print samples)
- `--device cuda|cpu`
- `--num-beams N`

