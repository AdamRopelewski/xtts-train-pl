#!/usr/bin/env python3


from faster_whisper import WhisperModel
import os
import sys
import json
import argparse
from datetime import datetime

from whisper_config import (
    MODELS_DIR,
    WHISPER_MODELS_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_WORD_TIMESTAMPS,
    WHISPER_VAD_FILTER,
    WHISPER_VAD_MIN_SILENCE_DURATION_MS,
    WHISPER_LANGUAGE,
    WHISPER_TEMPERATURE,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    WHISPER_PATIENCE,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_INITIAL_PROMPT,
    WHISPER_WRITE_PER_CLIP,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_MODE_DEFAULT,
    WHISPER_NO_REPEAT_NGRAM_SIZE_ACCURATE,
    WHISPER_REPETITION_PENALTY_ACCURATE,
    WHISPER_NO_REPEAT_NGRAM_SIZE_SAFE,
    WHISPER_REPETITION_PENALTY_SAFE,
    DATA_DIR,
    DEFAULT_AUDIO_DIR,
    DEFAULT_TRANSCRIPT_DIR,
    DEFAULT_OUTPUT_FILE,
)








def transcribe_file(model, audio_path, language=None, transcribe_opts=None):
    """Run model.transcribe with given options and return a dict with raw transcript and segment list.

    `transcribe_opts` is passed as keyword args to `model.transcribe`.
    Preserves the Whisper output verbatim by concatenating segment.text exactly as returned.
    """
    if transcribe_opts is None:
        transcribe_opts = {}
    # Always pass language explicitly if provided
    if language is not None:
        transcribe_opts['language'] = language

    segments_iter, _ = model.transcribe(audio_path, **transcribe_opts)

    # Since segments_iter may be a generator, turn into list to iterate twice
    segments = list(segments_iter)

    # Preserve raw text exactly as returned by Whisper (no lstrip/join normalization)
    whisper_raw = "".join(segment.text for segment in segments)
    segs = []
    for segment in segments:
        words = getattr(segment, "words", None)
        if words is not None:
            # Convert Word objects to simple serializable dicts
            serial_words = []
            for w in words:
                try:
                    serial_words.append({"start": getattr(w, "start", None), "end": getattr(w, "end", None), "text": getattr(w, "word", None) or getattr(w, "text", None)})
                except Exception:
                    serial_words.append(None)
        else:
            serial_words = None

        segs.append({
            "start": getattr(segment, "start", None),
            "end": getattr(segment, "end", None),
            "text": segment.text,
            "words": serial_words,
        })

    return {"whisper_raw": whisper_raw, "segments": segs}


def append_to_output(output_file, record):
    with open(output_file, "a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")


def init_output_file(output_file):
    # Ensure parent dir exists
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Truncate/create a fresh file for this run
    with open(output_file, "w", encoding="utf-8") as out:
        pass


def main():
    parser = argparse.ArgumentParser(description="Whisper transcribe stage (JSONL output only)")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR, help="Directory with audio clips")
    parser.add_argument("--output-file", default=None, help="JSONL output file to write transcriptions to (default: <transcriptions_dir>/<audio_dir_basename>_transcriptions.jsonl)")

    # Minimal operational CLI (the Whisper settings are configured in the module constants)
    parser.add_argument("--extensions", nargs="*", default=[".wav"], help="Allowed audio extensions")
    parser.add_argument("--max-files", type=int, default=None, help="Limit to first N files (applies only to directory mode)")

    args = parser.parse_args()

    # Use configured constants for Whisper behavior (no CLI overrides)
    model_name = WHISPER_MODEL
    models_dir = WHISPER_MODELS_DIR
    device = WHISPER_DEVICE
    compute_type = WHISPER_COMPUTE_TYPE
    whisper_mode = WHISPER_MODE_DEFAULT
    language = WHISPER_LANGUAGE
    beam_size = WHISPER_BEAM_SIZE
    best_of = WHISPER_BEST_OF
    patience = WHISPER_PATIENCE
    temperature = WHISPER_TEMPERATURE
    condition_on_previous_text = WHISPER_CONDITION_ON_PREVIOUS_TEXT
    word_timestamps = WHISPER_WORD_TIMESTAMPS
    vad_filter = WHISPER_VAD_FILTER
    vad_min_silence_ms = WHISPER_VAD_MIN_SILENCE_DURATION_MS
    compression_ratio_threshold = WHISPER_COMPRESSION_RATIO_THRESHOLD
    log_prob_threshold = WHISPER_LOG_PROB_THRESHOLD
    no_speech_threshold = WHISPER_NO_SPEECH_THRESHOLD
    no_repeat_ngram_size = WHISPER_NO_REPEAT_NGRAM_SIZE_ACCURATE if whisper_mode == "accurate" else WHISPER_NO_REPEAT_NGRAM_SIZE_SAFE
    repetition_penalty = WHISPER_REPETITION_PENALTY_ACCURATE if whisper_mode == "accurate" else WHISPER_REPETITION_PENALTY_SAFE

    if not os.path.isdir(args.audio_dir):
        print(f"Audio directory not found: {args.audio_dir}")
        sys.exit(2)

    # Determine output file (unique per audio-dir by default)
    if args.output_file:
        output_file = args.output_file
    else:
        # Use transcriptions dir and audio-dir basename to create a per-dir output file
        audio_basename = os.path.basename(os.path.normpath(args.audio_dir)) or "audio"
        output_file = os.path.join(DEFAULT_TRANSCRIPT_DIR, f"{audio_basename}_transcriptions.jsonl")

    # Ensure output directory exists and initialize fresh file
    init_output_file(output_file)
    print(f"Writing transcriptions to {output_file} (fresh file for this run)", file=sys.stderr)

    # Ensure models directory exists so Whisper can download or store models there
    os.makedirs(models_dir, exist_ok=True)

    # Decide runtime device: prefer GPU if requested and available
    import importlib
    try:
        import torch
        torch_avail = torch.cuda.is_available()
    except Exception:
        torch_avail = False

    # Interpret WHISPER_DEVICE: if 'auto' use torch availability; if 'cuda' require torch_avail else fallback to cpu
    desired_device = device
    if desired_device == "auto":
        desired_device = "cuda" if torch_avail else "cpu"

    effective_device = desired_device
    if desired_device == "cuda" and not torch_avail:
        print("Warning: CUDA device requested but not available via torch.cuda.is_available(); falling back to CPU.", file=sys.stderr)
        effective_device = "cpu"

    print(f"Loading Whisper model '{model_name}' on device='{effective_device}' compute_type='{compute_type}' (models dir: {models_dir})", file=sys.stderr)
    model_kwargs = {"download_root": models_dir}

    try:
        model = WhisperModel(model_name, device=effective_device, compute_type=compute_type, **model_kwargs)
    except Exception as e:
        # Enhanced diagnostics on GPU failures (to stderr)
        err = str(e)
        print(f"Warning: failed to load model on device='{effective_device}': {err}", file=sys.stderr)
        # If we tried GPU, attempt to gather helpful diagnostics
        if effective_device == "cuda":
            print("Diagnostic: torch.cuda.is_available() =", torch_avail, file=sys.stderr)
            # check for nvidia-smi
            try:
                import shutil, subprocess
                nvid = shutil.which("nvidia-smi")
                if nvid:
                    out = subprocess.run([nvid, "-L"], capture_output=True, text=True, timeout=5)
                    print("nvidia-smi output:", out.stdout.strip(), file=sys.stderr)
                else:
                    print("nvidia-smi not found on PATH", file=sys.stderr)
            except Exception:
                pass

            # Detect common missing library hints
            if "libcublas" in err or "CUDA" in err or "cu" in err:
                print("It looks like CUDA libraries required by faster-whisper/ctranslate2 are missing or incompatible (example: libcublas.so.12).", file=sys.stderr)
                print("If you want to use GPU, ensure matching NVIDIA drivers and CUDA runtime are installed in the container/host. Example approaches:", file=sys.stderr)
                print("  - Install CUDA toolkit for your platform (e.g., CUDA 12.x) or the NVIDIA runtime packages.", file=sys.stderr)
                print("  - Ensure the container has access to the GPU (run with --gpus=all or use the NVIDIA Container Toolkit).", file=sys.stderr)
                print("  - If libraries are present but use a different SONAME, consider adding an appropriate LD_LIBRARY_PATH or symlink (advanced).", file=sys.stderr)

        # Fall back to CPU to allow progress
        if effective_device != "cpu":
            print("Falling back to CPU model to continue transcriptions.", file=sys.stderr)
            effective_device = "cpu"
            compute_type = "default"
            model = WhisperModel(model_name, device=effective_device, compute_type=compute_type, **model_kwargs)
        else:
            # If already on CPU and still failing, re-raise the error
            raise
    # update device variable to final effective device for bookkeeping
    device = effective_device

    files = sorted([f for f in os.listdir(args.audio_dir) if os.path.splitext(f)[1].lower() in args.extensions])
    if not files:
        print(f"No audio files with extensions {args.extensions} found in {args.audio_dir}", file=sys.stderr)
        sys.exit(0)
    # Optionally limit the number of files to process
    if args.max_files is not None:
        files = files[: args.max_files]


    from tqdm import tqdm

    total = len(files)

    # Build base transcribe options from configured constants
    transcribe_opts = {
        "beam_size": beam_size,
        "best_of": best_of,
        "patience": patience,
        "temperature": temperature,
        "compression_ratio_threshold": compression_ratio_threshold,
        "log_prob_threshold": log_prob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "initial_prompt": WHISPER_INITIAL_PROMPT,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
        "word_timestamps": word_timestamps,
        "vad_filter": vad_filter,
        "vad_parameters": {"min_silence_duration_ms": vad_min_silence_ms} if vad_filter else None,
    }

    transcribed = 0
    skipped = 0
    errors = 0

    with tqdm(total=total, unit="file", desc="Transcribing") as pbar:
        for idx, filename in enumerate(files, start=1):
            path = os.path.join(args.audio_dir, filename)
            clip_id = os.path.splitext(filename)[0]
            # Update progress bar with current clip id
            pbar.set_postfix_str(clip_id)


            try:
                # Use configured transcribe options
                result = transcribe_file(model, path, language=language, transcribe_opts=transcribe_opts)

                record = {
                    "clip_id": clip_id,
                    "file_path": os.path.abspath(path),
                    "whisper_raw": result["whisper_raw"],
                    "model": model_name,
                }
                append_to_output(output_file, record)
                # We intentionally DO NOT write per-clip .txt files here. Use the JSONL JSONL output for downstream steps.
                transcribed += 1
                pbar.update(1)
            except Exception as e:
                errors += 1
                # Log a simple error record so we know it failed for this clip
                err_record = {
                    "clip_id": clip_id,
                    "file_path": os.path.abspath(path),
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                append_to_output(output_file, err_record)
                tqdm.write(f"ERROR {idx}/{total}: {e}")
                pbar.update(1)

    print(f"Done. Transcribed: {transcribed}, Skipped: {skipped}, Errors: {errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
