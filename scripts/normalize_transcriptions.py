#!/usr/bin/env python3
"""Normalize Whisper "preserve" transcriptions to "expanded" using
a local ByT5 Polish text normalization model (e.g. Folx/byt5-small-pl-text-normalization).

This script requires that the model is present in a local directory (--model-dir) and
will refuse to use the network. It also optionally writes a CSV file in LJSpeech-style
format (default: metadata_t5.csv) with columns: <basename>|<preserve>|<expanded>.

Usage examples:
  pip install torch transformers tqdm
  python scripts/normalize_transcriptions.py \
    --input train-data/transcriptions/wavs_verbal_transcriptions.jsonl \
    --output train-data/transcriptions/wavs_verbal_transcriptions.normalized.jsonl \
    --model-dir ./byt5-small-pl-text-normalization \
    --csv-output train-data/transcriptions/metadata_t5.csv

By default this will only update entries where `variants.expanded` is empty or different
from the model output. Use --overwrite to force rewriting all entries.
"""

import argparse
import json
import os
import sys
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

DEFAULT_MODEL_DIR = "./byt5-small-pl-text-normalization"


import glob


def load_model(model_dir: str, device: str):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Look for common HF model weight files
    weight_patterns = ["pytorch_model*.bin", "*.safetensors", "tf_model.h5", "flax_model.msgpack"]
    weights_found = False
    for pattern in weight_patterns:
        if glob.glob(os.path.join(model_dir, pattern)):
            weights_found = True
            break
    if not weights_found and not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        raise FileNotFoundError(f"No model weights found in {model_dir}. Put the Hugging Face model files there.")

    # Verify tokenizer files are present (tokenizer is required locally)
    tokenizer_candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.txt",
        "merges.txt",
    ]
    tokenizer_found = False
    for fname in tokenizer_candidates:
        if os.path.exists(os.path.join(model_dir, fname)):
            tokenizer_found = True
            break

    if not tokenizer_found:
        raise FileNotFoundError(
            f"Tokenizer files not found in {model_dir}. Expected one of: {', '.join(tokenizer_candidates)}."
        )

    # Load tokenizer and model from local files only; provide clearer error if tokenizer loading fails
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, legacy=False, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load tokenizer from {model_dir}: {exc}")

    kwargs = {}
    if device.startswith("cuda"):
        # Use bfloat16 on CUDA to reduce memory and follow model recommendation
        kwargs["dtype"] = torch.bfloat16

    model = T5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True, **kwargs)

    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    model.eval()
    return tokenizer, model


class Normalizer:
    def __init__(self, tokenizer, model, device: str = "cpu", num_beams: int = 2, max_length: int = 400):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.num_beams = num_beams
        self.max_length = max_length

    def normalize(self, text: str) -> str:
        # Follow model's expected input format
        input_text = f"normalize: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        # move to right device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,
            )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return result


import re


def clean_text(s: Optional[str]) -> str:
    """Normalize text according to rules:
    - remove leading whitespace
    - strip leading quotes (", ', backtick, typographic quotes)
    - strip leading dashes or em-dashes
    - remove dashes after punctuation (e.g. ". -" -> ". ") by deleting the dash
    - remove any remaining dash-like sequences ("-", "—", "–") by replacing with ", "
    - collapse repeated punctuation/commas and whitespace
    """
    if not s:
        return ""
    s = s.strip()

    # remove leading quote-like characters and any following whitespace
    while s and s[0] in '"\'“”`':
        s = s[1:].lstrip()

    # remove leading dash-like characters and any following whitespace
    while s and s[0] in "-—–":
        s = s[1:].lstrip()

    # Replace punctuation followed by dash (with optional spaces) by the punctuation and a single space (drop the dash)
    s = re.sub(r'([\?\.!,:;])\s*[-—–]+\s*', r'\1 ', s)

    # Replace any remaining dash-like sequences with ", " (explicit pass to be safe)
    s = re.sub(r'[-—–]+', ', ', s)

    # Collapse multiple commas/spaces to single ", "
    s = re.sub(r'(,\s*){2,}', ', ', s)

    # Trim leading/trailing commas/spaces
    s = s.strip(" ,")

    # Collapse repeated whitespace
    s = " ".join(s.split())

    return s
    # Collapse multiple commas/spaces to single ", "
    s = re.sub(r'(,\s*){2,}', ', ', s)

    # Trim leading/trailing commas/spaces
    s = s.strip(" ,")

    # Collapse repeated whitespace
    s = " ".join(s.split())

    return s


def process_file(input_path: str, output_path: str, normalizer: Normalizer, overwrite: bool = False, dry_run: bool = False, csv_output: Optional[str] = None, csv_delim: str = "|", keep_ext: bool = False):
    total = 0
    changed = 0

    csv_fout = None
    if csv_output:
        csv_fout = open(csv_output, "w", encoding="utf-8")


    # Count lines up-front to provide an accurate tqdm total
    try:
        with open(input_path, "r", encoding="utf-8") as _countf:
            total_lines = sum(1 for _ in _countf)
    except FileNotFoundError:
        raise

    try:
        with open(input_path, "r", encoding="utf-8") as inf, open(output_path, "w", encoding="utf-8") as outf:

            for line in tqdm(inf, total=total_lines):
                total += 1
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: skipping invalid JSON line {total}", file=sys.stderr)
                    continue

                # Find preserve text. Support both top-level `preserve` or `variants.preserver` structure
                preserve_text = None
                if isinstance(obj.get("variants"), dict):
                    preserve_text = obj["variants"].get("preserve")
                    existing_expanded = obj["variants"].get("expanded")
                else:
                    preserve_text = obj.get("preserve")
                    existing_expanded = obj.get("expanded")

                if not preserve_text or not preserve_text.strip():
                    # nothing to normalize; still write object and CSV entry if requested
                    outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    if csv_fout:
                        file_path = obj.get("file_path") or obj.get("file") or ""
                        name = os.path.basename(file_path)
                        if not keep_ext:
                            name = os.path.splitext(name)[0]
                        preserve = ""
                        expanded = ""
                        csv_fout.write(f"{name}{csv_delim}{preserve}{csv_delim}{expanded}\n")
                    continue

                normalized = existing_expanded
                should_call = overwrite or (not existing_expanded) or (existing_expanded.strip() == "")

                if should_call:
                    # Clean input before feeding it to the model
                    preserve_for_model = clean_text(preserve_text)
                    normalized = normalizer.normalize(preserve_for_model)

                # Clean normalized text (remove leading whitespace/dashes, replace internal dashes)
                normalized = clean_text(normalized)

                # If dry run, just print the first few diffs
                if dry_run and total <= 5:
                    print("---\nPreserve:\n", preserve_text)
                    print("Normalized:\n", normalized)

                # Update object
                if isinstance(obj.get("variants"), dict):
                    if obj["variants"].get("expanded") != normalized:
                        changed += 1
                    obj["variants"]["expanded"] = normalized
                else:
                    if obj.get("expanded") != normalized:
                        changed += 1
                    obj["expanded"] = normalized

                outf.write(json.dumps(obj, ensure_ascii=False) + "\n")

                # write CSV line if requested
                if csv_fout:
                    file_path = obj.get("file_path") or obj.get("file") or ""
                    if not file_path:
                        print(f"Warning: missing file path for JSON object on line {total}", file=sys.stderr)
                        name = ""
                    else:
                        name = os.path.basename(file_path)
                        if not keep_ext:
                            name = os.path.splitext(name)[0]

                    preserve = (preserve_text or "").strip()
                    # Replace delimiter in text to avoid column shift
                    if csv_delim in preserve or csv_delim in normalized:
                        preserve = preserve.replace(csv_delim, "¦")
                        norm_for_csv = normalized.replace(csv_delim, "¦")
                    else:
                        norm_for_csv = normalized

                    csv_fout.write(f"{name}{csv_delim}{preserve}{csv_delim}{norm_for_csv}\n")

    finally:
        if csv_fout:
            csv_fout.close()

    return total, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file with preserve variants")
    parser.add_argument("--output", "-o", required=False, help="Output JSONL file (default: input.normalized.jsonl)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Local model directory (must exist locally)")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Device to run inference on")
    parser.add_argument("--num-beams", type=int, default=2, help="Beam size for generation (1-2 recommended)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing expanded variants")
    parser.add_argument("--dry-run", action="store_true", help="Show a few example conversions and exit")
    parser.add_argument("--csv-output", default="metadata_t5.csv", help="Output CSV metadata file in LJSpeech-style format (default: metadata_t5.csv)")
    parser.add_argument("--csv-delim", default="|", help="CSV field delimiter (default: '|')")
    parser.add_argument("--keep-ext", action="store_true", help="Keep file extension in the first column of CSV")
    args = parser.parse_args()

    out_path = args.output or (args.input + ".normalized.jsonl")

    print(f"Loading model from {args.model_dir} on {args.device} (local only)...")
    tokenizer, model = load_model(args.model_dir, args.device)
    normalizer = Normalizer(tokenizer, model, device=args.device, num_beams=args.num_beams)

    print(f"Processing {args.input} -> {out_path} and CSV -> {args.csv_output}")
    total, changed = process_file(
        args.input,
        out_path,
        normalizer,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        csv_output=args.csv_output,
        csv_delim=args.csv_delim,
        keep_ext=args.keep_ext,
    )

    print(f"Done. Processed {total} lines — updated {changed} entries. CSV written to {args.csv_output}")


if __name__ == "__main__":
    main()
