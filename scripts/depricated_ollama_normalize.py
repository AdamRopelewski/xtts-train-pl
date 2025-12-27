#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
from typing import List, Dict, Any

import requests
from difflib import SequenceMatcher


OLLAMA_BASE_URL = "http://host.docker.internal:11434"
OLLAMA_MODEL = "qwen2.5:7b"  # Default model for analysis (use host.docker.internal when running in containers)
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_prompt(items: List[Dict[str, Any]]) -> str:
    examples = (
        'Example: "1469 and 1470" -> "fourteen sixty-nine and fourteen seventy"',
        'Example: "It\'ll be $16 sir." -> "It will be sixteen dollars sir."',
        'Example: "audio1: I\'m" -> "I am"'
    )

    prompt_lines = [
        "You are a helpful assistant that normalizes short ASR transcriptions.",
        "Rules:",
        "- Return a JSON array of objects [{\"clip_id\": ..., \"normalized\": ...}] and nothing else.",
        "- Replace all digits with spelled-out words (no digits anywhere).",
        '- Expand contractions and abbreviations (e.g., "I\'m" -> "I am", "won\'t" -> "will not").',
        "- Expand currency symbols to words (e.g., \"$16\" -> \"sixteen dollars\").",
        "- Keep punctuation where it helps readability but do not use numerals or short forms.",
        "- Preserve the meaning of the original as closely as possible. If you are uncertain, prefer a literal expansion.",
        "Examples:",
    ]
    prompt_lines += ["- " + ex for ex in examples]
    prompt_lines.append("")
    prompt_lines.append("Now normalize the following transcriptions (return JSON array):")
    prompt_lines.append("")

    for it in items:
        clip = it.get("clip_id") or it.get("clip") or it.get("id")
        text = (it.get("whisper_raw") or it.get("text") or "").strip()
        prompt_lines.append(f"- {clip}: {text}")

    prompt = "\n".join(prompt_lines)
    return prompt


def send_to_ollama(ollama_url: str, model: str, prompt: str, timeout: int = 60) -> str:

    base = ollama_url.rstrip("/")
    url = base + "/api/generate"
    payload = {"model": model, "prompt": prompt, "temperature": 0}

    def _call(url_to_call: str):
        r = requests.post(url_to_call, json=payload, timeout=timeout)
        r.raise_for_status()
        return r

    try:
        r = _call(url)
    except requests.exceptions.ConnectionError as e:
        # If user runs inside Docker/devcontainer, localhost on host may be unreachable; try host.docker.internal
        if "localhost" in base or "127.0.0.1" in base:
            fallback_base = base.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
            fallback_url = fallback_base + "/api/generate"
            print(f"Connection to {base} refused; trying fallback {fallback_base} (or set --ollama-url to your host address).")
            try:
                r = _call(fallback_url)
            except Exception:
                # Re-raise the original connection error for visibility
                raise
        else:
            raise

    # Try to extract generated text.
    try:
        data = r.json()
    except Exception:
        return r.text

    # Ollama might return different shapes; try common ones
    if isinstance(data, dict):
        # Try common keys
        for key in ("text", "result", "output", "generation"):
            if key in data:
                return data[key]
        # Some versions return {'id':..., 'created':..., 'model':..., 'params':..., 'choices':[...]} 
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            ch = data["choices"][0]
            if isinstance(ch, dict) and "text" in ch:
                return ch["text"]
            if isinstance(ch, dict) and "message" in ch:
                return ch["message"].get("content", "")
    # Fallback: return raw text
    return r.text


def extract_json_from_response(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()

    # 1) Direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # 2) Try to find a bracketed JSON array in the whole text
    m = re.search(r"(\[.*\])", text, re.S)
    if m:
        candidate = m.group(1)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
        except Exception:
            cleaned = re.sub(r",\s*\]", "]", candidate)
            cleaned = re.sub(r",\s*\}", "}", cleaned)
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, list):
                    return obj
            except Exception:
                pass

    # 3) Ollama streaming can wrap the real content in repeated JSON objects with a "response" field.
    resp_pieces = re.findall(r'"response"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if resp_pieces:
        try:
            # Use JSON loads to unescape each piece reliably
            unescaped = "".join([json.loads('"' + p.replace('"', '\\"') + '"') for p in resp_pieces])
            # try parse the joined string as JSON
            obj = json.loads(unescaped)
            if isinstance(obj, list):
                return obj
        except Exception:
            first = unescaped.find("[") if 'unescaped' in locals() else -1
            last = unescaped.rfind("]") if 'unescaped' in locals() else -1
            if first != -1 and last != -1 and last > first:
                sub = unescaped[first : last + 1]
                try:
                    obj = json.loads(sub)
                    if isinstance(obj, list):
                        return obj
                except Exception:
                    pass

    # If nothing parsed, return empty list
    return []


def _unescape_json_string(s: str) -> str:
    """Robustly convert a JSON-escaped string (e.g., "\u0142") into a Python str.
    If s is already normal, it's returned unchanged.
    """
    if not isinstance(s, str):
        return s
    s = s.strip()
    try:
        # Wrap in quotes and let json.loads handle escape sequences
        return json.loads('"' + s.replace('"', '\\"') + '"')
    except Exception:
        # Last resort: decode unicode escapes
        try:
            return s.encode('utf-8').decode('unicode_escape')
        except Exception:
            return s


def extract_normalized_from_raw(text: str, clip_id: str | None) -> str | None:
    """Attempt to extract the normalized string for a given clip_id from raw Ollama text.
    Returns the extracted string (unescaped) or None if not found.
    Uses multiple heuristics to find the full sentence-like normalized value.
    """
    if not text:
        return None

    # 1) If we can parse a JSON array, use that result directly
    parsed = extract_json_from_response(text)
    if parsed:
        for item in parsed:
            if str(item.get('clip_id')) == str(clip_id):
                val = item.get('normalized')
                return _unescape_json_string(val) if isinstance(val, str) else val

    # Helper to join/clean streaming response pieces robustly
    def _join_pieces(pieces):
        out_parts = []
        for p in pieces:
            try:
                out_parts.append(json.loads('"' + p.replace('"', '\\"') + '"'))
            except Exception:
                try:
                    out_parts.append(p.encode('utf-8').decode('unicode_escape'))
                except Exception:
                    out_parts.append(p)
        return "".join(out_parts)

    # 2) Collect "response" pieces if present and build joined string
    resp_pieces = re.findall(r'"response"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    joined = None
    if resp_pieces:
        joined = _join_pieces(resp_pieces)

    # If we have a joined blob, try to find a pattern like '- clip_id: normalized' or 'clip_id: normalized' on its own line
    if clip_id and joined:
        patterns = [
            r'(?:^|\n)[\-\*\s]*' + re.escape(str(clip_id)) + r'\s*[:\-]\s*(.+?)(?:\n|$)',
            r'(?:^|\n)' + re.escape(str(clip_id)) + r'\s+(.+?)(?:\n|$)',
            r'"clip_id"\s*:\s*"' + re.escape(str(clip_id)) + r'".*?"normalized"\s*:\s*"((?:\\.|[^"\\])*)"',
        ]
        for pat in patterns:
            m = re.search(pat, joined, re.S)
            if m:
                candidate = m.group(1).strip()
                # unescape any json-style escapes
                return _unescape_json_string(candidate)

    # 3) If no clip-specific match, try to find any normalized field in joined or raw text
    for blob in (joined, text):
        if not blob:
            continue
        m2 = re.search(r'"normalized"\s*:\s*"((?:\\.|[^"\\])*)"', blob, re.S)
        if m2:
            return _unescape_json_string(m2.group(1))

    # 4) Try line-based heuristics: find line that looks sentence-like (no JSON braces, not too short)
    candidate_lines = []
    source_lines = (joined or text).splitlines() if (joined or text) else []
    for ln in source_lines:
        ln_stripped = ln.strip()
        if not ln_stripped:
            continue
        # skip lines that look like JSON metadata
        if ln_stripped.startswith('{') or ln_stripped.startswith('['):
            continue
        # skip lines that look like 'clip_id: ...' without normalized
        if re.search(r'clip_id\s*:', ln_stripped):
            # try to extract trailing text after colon
            parts = re.split(r'[:\-]', ln_stripped, maxsplit=1)
            if len(parts) > 1 and len(parts[1].strip()) > 2:
                candidate_lines.append(parts[1].strip())
            continue
        # otherwise consider as candidate sentence
        if len(ln_stripped) > 10:
            candidate_lines.append(ln_stripped)

    if candidate_lines:
        # prefer the longest candidate as most likely full sentence
        best = max(candidate_lines, key=len)
        return _unescape_json_string(best)

    return None


def similarity(a: str, b: str) -> float:
    a2 = re.sub(r"\s+", " ", a.strip().lower())
    b2 = re.sub(r"\s+", " ", b.strip().lower())
    return SequenceMatcher(None, a2, b2).ratio()


def count_digits(s: str) -> int:
    return len(re.findall(r"\d", s))


def process(input_path: str, output_csv: str, ollama_url: str, model: str, batch_size: int, min_similarity: float, limit: int | None = None, simple: bool = True):
    entries = list(read_jsonl(input_path))
    if limit:
        entries = entries[:limit]
    print(f"Read {len(entries)} items from {input_path} (limit={limit})")

    # create batches
    batches = [entries[i : i + batch_size] for i in range(0, len(entries), batch_size)]
    print(f"Processing {len(batches)} batches (batch_size={batch_size})")

    # Prepare output file (truncate / write header if necessary)
    fieldnames = ["clip_id", "wav_path", "whisper_raw", "normalized", "similarity", "flagged"]
    if simple:
        # truncate file so we start fresh
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            pass
    else:
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Use a small thread pool to prefetch the next batch while processing current
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=1)
    future = None

    if batches:
        print(f"Submitting request for batch 0")
        future = executor.submit(send_to_ollama, ollama_url, model, make_prompt(batches[0]))

    for i, batch in enumerate(batches):
        try:
            resp_text = future.result() if future is not None else ""
        except Exception as e:
            print(f"Error retrieving response for batch {i}: {e}")
            resp_text = ""

        # Submit the next batch request immediately, so the model can start preparing it
        if i + 1 < len(batches):
            print(f"Received response for batch {i}; submitting request for batch {i+1}")
            future = executor.submit(send_to_ollama, ollama_url, model, make_prompt(batches[i + 1]))
        else:
            future = None

        # Process the response for this batch and append to file immediately
        parsed = extract_json_from_response(resp_text)
        rows_for_batch = []

        if not parsed:
            print("Warning: failed to parse JSON from model response; attempting to extract normalized strings from raw response for each item in batch.")
            for it in batch:
                clip_id = it.get("clip_id")
                whisper_raw = (it.get("whisper_raw") or it.get("text") or "").strip()
                extracted = extract_normalized_from_raw(resp_text, clip_id)
                normalized_val = extracted if extracted is not None else resp_text.strip()
                normalized_val = str(normalized_val).replace("\n", " ").replace("|", "/").strip()
                sim = similarity(whisper_raw, normalized_val)
                flagged = count_digits(whisper_raw) >= 2 and sim < min_similarity
                if count_digits(normalized_val) > 0:
                    flagged = True
                rows_for_batch.append({
                    "clip_id": clip_id,
                    "wav_path": it.get("file_path"),
                    "whisper_raw": whisper_raw,
                    "normalized": normalized_val,
                    "similarity": sim,
                    "flagged": flagged,
                })
        else:
            normalized_map = {str(item.get("clip_id")): item.get("normalized") for item in parsed if item.get("clip_id")}
            for it in batch:
                clip = str(it.get("clip_id"))
                whisper_raw = (it.get("whisper_raw") or it.get("text") or "").strip()
                normalized = normalized_map.get(clip)
                if normalized is None:
                    # try by substring match: model might have included index or different format
                    normalized = ""
                    for item in parsed:
                        if item.get("clip_id") and str(item.get("clip_id")).strip() == clip:
                            normalized = item.get("normalized") or ""
                            break
                if normalized is None:
                    normalized = ""

                # ensure proper unescaping of JSON-escaped responses (preserve polish / unicode)
                if isinstance(normalized, str):
                    normalized = _unescape_json_string(normalized)

                sim = similarity(whisper_raw, normalized)
                flagged = False
                if count_digits(whisper_raw) >= 2 and sim < min_similarity:
                    flagged = True
                if count_digits(str(normalized)) > 0:
                    flagged = True

                rows_for_batch.append({
                    "clip_id": clip,
                    "wav_path": it.get("file_path"),
                    "whisper_raw": whisper_raw,
                    "normalized": normalized.strip() if isinstance(normalized, str) else json.dumps(normalized, ensure_ascii=False),
                    "similarity": sim,
                    "flagged": flagged,
                })

        # Append to output file immediately
        if simple:
            with open(output_csv, "a", encoding="utf-8", newline="") as f:
                for r in rows_for_batch:
                    wav = os.path.basename(r.get("wav_path") or "") or f"{r.get('clip_id')}.wav"
                    whisper = (r.get("whisper_raw") or "").replace("\n", " ").replace("|", "/")
                    normalized = (r.get("normalized") or "").replace("\n", " ").replace("|", "/")
                    f.write(f"{wav}|{whisper}|{normalized}\n")
        else:
            with open(output_csv, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                for r in rows_for_batch:
                    writer.writerow([r.get("clip_id"), r.get("wav_path"), r.get("whisper_raw"), r.get("normalized"), f"{r.get('similarity'):.4f}", "1" if r.get("flagged") else "0"])

        print(f"Appended {len(rows_for_batch)} rows for batch {i}")

    executor.shutdown(wait=True)
    print(f"Processing complete; appended output to {output_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="train-data/transcriptions/wave_transcriptions.jsonl")
    p.add_argument("--output", default="metadata.csv")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--min-similarity", type=float, default=0.65, help="Threshold to flag low similarity when many digits present")
    p.add_argument("--limit", type=int, default=None, help="Process only first N items")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        print("Dry run: will not call Ollama. Instead will print the first prompt and exit.")
        # print prompt for first batch (respect limit when present)
        entries = list(read_jsonl(args.input))
        if not entries:
            print("No entries found")
            sys.exit(1)
        if args.limit:
            entries = entries[: args.limit]
        print(make_prompt(entries[: args.batch_size]))
        sys.exit(0)

    # simple output is the default (pipe-separated lines: wav_filename|whisper_raw|normalized)
    process(args.input, args.output, args.ollama_url, args.model, args.batch_size, args.min_similarity, args.limit)


if __name__ == "__main__":
    main()
