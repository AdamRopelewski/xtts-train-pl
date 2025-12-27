#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Optional


def convert_line(entry: dict, keep_ext: bool, delim: str) -> Optional[str]:
    file_path = entry.get("file_path") or entry.get("file")
    if not file_path:
        return None

    name = os.path.basename(file_path)
    if not keep_ext:
        name = os.path.splitext(name)[0]

    variants = entry.get("variants", {})
    preserve = (variants.get("preserve") or "").strip()
    expanded = (variants.get("expanded") or "").strip()

    # If delimiter appears in text, replace it with a visually similar character to avoid column shifts.
    if delim in preserve or delim in expanded:
        preserve = preserve.replace(delim, "¦")
        expanded = expanded.replace(delim, "¦")

    return f"{name}{delim}{preserve}{delim}{expanded}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file (one JSON object per line)")
    parser.add_argument("--output", "-o", required=True, help="Output metadata CSV file")
    parser.add_argument("--delimiter", "-d", default="|", help="Field delimiter (default: '|')")
    parser.add_argument("--keep-ext", action="store_true", help="Keep file extension in the first column")
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    args = parser.parse_args()

    total = 0
    written = 0

    try:
        with open(args.input, "r", encoding=args.encoding) as fin, open(args.output, "w", encoding=args.encoding) as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: skipping invalid JSON on line {total}", file=sys.stderr)
                    continue

                out_line = convert_line(obj, args.keep_ext, args.delimiter)
                if out_line is None:
                    print(f"Warning: missing file path for JSON object on line {total}", file=sys.stderr)
                    continue

                fout.write(out_line + "\n")
                written += 1

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Processed {total} entries, wrote {written} lines to {args.output}")


if __name__ == "__main__":
    main()
