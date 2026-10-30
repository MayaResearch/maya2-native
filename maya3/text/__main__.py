"""Preview Maya 2 Native's text normalization without loading the model."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys

from maya3.text import normalise_text

LANGUAGES = ("as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te")


def _normalize(text, lang):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    if not isinstance(lang, str) or lang not in LANGUAGES:
        raise ValueError("lang must be one of: " + ", ".join(LANGUAGES))
    # The vendored normalizer can print fallback diagnostics. Keep stdout usable
    # as plain text / JSONL; do not change the vendored files or their behavior.
    with redirect_stdout(sys.stderr):
        normalized = normalise_text(text, lang)
    return {"lang": lang, "text": text, "normalized": normalized}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=LANGUAGES, help="language; required outside JSONL mode")
    parser.add_argument("--text", help="input text (otherwise read standard input)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--json", action="store_true", help="emit one JSON object")
    mode.add_argument("--jsonl", action="store_true", help="read {text, lang} JSON objects, one per line")
    args = parser.parse_args(argv)
    if args.jsonl and args.text is not None:
        parser.error("--text cannot be combined with --jsonl")
    if not args.jsonl and args.lang is None:
        parser.error("--lang is required outside JSONL mode")
    if args.jsonl:
        for line_number, line in enumerate(sys.stdin, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
                result = _normalize(record.get("text"), record.get("lang", args.lang))
            except (ValueError, TypeError):
                # No raw input in diagnostics: it may include private text.
                print(f"Invalid JSONL record on line {line_number}: expected nonempty text and a supported lang",
                      file=sys.stderr)
                return 2
            print(json.dumps(result, ensure_ascii=False))
    else:
        try:
            result = _normalize(args.text if args.text is not None else sys.stdin.read(), args.lang)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False) if args.json else result["normalized"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
