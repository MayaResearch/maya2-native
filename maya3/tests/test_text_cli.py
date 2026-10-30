"""Exercise the public CLI in a subprocess without site-packages or model dependencies."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run(*args, text=None):
    return subprocess.run([sys.executable, "-S", "-m", "maya3.text", *args], input=text,
                          capture_output=True, text=True, cwd=ROOT, timeout=15)


def test_single_text_and_json_preserve_unicode_and_tags():
    result = run("--lang", "hi", "--text", "मेरे पास 25 रुपये हैं [laughs]", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "lang": "hi", "text": "मेरे पास 25 रुपये हैं [laughs]",
        "normalized": "मेरे पास पच्चीस रुपये हैं [laughs]",
    }


@pytest.mark.parametrize("lang", ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"])
def test_stdin_works_in_all_languages_without_site_packages(lang):
    result = run("--lang", lang, text="25 [laughs]")
    assert result.returncode == 0, result.stderr
    assert "25" not in result.stdout
    assert result.stdout.strip().endswith("[laughs]")


def test_mixed_language_jsonl_and_default_language():
    result = run("--jsonl", "--lang", "en", text='{"text":"25"}\n\n{"text":"25","lang":"hi"}\n')
    assert result.returncode == 0, result.stderr
    assert [json.loads(line)["normalized"] for line in result.stdout.splitlines()] == ["twenty five", "पच्चीस"]


@pytest.mark.parametrize("record", ['invalid', '[]', '{"text":25,"lang":"en"}',
                                    '{"text":"private input","lang":"fr"}',
                                    '{"text":"","lang":"en"}', '{"text":"25"}'])
def test_bad_jsonl_fails_without_echoing_input(record):
    result = run("--jsonl", text='{"text":"25","lang":"en"}\n' + record)
    assert result.returncode == 2
    assert len(result.stdout.splitlines()) == 1  # Streaming output before failure is retained.
    assert "line 2" in result.stderr and "private input" not in result.stderr


@pytest.mark.parametrize("args,text", [([], "25"), (["--lang", "fr"], "25"),
                                      (["--lang", "en"], "  "),
                                      (["--text", "25", "--jsonl"], None)])
def test_invalid_arguments_and_empty_input(args, text):
    result = run(*args, text=text)
    assert result.returncode == 2
    assert result.stdout == ""


def test_fallback_diagnostics_do_not_corrupt_json():
    result = run("--lang", "en", "--text", "31/02/2025", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["text"] == "31/02/2025"
    assert "normalization_fallback" in result.stderr
