#!/usr/bin/env python3
"""Pi story reader: interactive loop for the viva demo.

Prompts for language and text, then speaks it in the caregiver voice via speak.py.

Usage:
    python story_reader.py [--fallback-url http://192.168.1.100:8000]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SPEAK = Path(__file__).parent / "speak.py"

LANG_LABELS = {
    "en": "English",
    "zh": "Mandarin",
    "yue": "Cantonese",
}


def select_language() -> str:
    print("\nSelect language:")
    langs = list(LANG_LABELS.items())
    for i, (code, label) in enumerate(langs, 1):
        print(f"  {i}. {label} ({code})")
    while True:
        try:
            choice = input("Enter number [1]: ").strip() or "1"
            idx = int(choice) - 1
            if 0 <= idx < len(langs):
                return langs[idx][0]
        except (ValueError, KeyboardInterrupt):
            pass
        print("Invalid choice, try again.")


def speak(text: str, lang: str, fallback_url: str | None) -> None:
    cmd = [sys.executable, str(SPEAK), "--text", text, "--lang", lang]
    if fallback_url:
        cmd += ["--fallback-url", fallback_url]
    subprocess.run(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive story reader for the Pi demo")
    parser.add_argument("--fallback-url", default=None,
                        help="Qwen3-TTS fallback server URL")
    parser.add_argument("--lang", default=None, choices=["en", "zh", "yue"],
                        help="Fix language (skips the language prompt)")
    args = parser.parse_args()

    print("=== Voice Pipeline — Story Reader ===")
    print("Type your story text and press Enter to hear it spoken.")
    print("Press Ctrl-C or type 'quit' to exit.\n")

    try:
        while True:
            lang = args.lang or select_language()
            print(f"\nLanguage: {LANG_LABELS[lang]}")
            text = input("Text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            speak(text, lang, args.fallback_url)
    except KeyboardInterrupt:
        print("\nBye.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
