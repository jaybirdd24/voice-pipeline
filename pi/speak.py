#!/usr/bin/env python3
"""Pi TTS client: synthesise text and play it.

Tries local Piper ONNX first; falls back to the remote Qwen3-TTS server.

Usage:
    python speak.py --text "Hello world" --lang en
    python speak.py --text "你好" --lang zh --fallback-url http://192.168.1.100:8000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# Default Piper ONNX bundle locations on the Pi (copy here after export_piper.sh)
BUNDLE_DIR = Path(__file__).parent / "bundles"

LANG_VOICE = {
    "en": "en-qwen3-synth",
    "zh": "zh-qwen3-synth",
    "yue": "yue-qwen3-synth",
}


def find_bundle(lang: str) -> tuple[Path, Path] | None:
    voice_name = LANG_VOICE.get(lang)
    if not voice_name:
        return None
    onnx = BUNDLE_DIR / f"{voice_name}.onnx"
    config = BUNDLE_DIR / f"{voice_name}.onnx.json"
    if onnx.is_file() and config.is_file():
        return onnx, config
    return None


def speak_piper(text: str, onnx: Path, config: Path) -> bool:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["piper", "--model", str(onnx), "--config", str(config),
             "--output_file", tmp_path],
            input=text.encode(),
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"[piper] error: {result.stderr.decode()}", file=sys.stderr)
            return False
        subprocess.run(["aplay", "-q", tmp_path], check=True)
        Path(tmp_path).unlink(missing_ok=True)
        return True
    except FileNotFoundError as exc:
        print(f"[piper] not found: {exc}", file=sys.stderr)
        return False


def speak_fallback(text: str, lang: str, fallback_url: str) -> bool:
    try:
        import urllib.request
        import json

        body = json.dumps({"text": text, "lang": lang}).encode("utf-8")
        req = urllib.request.Request(
            f"{fallback_url.rstrip('/')}/speak",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            wav_bytes = resp.read()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        subprocess.run(["aplay", "-q", tmp_path], check=True)
        Path(tmp_path).unlink(missing_ok=True)
        return True
    except Exception as exc:
        print(f"[fallback] error: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Pi TTS: speak text in the caregiver voice")
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="en", choices=["en", "zh", "yue"])
    parser.add_argument("--fallback-url", default=None,
                        help="Qwen3-TTS server URL, e.g. http://192.168.1.100:8000")
    parser.add_argument("--force-fallback", action="store_true",
                        help="Skip Piper and go straight to fallback server")
    args = parser.parse_args()

    if not args.force_fallback:
        bundle = find_bundle(args.lang)
        if bundle:
            onnx, config = bundle
            print(f"[piper] {onnx.name}", file=sys.stderr)
            if speak_piper(args.text, onnx, config):
                return 0
            print("[piper] failed, trying fallback...", file=sys.stderr)

    if args.fallback_url:
        print(f"[fallback] {args.fallback_url}", file=sys.stderr)
        if speak_fallback(args.text, args.lang, args.fallback_url):
            return 0
        print("[fallback] failed.", file=sys.stderr)
        return 1

    print(
        "ERROR: no Piper bundle and no --fallback-url provided.\n"
        "Copy ONNX bundles to pi/bundles/ or pass --fallback-url.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
