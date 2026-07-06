#!/usr/bin/env python3
"""Stage 1: Generate a synthetic LJSpeech-format dataset using Qwen3-TTS voice cloning.

Usage:
    python generate_dataset.py --lang en [--count 500] [--start-idx 0]
    python generate_dataset.py --lang en --count 5   # smoke test

Output: datasets/<lang>_qwen3_synth/
    wavs/jay_0001.wav ...
    metadata.csv  (stem|text, Piper-compatible LJSpeech format at 22050 Hz)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from math import gcd
from pathlib import Path

import yaml

PIPER_SAMPLE_RATE = 22050
MIN_WORDS = 5
MAX_WORDS = 30


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_texts(corpus_path: Path, start_idx: int, count: int) -> list[tuple[str, str]]:
    rows = []
    with open(corpus_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 2:
                continue
            stem, text = row[0], row[1].strip()
            if MIN_WORDS <= len(text.split()) <= MAX_WORDS:
                rows.append((stem, text))
    return rows[start_idx : start_idx + count]


def load_existing_stems(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()
    stems = set()
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split("|")
            if parts:
                stems.add(parts[0].strip())
    return stems


def resample_to_piper(wav, orig_sr: int):
    """Resample numpy array from orig_sr to PIPER_SAMPLE_RATE using scipy."""
    if orig_sr == PIPER_SAMPLE_RATE:
        return wav
    from scipy.signal import resample_poly
    import numpy as np
    g = gcd(orig_sr, PIPER_SAMPLE_RATE)
    resampled = resample_poly(wav, PIPER_SAMPLE_RATE // g, orig_sr // g)
    return resampled.astype("float32")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-TTS synthetic dataset generator")
    parser.add_argument("--lang", required=True, choices=["en", "zh", "yue"])
    parser.add_argument("--count", type=int, default=500, help="Number of clips to generate")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index into text corpus")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    lang_cfg = cfg["languages"][args.lang]

    reference_audio = Path(lang_cfg["reference_audio"]).expanduser()
    reference_transcript = lang_cfg["reference_transcript"].strip()
    corpus_path = Path(lang_cfg["corpus"]).expanduser() if lang_cfg.get("corpus") else None
    qwen_language = lang_cfg["qwen_language"]
    model_name = cfg.get("qwen3_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    device = cfg.get("device", "cuda:0")

    output_dir = args.output_dir or (
        Path(cfg["datasets_dir"]).expanduser() / f"{args.lang}_qwen3_synth"
    )

    if not reference_audio.is_file():
        print(f"ERROR: reference audio not found: {reference_audio}", file=sys.stderr)
        print("Set reference_audio in config.yaml for this language.", file=sys.stderr)
        return 1

    if not reference_transcript:
        print("ERROR: reference_transcript is empty in config.yaml", file=sys.stderr)
        return 1

    if not corpus_path or not corpus_path.is_file():
        print(f"ERROR: text corpus not found: {corpus_path}", file=sys.stderr)
        print("Set corpus in config.yaml for this language.", file=sys.stderr)
        return 1

    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"

    texts = load_texts(corpus_path, args.start_idx, args.count)
    if not texts:
        print("ERROR: no texts loaded (check corpus path and word-count filter)", file=sys.stderr)
        return 1
    print(f"Loaded {len(texts)} sentences from {corpus_path}")

    existing = load_existing_stems(metadata_path)
    if existing:
        print(f"Resuming: {len(existing)} clips already exist, will skip them")

    import soundfile as sf
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        print(
            "ERROR: qwen-tts not installed.\n"
            "Activate the venv: source /home/jay/p4p/qwen3_tts_test/.venv/bin/activate\n"
            "Then: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {model_name} on {device}...")
    import torch
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"Model loaded. Generating {args.lang} clips -> {output_dir}")

    generated = skipped = errors = 0

    with open(metadata_path, "a", newline="", encoding="utf-8") as meta_f:
        for i, (_, text) in enumerate(texts):
            stem = f"jay_{args.start_idx + i + 1:04d}"
            wav_path = wavs_dir / f"{stem}.wav"

            if stem in existing or wav_path.exists():
                skipped += 1
                continue

            print(f"  [{i + 1}/{len(texts)}] {stem}: {text[:60]}...")
            t0 = time.perf_counter()
            try:
                wavs, sample_rate = model.generate_voice_clone(
                    text=text,
                    language=qwen_language,
                    ref_audio=str(reference_audio),
                    ref_text=reference_transcript,
                )

                if not wavs:
                    print("    WARN: empty output, skipping")
                    errors += 1
                    continue

                wav = wavs[0]
                if sample_rate != PIPER_SAMPLE_RATE:
                    wav = resample_to_piper(wav, sample_rate)

                sf.write(str(wav_path), wav, PIPER_SAMPLE_RATE)
                meta_f.write(f"{stem}|{text}\n")
                meta_f.flush()
                generated += 1
                print(f"    -> {wav_path.name}  ({time.perf_counter() - t0:.1f}s)")

            except Exception as exc:
                print(f"    ERROR: {exc}")
                errors += 1

    print(f"\nDone.  Generated: {generated}  Skipped: {skipped}  Errors: {errors}")
    print(f"Dataset: {output_dir}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
