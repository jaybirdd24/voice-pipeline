#!/usr/bin/env python3
"""Validate an LJSpeech-format dataset directory before Piper training.

Usage:
    python3 validate_dataset.py --lang en [--config config.yaml] [--min-clips 8]
    python3 validate_dataset.py --dataset-dir /path/to/dataset [--min-clips 8]

Checks that metadata.csv and wavs/ agree, that no quarantined clip leaked into
metadata, and that every WAV is 22050 Hz mono 16-bit PCM with audio in it.
Header-only reads via the stdlib wave module, so validating hundreds of clips
takes well under a second and works in any venv. Exit 0 = trainable dataset.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

EXPECTED_SAMPLE_RATE = 22050
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2  # bytes, i.e. 16-bit PCM


def validate(dataset_dir: Path, min_clips: int) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    metadata_path = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"
    rejected_dir = dataset_dir / "rejected"

    if not metadata_path.is_file():
        print(f"FAIL: metadata.csv not found: {metadata_path}", file=sys.stderr)
        return 1
    if not wavs_dir.is_dir():
        print(f"FAIL: wavs/ dir not found: {wavs_dir}", file=sys.stderr)
        return 1

    stems: list[str] = []
    seen: set[str] = set()
    for lineno, line in enumerate(
        metadata_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        stem, sep, text = line.partition("|")
        if not sep or not stem.strip() or not text.strip():
            errors.append(f"metadata.csv:{lineno}: malformed line (expected stem|text): {line!r}")
            continue
        stem = stem.strip()
        if stem in seen:
            errors.append(f"metadata.csv:{lineno}: duplicate stem {stem}")
            continue
        seen.add(stem)
        stems.append(stem)

    if len(stems) < min_clips:
        errors.append(f"only {len(stems)} clips in metadata.csv; need at least {min_clips}")

    total_seconds = 0.0
    for stem in stems:
        wav_path = wavs_dir / f"{stem}.wav"
        if not wav_path.is_file():
            errors.append(f"{stem}: no wav file at wavs/{stem}.wav")
            continue
        if (rejected_dir / f"{stem}.wav").exists():
            errors.append(f"{stem}: appears in BOTH metadata.csv and rejected/ (quarantine leak)")
        try:
            with wave.open(str(wav_path), "rb") as w:
                rate = w.getframerate()
                channels = w.getnchannels()
                width = w.getsampwidth()
                frames = w.getnframes()
        except (wave.Error, EOFError) as exc:
            errors.append(f"{stem}: unreadable wav ({exc})")
            continue
        if rate != EXPECTED_SAMPLE_RATE:
            errors.append(f"{stem}: sample rate {rate}, expected {EXPECTED_SAMPLE_RATE}")
        if channels != EXPECTED_CHANNELS:
            errors.append(f"{stem}: {channels} channels, expected mono")
        if width != EXPECTED_SAMPLE_WIDTH:
            errors.append(f"{stem}: sample width {width * 8}-bit, expected 16-bit PCM")
        if frames == 0:
            errors.append(f"{stem}: wav contains no audio frames")
        else:
            total_seconds += frames / rate

    orphans = sorted(p.stem for p in wavs_dir.glob("*.wav") if p.stem not in seen)
    if orphans:
        warnings.append(
            f"{len(orphans)} wav(s) in wavs/ not referenced by metadata.csv "
            f"(e.g. {', '.join(orphans[:3])}) — ignored by training"
        )

    for warning in warnings:
        print(f"WARN: {warning}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"\nDataset INVALID: {len(errors)} error(s) in {dataset_dir}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(stems)} clips, {total_seconds / 60:.1f} min total, "
        f"{EXPECTED_SAMPLE_RATE} Hz mono 16-bit — {dataset_dir}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Piper training dataset")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--lang", choices=["en", "zh", "yue"])
    target.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "config.yaml"
    )
    parser.add_argument(
        "--min-clips", type=int, default=8,
        help="Fail if fewer clips than this (default 8, one training batch)",
    )
    args = parser.parse_args()

    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        import yaml

        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        dataset_dir = Path(cfg["datasets_dir"]).expanduser() / f"{args.lang}_qwen3_synth"

    return validate(dataset_dir, args.min_clips)


if __name__ == "__main__":
    raise SystemExit(main())
