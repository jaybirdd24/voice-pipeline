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
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

PIPER_SAMPLE_RATE = 22050
MIN_WORDS = 5
MAX_WORDS = 20  # lowered from 30: long utterances risk autoregressive collapse in Qwen3-TTS

# --- Quality-control gate for generated clips -------------------------------
# Qwen3-TTS is autoregressive: past ~10-12s in a single generation the acoustic
# token trajectory can drift out-of-distribution and collapse. Observed failure
# mode is a high-pitched "stuck" screech in the tail (the decoder loops on a
# token): the spectral centroid jumps up and frame-to-frame spectral flux drops.
# A clip with a garbage tail is poison for Piper training, so we reject it here.
#
# NOTE: the tail thresholds below were calibrated on a small number of real
# rejects. As rejects.csv accumulates, revisit them against that corpus.
MAX_DURATION_S = 11.0          # hard duration cap; reject anything longer
QC_FRAME_MS = 50               # analysis frame size
QC_TAIL_FRAC = 0.25            # "tail" = last 25% of the clip
QC_BODY_FRAC = 0.70            # "body" = first 70% of the clip
QC_ACTIVE_RMS_FRAC = 0.15      # a frame is "active" (non-silence) if rms > this * peak_rms
QC_CENTROID_RATIO = 1.5        # tail centroid this much higher than body => suspicious
QC_CENTROID_ABS_HZ = 1600.0    # ...and above this absolute pitch (avoids quiet-taper false positives)
QC_FLUX_RATIO = 0.75           # ...and tail spectrum this much more static (stuck) than body
QC_HARD_NOISE_FLATNESS = 0.5   # any active tail frame this flat => outright broadband noise


def _frame_stats(wav: np.ndarray, sr: int, frame_ms: int):
    """Per-frame RMS, spectral flatness, centroid (Hz), flux, and center times."""
    n = max(1, int(sr * frame_ms / 1000))
    total = len(wav) // n
    if total == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty
    frames = wav[: total * n].reshape(total, n).astype("float64")
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(frames * win, axis=1)) + 1e-10
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    power = mag ** 2
    geo = np.exp(np.mean(np.log(power), axis=1))
    flatness = geo / np.mean(power, axis=1)            # Wiener entropy: ~0 tonal, ~1 noise
    centroid = (mag * freqs).sum(axis=1) / mag.sum(axis=1)  # spectral centroid in Hz

    norm = mag / mag.sum(axis=1, keepdims=True)        # L1-normalized spectra
    flux = np.concatenate([[0.0], np.sqrt(((norm[1:] - norm[:-1]) ** 2).sum(axis=1))])

    centers = (np.arange(total) + 0.5) * n / sr
    return rms, flatness, centroid, flux, centers


def qc_check(wav: np.ndarray, sr: int, max_duration: float) -> tuple[bool, str]:
    """Return (ok, reason). Flags over-long clips and degenerate/screech tails."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    duration = len(wav) / sr
    if duration > max_duration:
        return False, f"too_long ({duration:.1f}s > {max_duration:.1f}s)"

    rms, flatness, centroid, flux, centers = _frame_stats(wav, sr, QC_FRAME_MS)
    if len(rms) < 6:
        return True, ""  # too short to analyze; duration cap already passed

    peak = float(rms.max())
    if peak <= 0:
        return False, "silent"
    active = rms > QC_ACTIVE_RMS_FRAC * peak
    tail_mask = active & (centers >= duration * (1 - QC_TAIL_FRAC))
    body_mask = active & (centers <= duration * QC_BODY_FRAC)

    if not tail_mask.any() or not body_mask.any():
        return True, ""  # tail is silence (clean taper) or no usable body — fine

    # Broadband-noise collapse: any active tail frame that is noise-flat.
    if float(flatness[tail_mask].max()) > QC_HARD_NOISE_FLATNESS:
        return False, f"noise_tail (flatness={flatness[tail_mask].max():.2f})"

    # Stuck-tone / screech collapse: tail pitch jumps up AND spectrum goes static.
    tail_cent = float(np.median(centroid[tail_mask]))
    body_cent = float(np.median(centroid[body_mask]))
    tail_flux = float(np.median(flux[tail_mask]))
    body_flux = float(np.median(flux[body_mask])) + 1e-9
    if (
        tail_cent > QC_CENTROID_RATIO * body_cent
        and tail_cent > QC_CENTROID_ABS_HZ
        and tail_flux < QC_FLUX_RATIO * body_flux
    ):
        return False, (
            f"screech_tail (centroid {body_cent:.0f}->{tail_cent:.0f}Hz, "
            f"flux ratio {tail_flux / body_flux:.2f})"
        )

    return True, ""


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_texts(
    corpus_path: Path, start_idx: int, count: int, max_words: int
) -> list[str]:
    rows = []
    with open(corpus_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 2:
                continue
            text = row[1].strip()
            if MIN_WORDS <= len(text.split()) <= max_words:
                # Ensure terminal punctuation so the model has a clean stop cue.
                if text and text[-1] not in ".?!":
                    text += "."
                rows.append(text)
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


@dataclass
class GenerationStats:
    generated: int = 0
    skipped: int = 0
    rejected: int = 0
    errors: int = 0


# A synthesizer takes a sentence and returns (waveform, sample_rate).
Synthesizer = Callable[[str], "tuple[np.ndarray, int]"]


def generate_dataset(
    texts: list[str],
    output_dir: Path,
    synthesize: Synthesizer,
    *,
    stem_prefix: str = "jay",
    start_idx: int = 0,
    max_duration: float = MAX_DURATION_S,
    qc: bool = True,
) -> GenerationStats:
    """Generate an LJSpeech-format dataset directory from texts.

    Writes wavs/<stem>.wav at 22050 Hz and appends `stem|text` lines to
    metadata.csv. QC-failing clips are quarantined to rejected/ and logged to
    rejects.csv; they never enter metadata.csv. Stems already present in
    metadata.csv (or on disk) are skipped, so an interrupted run resumes.
    """
    import soundfile as sf

    output_dir = Path(output_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = output_dir / "rejected"
    metadata_path = output_dir / "metadata.csv"
    rejects_path = output_dir / "rejects.csv"

    existing = load_existing_stems(metadata_path)
    if existing:
        print(f"Resuming: {len(existing)} clips already exist, will skip them")

    stats = GenerationStats()
    with open(metadata_path, "a", newline="", encoding="utf-8") as meta_f:
        for i, text in enumerate(texts):
            stem = f"{stem_prefix}_{start_idx + i + 1:04d}"
            wav_path = wavs_dir / f"{stem}.wav"

            if stem in existing or wav_path.exists():
                stats.skipped += 1
                continue

            print(f"  [{i + 1}/{len(texts)}] {stem}: {text[:60]}...")
            t0 = time.perf_counter()
            try:
                wav, sample_rate = synthesize(text)

                if wav is None or len(wav) == 0:
                    print("    WARN: empty output, skipping")
                    stats.errors += 1
                    continue

                if sample_rate != PIPER_SAMPLE_RATE:
                    wav = resample_to_piper(wav, sample_rate)

                # Quality-control gate: quarantine bad clips instead of training on them.
                ok, reason = (True, "") if not qc else qc_check(
                    np.asarray(wav), PIPER_SAMPLE_RATE, max_duration
                )
                if not ok:
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(str(rejected_dir / f"{stem}.wav"), wav, PIPER_SAMPLE_RATE)
                    write_header = not rejects_path.exists()
                    with open(rejects_path, "a", newline="", encoding="utf-8") as rej_f:
                        if write_header:
                            rej_f.write("stem|reason|text\n")
                        rej_f.write(f"{stem}|{reason}|{text}\n")
                    stats.rejected += 1
                    print(f"    QUARANTINED ({reason}) -> rejected/{stem}.wav")
                    continue

                sf.write(str(wav_path), wav, PIPER_SAMPLE_RATE)
                meta_f.write(f"{stem}|{text}\n")
                meta_f.flush()
                stats.generated += 1
                print(f"    -> {wav_path.name}  ({time.perf_counter() - t0:.1f}s)")

            except Exception as exc:
                print(f"    ERROR: {exc}")
                stats.errors += 1

    return stats


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
    parser.add_argument(
        "--max-words",
        type=int,
        default=MAX_WORDS,
        help=f"Skip corpus sentences longer than this (default {MAX_WORDS}).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_DURATION_S,
        help=f"Reject generated clips longer than this many seconds (default {MAX_DURATION_S}).",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Disable the quality-control gate (not recommended for training data).",
    )
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

    texts = load_texts(corpus_path, args.start_idx, args.count, args.max_words)
    if not texts:
        print("ERROR: no texts loaded (check corpus path and word-count filter)", file=sys.stderr)
        return 1
    print(f"Loaded {len(texts)} sentences from {corpus_path}")

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

    def synthesize(text: str):
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=qwen_language,
            ref_audio=str(reference_audio),
            ref_text=reference_transcript,
        )
        return (wavs[0] if wavs else None), sample_rate

    stats = generate_dataset(
        texts,
        output_dir,
        synthesize,
        start_idx=args.start_idx,
        max_duration=args.max_duration,
        qc=not args.no_qc,
    )

    print(
        f"\nDone.  Generated: {stats.generated}  Skipped: {stats.skipped}  "
        f"Rejected: {stats.rejected}  Errors: {stats.errors}"
    )
    print(f"Dataset: {output_dir}")
    if stats.rejected:
        print(f"Quarantined clips logged in: {output_dir / 'rejects.csv'}")
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
