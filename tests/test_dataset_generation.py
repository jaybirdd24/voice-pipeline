"""Seam 1 tests: the dataset directory produced by Stage 1 generation.

The Qwen3-TTS synthesizer is a system boundary and is stubbed here; the real
model is exercised only by an explicit smoke run (see issue #2). Everything is
asserted through the public interface: the files a run leaves on disk and the
stats it reports.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from generate_dataset import PIPER_SAMPLE_RATE, generate_dataset


def speech_like(duration_s: float, sr: int = 24000, freq: float = 300.0) -> np.ndarray:
    """A steady low tone: passes the QC gate (short, low centroid, no tail jump)."""
    t = np.arange(int(duration_s * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32")


def test_run_produces_valid_ljspeech_dataset(tmp_path):
    texts = ["The cat sat on the mat.", "A dog barked at the moon."]

    def synth(text):
        return speech_like(1.5), 24000

    stats = generate_dataset(texts, tmp_path, synth)

    meta_lines = (tmp_path / "metadata.csv").read_text(encoding="utf-8").strip().splitlines()
    assert meta_lines == [
        "jay_0001|The cat sat on the mat.",
        "jay_0002|A dog barked at the moon.",
    ]
    for stem in ("jay_0001", "jay_0002"):
        wav, sr = sf.read(tmp_path / "wavs" / f"{stem}.wav")
        assert sr == PIPER_SAMPLE_RATE
        assert len(wav) > 0
    assert stats.generated == 2
    assert stats.rejected == 0
    assert stats.errors == 0


def test_overlong_clip_is_quarantined_and_kept_out_of_metadata(tmp_path):
    texts = ["A short and fine sentence here.", "This one collapses into rambling."]

    def synth(text):
        duration = 12.0 if "collapses" in text else 1.5
        return speech_like(duration), 24000

    stats = generate_dataset(texts, tmp_path, synth)

    metadata = (tmp_path / "metadata.csv").read_text(encoding="utf-8")
    assert "jay_0001|" in metadata
    assert "jay_0002" not in metadata

    assert (tmp_path / "rejected" / "jay_0002.wav").exists()
    rejects = (tmp_path / "rejects.csv").read_text(encoding="utf-8").splitlines()
    assert rejects[0] == "stem|reason|text"
    assert rejects[1].startswith("jay_0002|too_long")
    assert stats.generated == 1
    assert stats.rejected == 1


def screech_tail_clip(sr: int = 22050) -> np.ndarray:
    """Good-sounding body, then the observed failure mode: a stuck high tone.

    Body: tone alternating 200/500 Hz every 60 ms (low centroid, changing
    spectrum). Tail (last 25%): constant 4 kHz tone (centroid jump, zero flux).
    """
    body_t = np.arange(int(4.5 * sr)) / sr
    hop = ((body_t // 0.06) % 2).astype(bool)
    body = 0.3 * np.sin(2 * np.pi * np.where(hop, 500.0, 200.0) * body_t)
    tail_t = np.arange(int(1.5 * sr)) / sr
    tail = 0.3 * np.sin(2 * np.pi * 4000.0 * tail_t)
    return np.concatenate([body, tail]).astype("float32")


def test_screech_tail_clip_is_quarantined(tmp_path):
    texts = ["A normal sentence that is fine.", "This one ends in a screech."]

    def synth(text):
        if "screech" in text:
            return screech_tail_clip(), 22050
        return speech_like(1.5), 24000

    stats = generate_dataset(texts, tmp_path, synth)

    metadata = (tmp_path / "metadata.csv").read_text(encoding="utf-8")
    assert "jay_0002" not in metadata
    rejects = (tmp_path / "rejects.csv").read_text(encoding="utf-8")
    assert "jay_0002|screech_tail" in rejects
    assert stats.rejected == 1


def test_interrupted_run_resumes_without_regenerating(tmp_path):
    texts = ["Sentence number one is here.", "Sentence number two is here.",
             "Sentence number three is here."]
    synthesized: list[str] = []

    def synth(text):
        synthesized.append(text)
        return speech_like(1.5), 24000

    # First run "interrupted" after two clips.
    generate_dataset(texts[:2], tmp_path, synth)
    first_wav_mtime = (tmp_path / "wavs" / "jay_0001.wav").stat().st_mtime

    synthesized.clear()
    stats = generate_dataset(texts, tmp_path, synth)

    assert synthesized == ["Sentence number three is here."]
    assert stats.skipped == 2
    assert stats.generated == 1
    assert (tmp_path / "wavs" / "jay_0001.wav").stat().st_mtime == first_wav_mtime
    meta_lines = (tmp_path / "metadata.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(meta_lines) == 3
    assert len({line.split("|")[0] for line in meta_lines}) == 3


def test_synthesis_failure_is_counted_and_run_continues(tmp_path):
    texts = ["This sentence works just fine.", "This sentence explodes badly.",
             "This later sentence still runs."]

    def synth(text):
        if "explodes" in text:
            raise RuntimeError("CUDA out of memory")
        return speech_like(1.5), 24000

    stats = generate_dataset(texts, tmp_path, synth)

    assert stats.errors == 1
    assert stats.generated == 2
    metadata = (tmp_path / "metadata.csv").read_text(encoding="utf-8")
    assert "jay_0002" not in metadata
    assert "jay_0003|" in metadata
