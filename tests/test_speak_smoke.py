"""Explicit smoke tests: the real piper CLI synthesises real voices.

Deselected by default (pytest.ini addopts); run with:

    .venv/bin/python -m pytest -m smoke -v

Note the marker filter is needed even when targeting this file directly.
Offline by construction: a local binary and local model files, nothing else.
Paths are env-overridable (PIPER_BIN, STOCK_VOICE) so the suite ports to the Pi.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pi.speak import synthesize_wav

PIPER_BIN = os.environ.get(
    "PIPER_BIN", "/home/jay/p4p/piper_training/repo/piper1-gpl/.venv/bin/piper"
)
STOCK_VOICE = Path(os.environ.get(
    "STOCK_VOICE",
    "/home/jay/p4p/piper_training/pretrained/lessac_onnx/en_US-lessac-medium.onnx",
))
TRAINED_VOICE = Path(__file__).parent.parent / "export/en_pi_bundle/en-qwen3-synth.onnx"

SENTENCE = "The quick brown fox jumps over the lazy dog."


@pytest.mark.smoke
@pytest.mark.parametrize(
    "onnx",
    [
        pytest.param(STOCK_VOICE, id="stock-lessac"),
        pytest.param(TRAINED_VOICE, id="trained-en-qwen3-synth"),
    ],
)
def test_real_piper_synthesises_audible_wav(onnx, tmp_path):
    config = Path(f"{onnx}.json")
    if not Path(PIPER_BIN).is_file():
        pytest.skip(f"piper binary not found: {PIPER_BIN}")
    if not (onnx.is_file() and config.is_file()):
        pytest.skip(f"voice bundle not found: {onnx}")

    out = tmp_path / "smoke.wav"
    t0 = time.monotonic()
    ok = synthesize_wav(SENTENCE, onnx, config, out, piper_bin=PIPER_BIN)
    elapsed = time.monotonic() - t0

    assert ok is True
    wav, sr = sf.read(out)
    duration = len(wav) / sr
    rms = float(np.sqrt(np.mean(np.asarray(wav) ** 2)))
    print(f"\n{onnx.name}: {duration:.2f}s of audio, rms={rms:.3f}, "
          f"synthesis took {elapsed:.2f}s")

    assert sr == json.loads(config.read_text())["audio"]["sample_rate"]
    assert 1.0 < duration < 8.0  # sane length for one pangram sentence
    assert rms > 0.01  # audible, not silence
    assert elapsed < 30  # generous ceiling; value printed above is the evidence
