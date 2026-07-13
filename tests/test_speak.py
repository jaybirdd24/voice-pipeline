"""Seam 3 tests: the Pi speak interface — voice bundle + text -> audible WAV.

The piper CLI and the audio player are system boundaries: tests exercise the
real subprocess wiring against fake executables written to tmp_path, so they
need no piper install, no audio hardware, and no network. The real piper runs
only in the explicit smoke tests (tests/test_speak_smoke.py).
"""
from __future__ import annotations

import json
import os
import stat
import wave
from pathlib import Path

import pytest

from pi.speak import find_bundle, speak_piper, synthesize_wav

# Logs each invocation, plays nothing (no audio hardware in CI/WSL).
FAKE_PLAYER_OK = """#!/usr/bin/env python3
import sys
with open({log!r}, "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
"""

FAKE_PLAYER_FAIL = """#!/usr/bin/env python3
import sys
sys.exit(1)
"""


@pytest.fixture
def tmp_tempdir(tmp_path, monkeypatch):
    """Route tempfile into tmp_path so orphaned temp WAVs are detectable."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path

# A stand-in piper: honours the exact CLI contract speak.py uses (--output_file,
# text on stdin), writes half a second of loud square wave, and logs its
# invocation to <output>.log so tests can assert the wiring.
FAKE_PIPER_OK = """#!/usr/bin/env python3
import json, struct, sys, wave
args = sys.argv[1:]
text = sys.stdin.read()
out = args[args.index("--output_file") + 1]
with wave.open(out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(22050)
    w.writeframes(b"".join(
        struct.pack("<h", 8000 if i % 50 < 25 else -8000) for i in range(11025)
    ))
with open(out + ".log", "w") as f:
    json.dump({"argv": args, "stdin": text}, f)
"""

FAKE_PIPER_FAIL = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
sys.stdin.read()
# Write a partial garbage file before dying, like a real crash mid-synthesis.
with open(args[args.index("--output_file") + 1], "wb") as f:
    f.write(b"RIFFgarbage")
sys.stderr.write("boom: model load failed\\n")
sys.exit(1)
"""


def write_script(path: Path, source: str) -> Path:
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_synthesize_wav_writes_valid_wav_via_piper_cli(tmp_path):
    piper = write_script(tmp_path / "piper", FAKE_PIPER_OK)
    onnx, config = tmp_path / "v.onnx", tmp_path / "v.onnx.json"
    out = tmp_path / "out.wav"

    ok = synthesize_wav("Hello seam.", onnx, config, out, piper_bin=str(piper))

    assert ok is True
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() > 0
    log = json.loads((tmp_path / "out.wav.log").read_text())
    assert log["stdin"] == "Hello seam."
    assert str(onnx) in log["argv"]
    assert str(config) in log["argv"]


def test_synthesize_wav_failure_leaves_no_wav_behind(tmp_path):
    piper = write_script(tmp_path / "piper", FAKE_PIPER_FAIL)
    out = tmp_path / "out.wav"

    ok = synthesize_wav("Anything.", tmp_path / "v.onnx", tmp_path / "v.onnx.json",
                        out, piper_bin=str(piper))

    assert ok is False
    assert not out.exists()


def test_synthesize_wav_handles_missing_piper_binary(tmp_path):
    ok = synthesize_wav("Anything.", tmp_path / "v.onnx", tmp_path / "v.onnx.json",
                        tmp_path / "out.wav", piper_bin="/nonexistent/piper")
    assert ok is False


def test_find_bundle_resolves_voice_files_in_given_dir(tmp_path):
    (tmp_path / "en-qwen3-synth.onnx").write_bytes(b"onnx")
    (tmp_path / "en-qwen3-synth.onnx.json").write_text("{}")

    bundle = find_bundle("en", bundle_dir=tmp_path)
    assert bundle is not None
    onnx, config = bundle
    assert onnx.name == "en-qwen3-synth.onnx"
    assert config.name == "en-qwen3-synth.onnx.json"

    (tmp_path / "en-qwen3-synth.onnx.json").unlink()
    assert find_bundle("en", bundle_dir=tmp_path) is None
    assert find_bundle("klingon", bundle_dir=tmp_path) is None


def test_speak_piper_synthesizes_plays_once_and_cleans_temp(tmp_tempdir, tmp_path):
    piper = write_script(tmp_path / "piper", FAKE_PIPER_OK)
    player_log = tmp_path / "player.log"
    player = write_script(tmp_path / "player", FAKE_PLAYER_OK.format(log=str(player_log)))

    ok = speak_piper("Hello.", tmp_path / "v.onnx", tmp_path / "v.onnx.json",
                     piper_bin=str(piper), player=(str(player),))

    assert ok is True
    played = player_log.read_text().strip().splitlines()
    assert len(played) == 1
    assert played[0].endswith(".wav")
    assert not Path(played[0]).exists()  # temp cleaned after playback


def test_speak_piper_failure_skips_playback_and_leaves_no_orphan(tmp_tempdir, tmp_path):
    piper = write_script(tmp_path / "piper", FAKE_PIPER_FAIL)
    player_log = tmp_path / "player.log"
    player = write_script(tmp_path / "player", FAKE_PLAYER_OK.format(log=str(player_log)))

    ok = speak_piper("Hello.", tmp_path / "v.onnx", tmp_path / "v.onnx.json",
                     piper_bin=str(piper), player=(str(player),))

    assert ok is False
    assert not player_log.exists()  # playback never attempted
    assert list(tmp_tempdir.glob("tmp*.wav")) == []  # no orphaned temp


def test_speak_piper_survives_player_failure_and_cleans_temp(tmp_tempdir, tmp_path):
    piper = write_script(tmp_path / "piper", FAKE_PIPER_OK)
    player = write_script(tmp_path / "player", FAKE_PLAYER_FAIL)

    # A failing audio player must not crash speak.py (was: unhandled
    # CalledProcessError from aplay check=True).
    ok = speak_piper("Hello.", tmp_path / "v.onnx", tmp_path / "v.onnx.json",
                     piper_bin=str(piper), player=(str(player),))

    assert ok is False
    assert list(tmp_tempdir.glob("tmp*.wav")) == []


SPEAK_CLI = Path(__file__).parent.parent / "pi" / "speak.py"


def run_cli(*args: str, piper_bin: str) -> "subprocess.CompletedProcess[bytes]":
    import subprocess
    import sys as _sys

    return subprocess.run(
        [_sys.executable, str(SPEAK_CLI), *args],
        capture_output=True,
        env={**os.environ, "PIPER_BIN": piper_bin},
    )


def test_cli_output_flag_writes_wav_without_playback(tmp_path):
    piper = write_script(tmp_path / "piper", FAKE_PIPER_OK)
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    (bundles / "en-qwen3-synth.onnx").write_bytes(b"onnx")
    (bundles / "en-qwen3-synth.onnx.json").write_text("{}")
    out = tmp_path / "story.wav"

    result = run_cli("--text", "Hello from the CLI.", "--lang", "en",
                     "--bundle-dir", str(bundles), "--output", str(out),
                     piper_bin=str(piper))

    assert result.returncode == 0, result.stderr.decode()
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() > 0
    log = json.loads((tmp_path / "story.wav.log").read_text())
    assert log["stdin"] == "Hello from the CLI."


def test_cli_exits_1_with_helpful_message_when_no_bundle(tmp_path):
    empty = tmp_path / "bundles"
    empty.mkdir()

    result = run_cli("--text", "Hello.", "--lang", "en",
                     "--bundle-dir", str(empty), piper_bin="piper")

    assert result.returncode == 1
    assert b"bundle" in result.stderr.lower()
