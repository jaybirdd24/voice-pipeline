#!/usr/bin/env python3
"""Fallback TTS server: Qwen3-TTS with per-story disk cache.

POST /speak  {text: str, lang: str}  → WAV audio (audio/wav)
GET  /health                          → {"status": "ok", "model": "..."}

Cache: fallback/cache/<sha256(lang+text)>.wav
Each unique (lang, text) pair is synthesised once; subsequent requests return the cached file.

Usage:
    # activate the qwen3 venv first
    source /home/jay/p4p/qwen3_tts_test/.venv/bin/activate
    pip install -r requirements.txt   # if not already installed
    python fallback/serve.py [--config ../config.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent

LANG_TO_QWEN = {
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
}


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_key(lang: str, text: str) -> str:
    return hashlib.sha256(f"{lang}:{text}".encode("utf-8")).hexdigest()


def build_app(cfg: dict):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel
    import soundfile as sf
    import numpy as np

    app = FastAPI(title="voice-pipeline fallback TTS")

    model_name = cfg.get("qwen3_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    device = cfg.get("device", "cuda:0")
    cache_dir = Path(cfg["fallback"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        print("Activate the qwen3 venv and install requirements.txt", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_name} on {device}...")
    _model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print("Model ready.")

    def _synthesise(lang: str, text: str, lang_cfg: dict) -> bytes:
        reference_audio = Path(lang_cfg["reference_audio"]).expanduser()
        reference_transcript = lang_cfg["reference_transcript"].strip()
        qwen_language = LANG_TO_QWEN[lang]

        wavs, sample_rate = _model.generate_voice_clone(
            text=text,
            language=qwen_language,
            ref_audio=str(reference_audio),
            ref_text=reference_transcript,
        )
        if not wavs:
            raise RuntimeError("Qwen3-TTS returned empty audio")

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sample_rate, format="WAV")
        return buf.getvalue()

    class SpeakRequest(BaseModel):
        text: str
        lang: str = "en"

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.post("/speak")
    def speak(req: SpeakRequest):
        lang = req.lang.lower()
        if lang not in LANG_TO_QWEN:
            raise HTTPException(400, f"Unsupported lang '{lang}'. Use: {list(LANG_TO_QWEN)}")

        lang_cfg = cfg["languages"].get(lang, {})
        if not lang_cfg.get("reference_audio"):
            raise HTTPException(400, f"No reference_audio configured for lang '{lang}'")

        key = cache_key(lang, req.text)
        cached = cache_dir / f"{key}.wav"
        if cached.exists():
            return Response(content=cached.read_bytes(), media_type="audio/wav",
                            headers={"X-Cache": "HIT"})

        wav_bytes = _synthesise(lang, req.text, lang_cfg)
        cached.write_bytes(wav_bytes)
        return Response(content=wav_bytes, media_type="audio/wav",
                        headers={"X-Cache": "MISS"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg["fallback"].get("host", "0.0.0.0")
    port = args.port or cfg["fallback"].get("port", 8000)

    import uvicorn
    app = build_app(cfg)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
