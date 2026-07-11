#!/usr/bin/env bash
# Stage 3: Export the trained Piper model to ONNX and verify the Pi bundle.
# Usage: ./export_piper.sh --lang en [--checkpoint /path/to/ckpt]
#
# Writes the bundle to the pi_bundle path from config.yaml and refuses to
# finish unless the exported voice actually loads and synthesises a sentence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"

LANG_CODE=""
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang) LANG_CODE="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -n "$LANG_CODE" ]] || { echo "ERROR: --lang required (en, zh, yue)"; exit 1; }

eval "$(python3 "$SCRIPT_DIR/scripts/read_config.py" --config "$CONFIG" --lang "$LANG_CODE" \
    PIPER_REPO=piper_repo \
    PIPER_WORKSPACE=piper_training_output_dir \
    PI_BUNDLE='languages.{lang}.pi_bundle')"

VENV="$PIPER_REPO/.venv"
OUTPUT_DIR="$PIPER_WORKSPACE/training_output_${LANG_CODE}_qwen3_synth"
EXPORT_DIR="$PI_BUNDLE"
VOICE_NAME="${LANG_CODE}-qwen3-synth"

find_latest_checkpoint() {
    find "$OUTPUT_DIR/lightning_logs" -name "*.ckpt" -printf '%T@ %p\n' 2>/dev/null \
        | awk '/step=[0-9]+/{
            path=substr($0,index($0,$2))
            step=path; sub(/^.*step=/,"",step); sub(/[^0-9].*$/,"",step)
            printf "%012d %s\n", step, path
          }' \
        | sort -rn | head -1 | cut -d' ' -f2-
}

if [ -z "$CHECKPOINT" ]; then
    # `|| true`: find exits nonzero when lightning_logs/ doesn't exist,
    # which set -e would otherwise turn into a silent death.
    CHECKPOINT="$(find_latest_checkpoint || true)"
else
    echo "NOTE: --checkpoint override in use. The bundled .onnx.json is copied from"
    echo "      the LATEST training run's config.json and may not match this checkpoint."
fi
[ -n "$CHECKPOINT" ] || { echo "ERROR: no checkpoint found in $OUTPUT_DIR. Run train_piper.sh first."; exit 1; }
[ -f "$CHECKPOINT" ] || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

CONFIG_JSON="$OUTPUT_DIR/config.json"
[ -f "$CONFIG_JSON" ] || { echo "ERROR: config.json not found: $CONFIG_JSON. Run train_piper.sh first."; exit 1; }

mkdir -p "$EXPORT_DIR"
ONNX_OUT="$EXPORT_DIR/${VOICE_NAME}.onnx"
JSON_OUT="$EXPORT_DIR/${VOICE_NAME}.onnx.json"

echo "=== Piper ONNX Export: $LANG_CODE ==="
echo "Checkpoint: $CHECKPOINT"
echo "Bundle:     $EXPORT_DIR"
echo ""

source "$VENV/bin/activate"

python3 -m piper.train.export_onnx \
    --checkpoint "$CHECKPOINT" \
    --output-file "$ONNX_OUT"

cp "$CONFIG_JSON" "$JSON_OUT"

# Verify the bundle: the exported voice must load and speak one sentence.
# The check wav stays outside the bundle dir (pi/speak.py expects exactly
# {voice}.onnx + {voice}.onnx.json there).
CHECK_WAV="$(mktemp -d)/bundle_check.wav"
echo "Hello, this is a check of the exported voice." \
    | piper -m "$ONNX_OUT" -c "$JSON_OUT" -f "$CHECK_WAV"
python3 - "$CHECK_WAV" <<'PYEOF'
import sys
import wave

with wave.open(sys.argv[1], "rb") as w:
    rate, frames = w.getframerate(), w.getnframes()
assert rate == 22050, f"unexpected sample rate: {rate}"
assert frames > 0, "exported voice produced no audio"
print(f"Bundle check OK: synthesised {frames / rate:.1f}s of audio at {rate} Hz")
PYEOF

echo ""
echo "=== Export complete ==="
ls -lh "$EXPORT_DIR/"
echo ""
echo "Listen locally:"
echo "  aplay $CHECK_WAV"
echo ""
echo "Copy to Pi:"
echo "  scp $ONNX_OUT $JSON_OUT pi@raspberrypi.local:~/tts/"
