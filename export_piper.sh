#!/usr/bin/env bash
# Stage 3: Export the trained Piper model to ONNX and prepare the Pi bundle.
# Usage: ./export_piper.sh --lang en [--checkpoint /path/to/ckpt]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PIPER_WORKSPACE="/home/jay/p4p/piper_training"
REPO="$PIPER_WORKSPACE/repo/piper1-gpl"
VENV="$REPO/.venv"

LANG=""
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang) LANG="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -n "$LANG" ]] || { echo "ERROR: --lang required (en, zh, yue)"; exit 1; }

OUTPUT_DIR="$PIPER_WORKSPACE/training_output_${LANG}_qwen3_synth"
EXPORT_DIR="$SCRIPT_DIR/export/${LANG}_pi_bundle"
VOICE_NAME="${LANG}-qwen3-synth"

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
    CHECKPOINT="$(find_latest_checkpoint)"
fi
[ -n "$CHECKPOINT" ] || { echo "ERROR: no checkpoint found in $OUTPUT_DIR. Run train_piper.sh first."; exit 1; }
[ -f "$CHECKPOINT" ] || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

CONFIG_JSON="$OUTPUT_DIR/config.json"
[ -f "$CONFIG_JSON" ] || { echo "ERROR: config.json not found: $CONFIG_JSON. Run train_piper.sh first."; exit 1; }

mkdir -p "$EXPORT_DIR"
ONNX_OUT="$EXPORT_DIR/${VOICE_NAME}.onnx"
JSON_OUT="$EXPORT_DIR/${VOICE_NAME}.onnx.json"

echo "=== Piper ONNX Export: $LANG ==="
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $ONNX_OUT"
echo ""

source "$VENV/bin/activate"

python3 -m piper.train.export_onnx \
    --checkpoint "$CHECKPOINT" \
    --output-file "$ONNX_OUT"

cp "$CONFIG_JSON" "$JSON_OUT"

echo ""
echo "=== Export complete ==="
ls -lh "$EXPORT_DIR/"
echo ""
echo "Test locally:"
echo "  echo 'Hello.' | piper --model $ONNX_OUT --output_file /tmp/test_${LANG}.wav && aplay /tmp/test_${LANG}.wav"
echo ""
echo "Copy to Pi:"
echo "  scp $ONNX_OUT $JSON_OUT pi@raspberrypi.local:~/tts/"
