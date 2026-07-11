#!/usr/bin/env bash
# End-to-end pipeline: generate dataset → train Piper → export ONNX
# Each stage is skipped if its output already exists (resume-friendly).
# Usage: ./run_pipeline.sh --lang en [--count 500] [--max-epochs 6440] [--smoke]
#
# --smoke passes through to train_piper.sh: one epoch past the checkpoint,
# a fast rehearsal of the full pipeline path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"

LANG_CODE=""
COUNT=500
MAX_EPOCHS=6440
SKIP_GENERATE=false
SKIP_TRAIN=false
SMOKE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang)          LANG_CODE="$2";    shift 2 ;;
        --count)         COUNT="$2";        shift 2 ;;
        --max-epochs)    MAX_EPOCHS="$2";   shift 2 ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --skip-train)    SKIP_TRAIN=true;   shift ;;
        --smoke)         SMOKE=true;        shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -n "$LANG_CODE" ]] || { echo "ERROR: --lang required (en, zh, yue)"; exit 1; }

eval "$(python3 "$SCRIPT_DIR/scripts/read_config.py" --config "$CONFIG" --lang "$LANG_CODE" \
    DATASETS_DIR=datasets_dir \
    PIPER_WORKSPACE=piper_training_output_dir \
    EXPORT_DIR='languages.{lang}.pi_bundle')"

DATASET_DIR="${DATASETS_DIR}/${LANG_CODE}_qwen3_synth"

echo "================================================================"
echo " Voice Pipeline — $LANG_CODE"
echo " Stages: generate → train → export"
echo "================================================================"

# ----- Stage 1: Generate dataset -----
if $SKIP_GENERATE; then
    echo "[Stage 1] Skipped (--skip-generate)"
elif [ -f "$DATASET_DIR/metadata.csv" ]; then
    EXISTING=$(wc -l < "$DATASET_DIR/metadata.csv")
    echo "[Stage 1] Dataset exists ($EXISTING clips) — skipping generation"
    echo "          Delete $DATASET_DIR/metadata.csv to regenerate"
else
    echo "[Stage 1] Generating synthetic dataset ($COUNT clips, lang=$LANG_CODE)..."
    source /home/jay/p4p/qwen3_tts_test/.venv/bin/activate
    python3 "$SCRIPT_DIR/generate_dataset.py" \
        --lang "$LANG_CODE" \
        --count "$COUNT" \
        --config "$CONFIG"
    echo "[Stage 1] Done."
fi

echo ""

# ----- Stage 2: Train Piper -----
if $SKIP_TRAIN; then
    echo "[Stage 2] Skipped (--skip-train)"
else
    TRAIN_ARGS=(--lang "$LANG_CODE")
    if $SMOKE; then
        TRAIN_ARGS+=(--smoke)
        echo "[Stage 2] Training Piper (lang=$LANG_CODE, smoke mode)..."
    else
        TRAIN_ARGS+=(--max-epochs "$MAX_EPOCHS")
        echo "[Stage 2] Training Piper (lang=$LANG_CODE, max_epochs=$MAX_EPOCHS)..."
    fi
    echo "          Watch TensorBoard: tensorboard --logdir $PIPER_WORKSPACE/training_output_${LANG_CODE}_qwen3_synth"
    bash "$SCRIPT_DIR/train_piper.sh" "${TRAIN_ARGS[@]}"
    echo "[Stage 2] Done."
fi

echo ""

# ----- Stage 3: Export -----
if [ -d "$EXPORT_DIR" ] && ls "$EXPORT_DIR"/*.onnx &>/dev/null; then
    echo "[Stage 3] ONNX bundle exists in $EXPORT_DIR — skipping export"
    echo "          Delete the .onnx file to re-export"
else
    echo "[Stage 3] Exporting ONNX..."
    bash "$SCRIPT_DIR/export_piper.sh" --lang "$LANG_CODE"
    echo "[Stage 3] Done."
fi

echo ""
echo "================================================================"
echo " Pipeline complete for lang=$LANG_CODE"
echo " Bundle: $EXPORT_DIR"
echo ""
echo " Next steps:"
echo "   • Listen: echo 'Hello.' | piper --model $EXPORT_DIR/*.onnx --output_file /tmp/test.wav && aplay /tmp/test.wav"
echo "   • Copy to Pi: scp $EXPORT_DIR/* pi@raspberrypi.local:~/tts/"
echo "   • Run demo: python pi/story_reader.py"
echo "================================================================"
