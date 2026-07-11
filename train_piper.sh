#!/usr/bin/env bash
# Stage 2: Fine-tune a Piper model on the Qwen3-TTS synthetic dataset.
# Usage: ./train_piper.sh --lang en [--smoke] [--max-epochs N] [--batch-size N]
#
# Reads all paths from config.yaml. Run generate_dataset.py first.
# --smoke trains exactly one epoch past the resolved checkpoint (plumbing
# check); note that trainer.max_epochs is ABSOLUTE, not incremental — a value
# at or below the checkpoint's epoch trains nothing.
# Every run appends an entry (success or failure) to docs/training-notes.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
NOTES_FILE="$SCRIPT_DIR/docs/training-notes.md"

# Defaults (overridable via env vars)
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.00002}"
LEARNING_RATE_D="${LEARNING_RATE_D:-0.00001}"
MAX_EPOCHS="${MAX_EPOCHS:-6440}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
SAMPLE_RATE=22050

LANG_CODE=""
SMOKE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang) LANG_CODE="$2"; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -n "$LANG_CODE" ]] || { echo "ERROR: --lang required (en, zh, yue)"; exit 1; }

eval "$(python3 "$SCRIPT_DIR/scripts/read_config.py" --config "$CONFIG" --lang "$LANG_CODE" \
    DATASETS_DIR=datasets_dir \
    PIPER_REPO=piper_repo \
    PIPER_WORKSPACE=piper_training_output_dir \
    ESPEAK_VOICE='languages.{lang}.espeak_voice' \
    PRETRAINED_CHECKPOINT='languages.{lang}.pretrained_checkpoint')"

VENV="$PIPER_REPO/.venv"
DATASET_DIR="${DATASETS_DIR}/${LANG_CODE}_qwen3_synth"
VOICE_NAME="${LANG_CODE}-qwen3-synth"
CSV_PATH="$DATASET_DIR/metadata.csv"
AUDIO_DIR="$DATASET_DIR/wavs"
OUTPUT_DIR="$PIPER_WORKSPACE/training_output_${LANG_CODE}_qwen3_synth"
CACHE_DIR="$OUTPUT_DIR/cache"
CONFIG_OUT="$OUTPUT_DIR/config.json"

# Find the latest fine-tuned checkpoint in OUTPUT_DIR; fall back to pretrained
find_latest_checkpoint() {
    find "$OUTPUT_DIR/lightning_logs" -name "*.ckpt" -printf '%T@ %p\n' 2>/dev/null \
        | awk '/step=[0-9]+/{
            path=substr($0,index($0,$2))
            step=path; sub(/^.*step=/,"",step); sub(/[^0-9].*$/,"",step)
            printf "%012d %s\n", step, path
          }' \
        | sort -rn | head -1 | cut -d' ' -f2-
}

if [ -n "${CHECKPOINT:-}" ]; then
    CHECKPOINT_SOURCE="manual override"
else
    # `|| true`: find exits nonzero when lightning_logs/ doesn't exist yet
    # (first run), which set -e would otherwise turn into a silent death.
    CHECKPOINT="$(find_latest_checkpoint || true)"
    CHECKPOINT_SOURCE="latest fine-tuned checkpoint"
fi
if [ -z "${CHECKPOINT:-}" ]; then
    CHECKPOINT="$PRETRAINED_CHECKPOINT"
    CHECKPOINT_SOURCE="pretrained base checkpoint"
fi

[ -f "$CHECKPOINT" ]  || { echo "ERROR: checkpoint not found: $CHECKPOINT"; echo "Set pretrained_checkpoint in config.yaml for lang '$LANG_CODE'."; exit 1; }
command -v espeak-ng >/dev/null || { echo "ERROR: espeak-ng not installed. sudo apt-get install -y espeak-ng espeak-ng-data"; exit 1; }

# Validate the dataset before spending GPU time on it.
python3 "$SCRIPT_DIR/validate_dataset.py" --dataset-dir "$DATASET_DIR" --min-clips "$BATCH_SIZE"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
source "$VENV/bin/activate"

MODE="production"
if [ "$SMOKE" -eq 1 ]; then
    MODE="smoke"
    # One epoch past the checkpoint. Epoch comes from the filename when
    # present, otherwise from the checkpoint payload itself.
    CKPT_EPOCH="$(basename "$CHECKPOINT" | grep -oP 'epoch=\K[0-9]+' || true)"
    if [ -z "$CKPT_EPOCH" ]; then
        CKPT_EPOCH="$(python3 - "$CHECKPOINT" <<'PYEOF'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch = ckpt.get("epoch")
if epoch is None:
    sys.exit("ERROR: checkpoint has no 'epoch' key; pass --max-epochs explicitly")
print(int(epoch))
PYEOF
)"
    fi
    # Lightning's epoch counter is 0-based: a checkpoint stamped epoch=N has
    # completed N+1 epochs, so max_epochs=N+1 trains nothing. N+2 = one epoch.
    MAX_EPOCHS=$((CKPT_EPOCH + 2))
    echo "Smoke mode: checkpoint at epoch $CKPT_EPOCH ($((CKPT_EPOCH + 1)) epochs done) -> training to max_epochs=$MAX_EPOCHS"
fi

CLIP_COUNT="$(grep -c '|' "$CSV_PATH")"

echo "=== Piper Training: $LANG_CODE ($MODE) ==="
echo "Voice:       $VOICE_NAME"
echo "Dataset:     $DATASET_DIR ($CLIP_COUNT clips)"
echo "Checkpoint:  $CHECKPOINT"
echo "Source:      $CHECKPOINT_SOURCE"
echo "espeak:      $ESPEAK_VOICE"
echo "Max epochs:  $MAX_EPOCHS"
echo "Output:      $OUTPUT_DIR"
echo ""

append_notes() {
    local outcome="$1"
    {
        echo ""
        echo "## $(date '+%Y-%m-%d %H:%M %Z') — $LANG_CODE ($MODE)"
        echo "- dataset: $DATASET_DIR ($CLIP_COUNT clips)"
        echo "- base checkpoint: $CHECKPOINT ($CHECKPOINT_SOURCE)"
        echo "- espeak voice: $ESPEAK_VOICE"
        echo "- hyperparams: batch=$BATCH_SIZE accum=$ACCUMULATE_GRAD_BATCHES lr=$LEARNING_RATE lr_d=$LEARNING_RATE_D max_epochs=$MAX_EPOCHS sample_rate=$SAMPLE_RATE"
        echo "- output dir: $OUTPUT_DIR"
        echo "- outcome: $outcome"
    } >> "$NOTES_FILE"
}

set +e
python3 -m piper.train fit \
    --data.voice_name "$VOICE_NAME" \
    --data.csv_path "$CSV_PATH" \
    --data.audio_dir "$AUDIO_DIR" \
    --model.sample_rate "$SAMPLE_RATE" \
    --data.espeak_voice "$ESPEAK_VOICE" \
    --data.cache_dir "$CACHE_DIR" \
    --data.config_path "$CONFIG_OUT" \
    --data.batch_size "$BATCH_SIZE" \
    --data.trim_silence false \
    --model.learning_rate "$LEARNING_RATE" \
    --model.learning_rate_d "$LEARNING_RATE_D" \
    --model.accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES" \
    --trainer.log_every_n_steps "$LOG_EVERY_N_STEPS" \
    --trainer.max_epochs "$MAX_EPOCHS" \
    --trainer.default_root_dir "$OUTPUT_DIR" \
    --ckpt_path "$CHECKPOINT"
TRAIN_EXIT=$?
set -e

if [ "$TRAIN_EXIT" -eq 0 ]; then
    FINAL_CKPT="$(find_latest_checkpoint || true)"
    append_notes "success — final ckpt: ${FINAL_CKPT:-<none found>}"
    echo ""
    echo "Training complete. Notes appended to $NOTES_FILE"
else
    append_notes "FAILED (exit $TRAIN_EXIT)"
    echo ""
    echo "Training FAILED (exit $TRAIN_EXIT). Notes appended to $NOTES_FILE" >&2
fi
exit "$TRAIN_EXIT"
