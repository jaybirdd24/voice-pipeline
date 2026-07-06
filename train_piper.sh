#!/usr/bin/env bash
# Stage 2: Train a Piper model on the Qwen3-TTS synthetic dataset.
# Usage: ./train_piper.sh --lang en [options]
#
# Reads paths from config.yaml. Run generate_dataset.py first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"

# Piper training workspace (from p4p)
PIPER_WORKSPACE="/home/jay/p4p/piper_training"
REPO="$PIPER_WORKSPACE/repo/piper1-gpl"
VENV="$REPO/.venv"

# Defaults (overridable via env vars)
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.00002}"
LEARNING_RATE_D="${LEARNING_RATE_D:-0.00001}"
MAX_EPOCHS="${MAX_EPOCHS:-6440}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
SAMPLE_RATE=22050

LANG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang) LANG="$2"; shift 2 ;;
        --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -n "$LANG" ]] || { echo "ERROR: --lang required (en, zh, yue)"; exit 1; }

# Read config.yaml using Python (avoids yq dependency)
read -r DATASETS_DIR ESPEAK_VOICE PRETRAINED_CHECKPOINT <<< "$(python3 - <<PYEOF
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
lang = cfg['languages']['$LANG']
print(cfg['datasets_dir'], lang['espeak_voice'], lang['pretrained_checkpoint'])
PYEOF
)"

DATASET_DIR="${DATASETS_DIR}/${LANG}_qwen3_synth"
VOICE_NAME="${LANG}-qwen3-synth"
CSV_PATH="$DATASET_DIR/metadata.csv"
AUDIO_DIR="$DATASET_DIR/wavs"
OUTPUT_DIR="$PIPER_WORKSPACE/training_output_${LANG}_qwen3_synth"
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
    CHECKPOINT="$(find_latest_checkpoint)"
    CHECKPOINT_SOURCE="latest fine-tuned checkpoint"
fi
if [ -z "${CHECKPOINT:-}" ]; then
    CHECKPOINT="$PRETRAINED_CHECKPOINT"
    CHECKPOINT_SOURCE="pretrained base checkpoint"
fi

echo "=== Piper Training: $LANG ==="
echo "Voice:       $VOICE_NAME"
echo "Dataset:     $DATASET_DIR"
echo "Checkpoint:  $CHECKPOINT"
echo "Source:      $CHECKPOINT_SOURCE"
echo "espeak:      $ESPEAK_VOICE"
echo "Output:      $OUTPUT_DIR"
echo ""

[ -f "$CSV_PATH" ]    || { echo "ERROR: metadata.csv not found: $CSV_PATH"; echo "Run generate_dataset.py --lang $LANG first."; exit 1; }
[ -d "$AUDIO_DIR" ]   || { echo "ERROR: wavs/ dir not found: $AUDIO_DIR"; exit 1; }
[ -f "$CHECKPOINT" ]  || { echo "ERROR: checkpoint not found: $CHECKPOINT"; echo "Set pretrained_checkpoint in config.yaml for lang '$LANG'."; exit 1; }
command -v espeak-ng >/dev/null || { echo "ERROR: espeak-ng not installed. sudo apt-get install -y espeak-ng espeak-ng-data"; exit 1; }

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
source "$VENV/bin/activate"

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
