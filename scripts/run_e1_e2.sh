#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
TEST_FILE="${TEST_FILE:-data/processed/gsm8k_test.jsonl}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-data/processed/sft_train.jsonl}"
SFT_VAL_FILE="${SFT_VAL_FILE:-data/processed/sft_val.jsonl}"
SFT_CONFIG="${SFT_CONFIG:-configs/sft_qwen_1.5b_local.yaml}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/base_1.5b}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-outputs/sft_qwen_1.5b}"
SFT_EVAL_DIR="${SFT_EVAL_DIR:-outputs/sft_qwen_1.5b_eval}"
RESULTS_DIR="${RESULTS_DIR:-results}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
    exit 1
  fi
}

echo "Using model: $MODEL_PATH"
require_dir "$MODEL_PATH"
require_file "$TEST_FILE"
require_file "$SFT_TRAIN_FILE"
require_file "$SFT_VAL_FILE"
require_file "$SFT_CONFIG"

mkdir -p "$BASE_OUTPUT_DIR" "$SFT_OUTPUT_DIR" "$SFT_EVAL_DIR" "$RESULTS_DIR"

echo
echo "===== E1: Base 1.5B evaluation ====="
python scripts/evaluate.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_file "$TEST_FILE" \
  --output_jsonl "$BASE_OUTPUT_DIR/predictions.jsonl" \
  --metrics_file "$RESULTS_DIR/base_1.5b_metrics.json" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS"

echo
echo "E1 metrics:"
cat "$RESULTS_DIR/base_1.5b_metrics.json"

echo
echo "===== E2: CoT-SFT training ====="
python scripts/train_sft.py --config "$SFT_CONFIG"

echo
echo "===== E2: CoT-SFT evaluation ====="
python scripts/evaluate.py \
  --model_name_or_path "$MODEL_PATH" \
  --adapter_path "$SFT_OUTPUT_DIR" \
  --data_file "$TEST_FILE" \
  --output_jsonl "$SFT_EVAL_DIR/predictions.jsonl" \
  --metrics_file "$RESULTS_DIR/sft_qwen_1.5b_metrics.json" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS"

echo
echo "E2 metrics:"
cat "$RESULTS_DIR/sft_qwen_1.5b_metrics.json"

echo
echo "Done. Key outputs:"
echo "- $RESULTS_DIR/base_1.5b_metrics.json"
echo "- $RESULTS_DIR/sft_qwen_1.5b_metrics.json"
echo "- $BASE_OUTPUT_DIR/predictions.jsonl"
echo "- $SFT_EVAL_DIR/predictions.jsonl"
