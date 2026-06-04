#!/usr/bin/env bash
# =============================================================================
# One-click runner for the full GRPO x CoT distillation study (E1 - E5).
#
# Pipeline:
#   data  -> (optional) prepare_data.py
#   test  -> (optional) answer-extraction unit test
#   e1    -> Base evaluation
#   e2    -> CoT-SFT training + evaluation
#   merge -> merge SFT LoRA into a standalone model (needed by E4/E5)
#   e3    -> GRPO on base model (correctness-only)
#   e4r1  -> GRPO on SFT-merged model (correctness)
#   e4r2  -> GRPO on SFT-merged model (correctness + format)
#   e5r3  -> GRPO on SFT-merged model (correctness + format + length penalty)
#   report-> aggregate all metrics to CSV + plots
#
# Usage:
#   bash scripts/run_all.sh                     # run every stage
#   STAGES="merge e3 e4r1 e4r2 e5r3 report" \
#       bash scripts/run_all.sh                 # only the GRPO stages + report
#   MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct \
#       bash scripts/run_all.sh                 # override base model path
#   FORCE=1 bash scripts/run_all.sh             # re-run even if outputs exist
#
# Every stage logs to logs/<stage>.log (also streamed to the console).
# Already-finished stages are skipped automatically unless FORCE=1.
# =============================================================================
set -euo pipefail

# ---- Configuration (override via environment variables) ----------------------
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"

DATA_DIR="${DATA_DIR:-data/processed}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/gsm8k_test.jsonl}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-$DATA_DIR/sft_train.jsonl}"
SFT_VAL_FILE="${SFT_VAL_FILE:-$DATA_DIR/sft_val.jsonl}"
GRPO_TRAIN_FILE="${GRPO_TRAIN_FILE:-$DATA_DIR/grpo_train.jsonl}"

SFT_CONFIG="${SFT_CONFIG:-configs/sft_qwen_1.5b.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results}"
FIG_DIR="${FIG_DIR:-results/figures}"
LOG_DIR="${LOG_DIR:-logs}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# Output locations
SFT_OUTPUT_DIR="outputs/sft_qwen_1.5b"
SFT_MERGED_DIR="outputs/sft_qwen_1.5b_merged"

# Which stages to run (space-separated). Default: everything.
STAGES="${STAGES:-data test e1 e2 merge e3 e4r1 e4r2 e5r3 report}"
# Set FORCE=1 to ignore "already done" checks and re-run.
FORCE="${FORCE:-0}"

mkdir -p "$RESULTS_DIR" "$FIG_DIR" "$LOG_DIR"

# ---- Helpers ----------------------------------------------------------------
want() { [[ " $STAGES " == *" $1 "* ]]; }

require_file() { [[ -f "$1" ]] || { echo "[FATAL] missing file: $1" >&2; exit 1; }; }
require_path() { [[ -e "$1" ]] || { echo "[FATAL] missing model/path: $1" >&2; exit 1; }; }

# done_marker <file-or-dir> : true if it exists and FORCE is not set.
done_marker() { [[ "$FORCE" != "1" && -e "$1" ]]; }

# run_stage <name> <done-marker> -- <command...>
# Skips if not selected, or if the done-marker exists (unless FORCE=1).
run_stage() {
  local name="$1"; local marker="$2"; shift 2
  [[ "$1" == "--" ]] && shift
  if ! want "$name"; then
    echo "[SKIP] $name (not in STAGES)"
    return 0
  fi
  if done_marker "$marker"; then
    echo "[SKIP] $name (already done: $marker ; set FORCE=1 to re-run)"
    return 0
  fi
  echo
  echo "==================================================================="
  echo ">>> STAGE $name  ->  $(date '+%F %T')"
  echo "==================================================================="
  # Stream to console AND capture to a per-stage log.
  "$@" 2>&1 | tee "$LOG_DIR/$name.log"
  echo "<<< STAGE $name done  ->  $(date '+%F %T')"
}

# eval_model <base-model> <adapter-or-"none"> <out-dir> <metrics-json>
eval_model() {
  local base="$1"; local adapter="$2"; local outdir="$3"; local metrics="$4"
  mkdir -p "$outdir"
  local adapter_args=()
  # NOTE: a bare `[[ ... ]] && ...` returns non-zero when the test is false,
  # which would abort this function under `set -e`. Use an explicit if.
  if [[ "$adapter" != "none" ]]; then
    adapter_args=(--adapter_path "$adapter")
  fi
  python scripts/evaluate.py \
    --model_name_or_path "$base" \
    "${adapter_args[@]}" \
    --data_file "$TEST_FILE" \
    --output_jsonl "$outdir/predictions.jsonl" \
    --metrics_file "$metrics" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"
}

echo "Base model : $MODEL_PATH"
echo "Stages     : $STAGES"
echo "Force rerun: $FORCE"
require_path "$MODEL_PATH"

# ---- data: prepare datasets (only if the test file is missing) ---------------
run_stage data "$TEST_FILE" -- \
  python scripts/prepare_data.py \
    --sft_size 1500 --grpo_size 800 --val_size 300 \
    --out_dir "$DATA_DIR" --seed 42

# These must exist before anything else.
require_file "$TEST_FILE"
require_file "$SFT_TRAIN_FILE"

# ---- test: answer-extraction unit test --------------------------------------
run_stage test "" -- python src/test_extraction.py

# ---- E1: Base evaluation -----------------------------------------------------
run_stage e1 "$RESULTS_DIR/base_1.5b_metrics.json" -- \
  eval_model "$MODEL_PATH" none outputs/base_1.5b "$RESULTS_DIR/base_1.5b_metrics.json"

# ---- E2: CoT-SFT training + evaluation ---------------------------------------
run_stage e2 "$RESULTS_DIR/sft_metrics.json" -- bash -c '
  set -euo pipefail
  python scripts/train_sft.py --config "'"$SFT_CONFIG"'" --model_name_or_path "'"$MODEL_PATH"'"
  python scripts/evaluate.py \
    --model_name_or_path "'"$MODEL_PATH"'" \
    --adapter_path "'"$SFT_OUTPUT_DIR"'" \
    --data_file "'"$TEST_FILE"'" \
    --output_jsonl outputs/sft/predictions.jsonl \
    --metrics_file "'"$RESULTS_DIR"'/sft_metrics.json" \
    --batch_size "'"$BATCH_SIZE"'" --max_new_tokens "'"$MAX_NEW_TOKENS"'"
'

# ---- merge: SFT LoRA -> standalone model (needed by E4/E5) -------------------
run_stage merge "$SFT_MERGED_DIR/config.json" -- \
  python scripts/merge_lora.py \
    --base_model "$MODEL_PATH" \
    --adapter_path "$SFT_OUTPUT_DIR" \
    --output_dir "$SFT_MERGED_DIR"

# ---- E3: GRPO on base model --------------------------------------------------
run_stage e3 "$RESULTS_DIR/grpo_base_r1_metrics.json" -- bash -c '
  set -euo pipefail
  python scripts/train_grpo.py --config configs/grpo_base_r1.yaml --model_name_or_path "'"$MODEL_PATH"'"
  python scripts/evaluate.py \
    --model_name_or_path "'"$MODEL_PATH"'" \
    --adapter_path outputs/grpo_base_r1 \
    --data_file "'"$TEST_FILE"'" \
    --output_jsonl outputs/grpo_base_r1/predictions.jsonl \
    --metrics_file "'"$RESULTS_DIR"'/grpo_base_r1_metrics.json" \
    --batch_size "'"$BATCH_SIZE"'" --max_new_tokens "'"$MAX_NEW_TOKENS"'"
'

# ---- E4-R1: GRPO on SFT-merged (correctness) --------------------------------
run_stage e4r1 "$RESULTS_DIR/grpo_sft_r1_correct_metrics.json" -- bash -c '
  set -euo pipefail
  python scripts/train_grpo.py --config configs/grpo_sft_r1_correct.yaml
  python scripts/evaluate.py \
    --model_name_or_path "'"$SFT_MERGED_DIR"'" \
    --adapter_path outputs/grpo_sft_r1_correct \
    --data_file "'"$TEST_FILE"'" \
    --output_jsonl outputs/grpo_sft_r1_correct/predictions.jsonl \
    --metrics_file "'"$RESULTS_DIR"'/grpo_sft_r1_correct_metrics.json" \
    --batch_size "'"$BATCH_SIZE"'" --max_new_tokens "'"$MAX_NEW_TOKENS"'"
'

# ---- E4-R2: GRPO on SFT-merged (correctness + format) -----------------------
run_stage e4r2 "$RESULTS_DIR/grpo_sft_r2_format_metrics.json" -- bash -c '
  set -euo pipefail
  python scripts/train_grpo.py --config configs/grpo_sft_r2_format.yaml
  python scripts/evaluate.py \
    --model_name_or_path "'"$SFT_MERGED_DIR"'" \
    --adapter_path outputs/grpo_sft_r2_format \
    --data_file "'"$TEST_FILE"'" \
    --output_jsonl outputs/grpo_sft_r2_format/predictions.jsonl \
    --metrics_file "'"$RESULTS_DIR"'/grpo_sft_r2_format_metrics.json" \
    --batch_size "'"$BATCH_SIZE"'" --max_new_tokens "'"$MAX_NEW_TOKENS"'"
'

# ---- E5-R3: GRPO on SFT-merged (correctness + format + length penalty) ------
run_stage e5r3 "$RESULTS_DIR/grpo_sft_r3_length_metrics.json" -- bash -c '
  set -euo pipefail
  python scripts/train_grpo.py --config configs/grpo_sft_r3_length.yaml
  python scripts/evaluate.py \
    --model_name_or_path "'"$SFT_MERGED_DIR"'" \
    --adapter_path outputs/grpo_sft_r3_length \
    --data_file "'"$TEST_FILE"'" \
    --output_jsonl outputs/grpo_sft_r3_length/predictions.jsonl \
    --metrics_file "'"$RESULTS_DIR"'/grpo_sft_r3_length_metrics.json" \
    --batch_size "'"$BATCH_SIZE"'" --max_new_tokens "'"$MAX_NEW_TOKENS"'"
'

# ---- report: aggregate + plot (only over prediction files that exist) -------
if want report; then
  echo
  echo "==================================================================="
  echo ">>> STAGE report  ->  $(date '+%F %T')"
  echo "==================================================================="
  PRED_FILES=()
  LABELS=()
  # Ends in an if (returns 0 even when the file is absent) so `set -e` is happy.
  add_pred() { if [[ -f "$1" ]]; then PRED_FILES+=("$1"); LABELS+=("$2"); fi; }
  add_pred outputs/base_1.5b/predictions.jsonl            Base
  add_pred outputs/sft/predictions.jsonl                  CoT-SFT
  add_pred outputs/grpo_base_r1/predictions.jsonl         GRPO
  add_pred outputs/grpo_sft_r1_correct/predictions.jsonl  SFT-GRPO-R1
  add_pred outputs/grpo_sft_r2_format/predictions.jsonl   SFT-GRPO-R2
  add_pred outputs/grpo_sft_r3_length/predictions.jsonl   SFT-GRPO-R3
  if [[ ${#PRED_FILES[@]} -eq 0 ]]; then
    echo "[WARN] no prediction files found; nothing to aggregate."
  else
    python scripts/analyze_outputs.py \
      --prediction_files "${PRED_FILES[@]}" \
      --labels "${LABELS[@]}" \
      --output_csv "$RESULTS_DIR/metrics_summary.csv" \
      --error_csv "$RESULTS_DIR/error_cases.csv" 2>&1 | tee "$LOG_DIR/report.log"
    python scripts/plot_results.py \
      --metrics_csv "$RESULTS_DIR/metrics_summary.csv" \
      --output_dir "$FIG_DIR" 2>&1 | tee -a "$LOG_DIR/report.log"
  fi
fi

# ---- Final summary ----------------------------------------------------------
echo
echo "==================================================================="
echo "ALL DONE. Metrics summary:"
echo "==================================================================="
for m in \
  "$RESULTS_DIR/base_1.5b_metrics.json" \
  "$RESULTS_DIR/sft_metrics.json" \
  "$RESULTS_DIR/grpo_base_r1_metrics.json" \
  "$RESULTS_DIR/grpo_sft_r1_correct_metrics.json" \
  "$RESULTS_DIR/grpo_sft_r2_format_metrics.json" \
  "$RESULTS_DIR/grpo_sft_r3_length_metrics.json"; do
  if [[ -f "$m" ]]; then
    acc=$(python -c "import json;d=json.load(open('$m'));print(f\"strict={d['accuracy_strict']:.3f} format={d['format_success_rate']:.3f} len={d['avg_response_length_words']:.1f}\")" 2>/dev/null || echo "(unreadable)")
    printf "  %-45s %s\n" "$(basename "$m")" "$acc"
  fi
done
echo
echo "CSV : $RESULTS_DIR/metrics_summary.csv"
echo "Figs: $FIG_DIR"
echo "Logs: $LOG_DIR/"
