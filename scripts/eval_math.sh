#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

CALLER_PWD="$(pwd)"
cd "$(cd "$(dirname "$0")/.." && pwd)/math_evaluation"

usage() {
  cat <<'USAGE' >&2
Usage:
  bash scripts/eval_math.sh \
    --model CHECKPOINT_OR_MODEL \
    --output-dir OUTPUT_DIR \
    [--prompt-type qwen-boxed] \
    [--n-sampling 16] \
    [--temperature 1] \
    [--tokenizer-mode slow] \
    [--cuda-visible-devices 0]
USAGE
  exit 2
}

resolve_model_path() {
  local path="$1"
  local latest

  if [[ -d "$path" && ! -f "$path/config.json" ]]; then
    latest=$(find "$path" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)
    if [[ -n "$latest" && -f "$latest/config.json" ]]; then
      echo "$latest"
      return
    fi
  fi

  echo "$path"
}

MODEL=""
OUTPUT_DIR=""
PROMPT_TYPE="qwen-boxed"
N_SAMPLING="16"
TEMPERATURE="1"
TOKENIZER_MODE="slow"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:-}"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"; shift 2 ;;
    --prompt-type)
      PROMPT_TYPE="${2:-}"; shift 2 ;;
    --n-sampling)
      N_SAMPLING="${2:-}"; shift 2 ;;
    --temperature)
      TEMPERATURE="${2:-}"; shift 2 ;;
    --tokenizer-mode)
      TOKENIZER_MODE="${2:-}"; shift 2 ;;
    --cuda-visible-devices|--gpus)
      CUDA_DEVICES="${2:-}"; shift 2 ;;
    *)
      usage ;;
  esac
done

[[ -n "$MODEL" && -n "$OUTPUT_DIR" ]] || usage

if [[ "$MODEL" != /* && -e "$CALLER_PWD/$MODEL" ]]; then
  MODEL="$CALLER_PWD/$MODEL"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$CALLER_PWD/$OUTPUT_DIR"
fi

MODEL=$(resolve_model_path "$MODEL")
mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
  --model_name_or_path "$MODEL" \
  --data_names math_oai,minerva_math,olympiadbench,aime24,amc23 \
  --output_dir "$OUTPUT_DIR" \
  --split test \
  --prompt_type "$PROMPT_TYPE" \
  --num_test_sample -1 \
  --seed 0 \
  --temperature "$TEMPERATURE" \
  --n_sampling "$N_SAMPLING" \
  --tokenizer_mode "$TOKENIZER_MODE" \
  --top_p 1 \
  --start 0 \
  --end -1 \
  --use_vllm

echo "Avg@${N_SAMPLING} / Pass@${N_SAMPLING} summary: ${OUTPUT_DIR}/summary_metrics.json"
