#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

MODEL="Qwen/Qwen3-8B-Base"
NPROC="${NPROC:-4}"
MICRO_BSZ="${MICRO_BSZ:-4}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen-boxed}"
NSAMP="${NSAMP:-16}"
TEMPERATURE="${TEMPERATURE:-1}"
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-0}"

if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  NPROC="$1"
  shift
fi

if [[ $# -gt 0 ]]; then
  METHODS=("$@")
else
  METHODS=(prift_prob prift_mass)
fi

SHORT=$(basename "$MODEL")
for METHOD in "${METHODS[@]}"; do
  conda run --no-capture-output -n prift-train-qwen \
    bash scripts/train.sh "$MODEL" "$METHOD" "$NPROC" "$MICRO_BSZ"

  conda run --no-capture-output -n prift-eval-qwen3 \
    bash scripts/eval_math.sh \
    --model "verl/checkpoints/numina-cot-${METHOD}-${SHORT}" \
    --output-dir "outputs/${METHOD}-${SHORT}" \
    --prompt-type "$PROMPT_TYPE" \
    --n-sampling "$NSAMP" \
    --temperature "$TEMPERATURE" \
    --cuda-visible-devices "$EVAL_CUDA_DEVICES"
done
