#!/bin/bash
set -ex

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NPROC=${1:-4}
MODEL="Qwen/Qwen2.5-Math-1.5B"
MICRO_BSZ=4
REF_MICRO_BSZ=2
PROMPT_TYPE="qwen-boxed"

for METHOD in prift_prob prift_mass; do
  SHORT=$(basename "${MODEL}")
  CKPT_ROOT="${REPO_ROOT}/verl/checkpoints/numina-cot-${METHOD}-${SHORT}"
  OUTPUT_DIR="${REPO_ROOT}/outputs/${METHOD}-${SHORT}"

  bash "${REPO_ROOT}/scripts/train.sh" "${MODEL}" "${METHOD}" "${NPROC}" "${MICRO_BSZ}" "${REF_MICRO_BSZ}"
  bash "${REPO_ROOT}/scripts/eval_math.sh" \
    --model "${CKPT_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --prompt-type "${PROMPT_TYPE}" \
    --n-sampling 16 \
    --temperature 1
done
