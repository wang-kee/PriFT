#!/bin/bash
# Evaluate a checkpoint on the five math benchmarks with 16 stochastic samples,
# reporting Avg@16 and Pass@16 for each benchmark (see summary_metrics.json).
#
# Usage:
#   bash scripts/eval_math.sh \
#     --model CHECKPOINT_OR_MODEL \
#     --output-dir OUTPUT_DIR \
#     [--prompt-type qwen-boxed] \
#     [--n-sampling 16] \
#     [--temperature 1] \
#     [--tokenizer-mode slow] \
#     [--cuda-visible-devices 0,1,2,3]
#
# Backward-compatible positional usage is also supported:
#   bash scripts/eval_math.sh MODEL OUTPUT_DIR [PROMPT_TYPE] [N_SAMPLING] [TEMPERATURE]
set -ex

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}/math_evaluation"

resolve_model_path() {
    local path="$1"
    if [[ -d "${path}" && ! -f "${path}/config.json" ]]; then
        local latest
        latest=$(find "${path}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)
        if [[ -n "${latest}" && -f "${latest}/config.json" ]]; then
            echo "${latest}"
            return 0
        fi
    fi
    echo "${path}"
}

MODEL=""
OUTPUT_DIR=""
PROMPT_TYPE="qwen-boxed"
N_SAMPLING="16"
TEMPERATURE="1"
TOKENIZER_MODE="slow"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ $# -gt 0 && "$1" == --* ]]; then
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                MODEL="$2"; shift 2 ;;
            --output-dir)
                OUTPUT_DIR="$2"; shift 2 ;;
            --prompt-type)
                PROMPT_TYPE="$2"; shift 2 ;;
            --n-sampling)
                N_SAMPLING="$2"; shift 2 ;;
            --temperature)
                TEMPERATURE="$2"; shift 2 ;;
            --tokenizer-mode)
                TOKENIZER_MODE="$2"; shift 2 ;;
            --cuda-visible-devices|--gpus)
                CUDA_DEVICES="$2"; shift 2 ;;
            *)
                echo "Unknown argument: $1" >&2
                exit 2 ;;
        esac
    done
else
    MODEL=${1:?"MODEL (checkpoint path) required"}
    OUTPUT_DIR=${2:?"OUTPUT_DIR required"}
    PROMPT_TYPE=${3:-qwen-boxed}
    N_SAMPLING=${4:-16}
    TEMPERATURE=${5:-1}
fi

MODEL=$(resolve_model_path "${MODEL}")
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

DATA_NAME="math_oai,minerva_math,olympiadbench,aime24,amc23"

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path "${MODEL}" \
    --data_names "${DATA_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --split test \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample -1 \
    --seed 0 \
    --temperature "${TEMPERATURE}" \
    --n_sampling "${N_SAMPLING}" \
    --tokenizer_mode "${TOKENIZER_MODE}" \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm

echo "Avg@${N_SAMPLING} / Pass@${N_SAMPLING} summary: ${OUTPUT_DIR}/summary_metrics.json"
