#!/bin/bash
# End-to-end math reproduction for the three target backbones with both PriFT
# variants.
#
# Usage:  bash scripts/reproduce_all.sh [NPROC]
#
# Adjust MICRO_BSZ / REF_MICRO per model below if you hit OOM.
set -ex

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NPROC=${1:-4}

bash "${REPO_ROOT}/scripts/reproduce_qwen25_math_1p5b.sh" "${NPROC}"
bash "${REPO_ROOT}/scripts/reproduce_qwen25_math_7b.sh" "${NPROC}"
bash "${REPO_ROOT}/scripts/reproduce_qwen3_8b_base.sh" "${NPROC}"
