#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

cd "$(cd "$(dirname "$0")/.." && pwd)"

FLASH_ATTN_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_ATTN_WHL}"
FLASH_ATTN_TMPDIR="$(mktemp -d)"
trap 'rm -rf "${FLASH_ATTN_TMPDIR}"' EXIT

conda env create -f envs/prift-train-qwen.yml
conda env create -f envs/prift-eval-qwen25.yml
conda env create -f envs/prift-eval-qwen3.yml

wget -nv -O "${FLASH_ATTN_TMPDIR}/${FLASH_ATTN_WHL}" "${FLASH_ATTN_URL}"
conda run --no-capture-output -n prift-train-qwen   pip install --no-cache-dir --no-deps "${FLASH_ATTN_TMPDIR}/${FLASH_ATTN_WHL}"
conda run --no-capture-output -n prift-train-qwen   pip install --no-deps -e ./verl
