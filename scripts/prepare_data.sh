#!/bin/bash
# Prepare the math SFT training/validation parquet files used by PriFT.
#
# Usage:
#   bash scripts/prepare_data.sh [NUM_TRAIN]
#
# NUM_TRAIN defaults to 100000 for the NuminaMath-CoT training subset.
set -ex

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUM_TRAIN=${1:-100000}

cd "${REPO_ROOT}/verl"
python examples/data_preprocess/numina_cot.py --train_end "${NUM_TRAIN}"
python examples/data_preprocess/math_dataset.py
