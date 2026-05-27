#!/bin/bash
# Train a math-reasoning SFT model with a selectable token-weighting method.
#
# Usage:
#   bash scripts/train.sh MODEL METHOD [NPROC] [MICRO_BSZ] [REF_MICRO_BSZ] [REFERENCE_MODEL]
#
#   MODEL          HF id or local path, e.g. Qwen/Qwen2.5-Math-1.5B
#   METHOD         sft | dft | prift_prob | prift_mass
#   NPROC          GPUs per node (default 4)
#   MICRO_BSZ      online micro batch size per GPU (default 4)
#   REF_MICRO_BSZ  reference forward micro batch size (default 1; prift_* only)
#   REFERENCE_MODEL optional frozen reference model path/id (default MODEL)
#
# Examples:
#   bash scripts/train.sh Qwen/Qwen2.5-Math-1.5B prift_mass 4
#   bash scripts/train.sh Qwen/Qwen2.5-Math-7B   prift_prob 4 2 1
#   bash scripts/train.sh Qwen/Qwen3-8B-Base     prift_mass 4 2 1
set -ex

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}/verl"

MODEL=${1:?"MODEL required, e.g. Qwen/Qwen2.5-Math-1.5B"}
METHOD=${2:?"METHOD required: sft|dft|prift_prob|prift_mass"}
NPROC=${3:-4}
MICRO_BSZ=${4:-4}
REF_MICRO_BSZ=${5:-1}
REFERENCE_MODEL=${6:-${REFERENCE_MODEL:-${MODEL}}}
MASS_THRESHOLD=${PRIFT_MASS_THRESHOLD:-0.5}

SHORT=$(basename "${MODEL}")
PROJECT_NAME=prift-numina-cot
EXPERIMENT_NAME=numina-cot-${METHOD}-${SHORT}
SAVE_PATH=${SAVE_PATH:-checkpoints/${EXPERIMENT_NAME}}

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    -m verl.trainer.fsdp_dft_trainer \
    data.train_files=data/numina_cot/train.parquet \
    data.val_files=data/math500/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.train_batch_size=256 \
    data.max_length=2048 \
    data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
    optim.lr=5e-5 \
    model.partial_pretrain="${MODEL}" \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    loss.method="${METHOD}" \
    loss.reference_model_path="${REFERENCE_MODEL}" \
    loss.mass_threshold="${MASS_THRESHOLD}" \
    loss.reference_micro_batch_size="${REF_MICRO_BSZ}" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true

echo "Checkpoints saved under: ${REPO_ROOT}/verl/${SAVE_PATH}"
