# Reproducing PriFT math-reasoning results

End-to-end commands to reproduce the supervised fine-tuning results for
`Qwen2.5-Math-1.5B`, `Qwen2.5-Math-7B`, and `Qwen3-8B-Base`. All training uses an
**online frozen reference** (no offline precomputation): on every batch the frozen
pretrained model is run under `no_grad` to produce the PriFT token weights/masks.

Tested on H100/H200 nodes (4 GPUs per run), `python 3.10`, `torch 2.6.0+cu124`.

## 1. Environment setup

```bash
conda create -n prift python=3.10 -y
conda activate prift

# Training stack (verl + vLLM/SGLang/Megatron-core deps)
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ..

# Evaluation stack (Qwen2.5-Math math eval)
pip install -r math_evaluation/requirements.txt
```

Optional sanity checks (CPU, torch only):

```bash
python tests/test_prift_weights.py
cd verl && python ../tests/check_trainer_import.py
```

## 2. Data preparation

```bash
# 100k NuminaMath-CoT for training + MATH500 for the trainer's val loss
bash scripts/prepare_data.sh 100000
```

Outputs:
- `verl/data/numina_cot/train.parquet`
- `verl/data/math500/test.parquet`

The five evaluation benchmarks (`math_oai`, `minerva_math`, `olympiadbench`,
`aime24`, `amc23`) ship under `math_evaluation/data/`.

## 3. Training (PriFT-prob / PriFT-mass)

`scripts/train.sh MODEL METHOD [NPROC] [MICRO_BSZ] [REF_MICRO_BSZ]`

```bash
# Qwen2.5-Math-1.5B
bash scripts/train.sh Qwen/Qwen2.5-Math-1.5B prift_prob 4 4 2
bash scripts/train.sh Qwen/Qwen2.5-Math-1.5B prift_mass 4 4 2

# Qwen2.5-Math-7B
bash scripts/train.sh Qwen/Qwen2.5-Math-7B   prift_prob 4 2 1
bash scripts/train.sh Qwen/Qwen2.5-Math-7B   prift_mass 4 2 1

# Qwen3-8B-Base
bash scripts/train.sh Qwen/Qwen3-8B-Base     prift_prob 4 2 1
bash scripts/train.sh Qwen/Qwen3-8B-Base     prift_mass 4 2 1
```

Baselines for comparison use the same script with `METHOD=sft` or `METHOD=dft`
(no reference model is loaded for those). Key settings (paper §5.1): 100k
NuminaMath-CoT, 1 epoch, `train_batch_size=256`, `max_length=2048`, `lr=5e-5`,
bf16, `mass_threshold=0.5`. Reduce `MICRO_BSZ` / `REF_MICRO_BSZ` if you hit OOM;
for very large vocabularies you can also set `loss.mass_vocab_chunk` to chunk the
PriFT-mass softmax.

Checkpoints are written to `verl/checkpoints/numina-cot-<method>-<model>/global_step_*`.

## 4. Evaluation (Avg@16 + Pass@16)

`scripts/eval_math.sh MODEL OUTPUT_DIR [PROMPT_TYPE] [N_SAMPLING] [TEMPERATURE]`

```bash
bash scripts/eval_math.sh \
  --model verl/checkpoints/numina-cot-prift_mass-Qwen2.5-Math-7B \
  --output-dir outputs/prift_mass-Qwen2.5-Math-7B \
  --prompt-type qwen-boxed \
  --n-sampling 16 \
  --temperature 1 \
  --cuda-visible-devices 0,1,2,3
```

Evaluation uses 16 stochastic samples (`temperature=1`, `top_p=1`). Per-benchmark
`Avg@16`/`Pass@16` are printed and saved to
`<OUTPUT_DIR>/summary_metrics.json`; per-benchmark detail is in
`<OUTPUT_DIR>/<dataset>_metrics.json`.

To run everything (train + eval, all three models, both variants):

```bash
bash scripts/prepare_data.sh 100000
bash scripts/reproduce_qwen25_math_1p5b.sh 4
bash scripts/reproduce_qwen25_math_7b.sh 4
bash scripts/reproduce_qwen3_8b_base.sh 4
# or:
bash scripts/reproduce_all.sh 4
```

## 5. Expected results (average over the five benchmarks)

From the PriFT paper (Table 3 for 7B/8B, Table 10 for 1.5B). Numbers are the
five-benchmark average; expect run-to-run variation of roughly ±1 point given
stochastic decoding.

| Model | Method | Avg@16 | Pass@16 |
|-------|--------|:------:|:-------:|
| Qwen2.5-Math-1.5B | PriFT-prob | 31.38 | 63.71 |
| Qwen2.5-Math-1.5B | PriFT-mass | 32.69 | 64.52 |
| Qwen2.5-Math-7B   | PriFT-prob | 39.65 | 68.62 |
| Qwen2.5-Math-7B   | PriFT-mass | 40.67 | 69.60 |
| Qwen3-8B-Base     | PriFT-prob | 36.65 | 63.77 |
| Qwen3-8B-Base     | PriFT-mass | 39.21 | 65.58 |

Reference baselines on Qwen2.5-Math-7B: SFT 23.54 / 55.52, DFT 35.89 / 53.86
(Avg@16 / Pass@16). The key PriFT signal is the large Pass@16 gain over DFT from
moving the reweighting source from the online model to the frozen reference.
