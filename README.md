<div align="center">

# PriFT: Prior-Support Guided Supervised Fine-Tuning

</div>

This is the official repository for "PriFT: Prior-Support Guided Supervised Fine-Tuning" by Ke Wang*, Shuangqi Li*, Mathieu Salzmann, and Pascal Frossard.

PriFT is a token-reweighted supervised fine-tuning framework that derives token
weights from a **frozen pretrained reference model** instead of the online model
being optimized. The pretrained reference gives a *prior-support* signal that is
decoupled from the optimization trajectory.

This repository is based on [DFT](https://github.com/yongliang-wu/DFT). 

## Methods

The training loss is selected through the `loss.method` config field:

| `loss.method` | Per-token weight | Source |
|---------------|------------------------|--------|
| `sft`         | `1`                    | —      |
| `dft`         | `sg(p_online(y_t))`    | online model |
| `prift_prob`  | `p_ref(y_t)`           | frozen pretrained reference |
| `prift_mass`  | `1[ u_t >= 0.5 ]`        | frozen pretrained reference |

where, under the frozen reference distribution `p_ref(· | x, y_<t)`,

```
u_t = sum_{v : p_ref(v) <= p_ref(y_t)} p_ref(v)
```

## Configuration

New fields in [`verl/trainer/config/sft_trainer.yaml`](verl/verl/trainer/config/sft_trainer.yaml):

```yaml
loss:
  method: dft                 # {sft, dft, prift_prob, prift_mass}
  reference_model_path: null  # defaults to model.partial_pretrain when null (the pretrained model in our paper)
```
(No need to manually configure them if you use our scripts below)

## Environments

```bash
cd Path-to-Project
bash envs/install_prift_envs.sh
```

This creates the three pinned reproduction environments:

- `prift-train-qwen` (training env, based on suggested setup from [DFT repo](https://github.com/yongliang-wu/DFT/blob/master/verl/scripts/install_vllm_sglang_mcore.sh))
- `prift-eval-qwen25` (eval env, based on suggested setup from [Qwen2.5-math repo](https://github.com/QwenLM/Qwen2.5-Math))
- `prift-eval-qwen3` (eval env, based on suggested setup from [Qwen3 repo](https://github.com/QwenLM/Qwen3))

Training uses `verl`. Please use `prift-eval-qwen25` for evaluating Qwen2.5 models and `prift-eval-qwen3` for evaluating Qwen3 models.

## Getting started

### Step1: Prepare data:

```bash
conda activate prift-train-qwen
bash scripts/prepare_data.sh 100000
```

This generates:

- `verl/data/numina_cot/train.parquet`
- `verl/data/math500/test.parquet`

### Step2: Launch training:

Please use the `prift-train-qwen` environment for training. The `METHOD` argument selects the token weighting rule:

- `sft`: standard supervised fine-tuning
- `dft`: online-model probability weighting
- `prift_prob`: pretrained-reference probability weighting
- `prift_mass`: pretrained-reference cumulative-mass thresholding

```bash
conda activate prift-train-qwen

NPROC=4
MICRO_BSZ=4
MODEL=Qwen/Qwen2.5-Math-7B
METHOD=prift_prob
REFERENCE_MODEL=Qwen/Qwen2.5-Math-7B

bash scripts/train.sh $MODEL $METHOD $NPROC $MICRO_BSZ $REFERENCE_MODEL
```

Checkpoints are written under `verl/checkpoints/numina-cot-${METHOD}-$(basename ${MODEL})`.

### Step3: Evaluation:

Please use `prift-eval-qwen25` for `Qwen2.5` models and `prift-eval-qwen3` for `Qwen3-8B-Base`. The evaluation script resolves the latest `global_step_*` checkpoint automatically and writes `summary_metrics.json` to the output directory.

```bash
conda activate prift-eval-qwen25

PROMPT_TYPE="qwen-boxed"
CUDA_VISIBLE_DEVICES=0
N_SAMPLING=16
TEMPERATURE=1
MODEL_NAME_OR_PATH=verl/checkpoints/numina-cot-prift_prob-Qwen2.5-Math-7B
OUTPUT_DIR=outputs/prift_prob-Qwen2.5-Math-7B

bash scripts/eval_math.sh \
  --model $MODEL_NAME_OR_PATH \
  --output-dir $OUTPUT_DIR \
  --prompt-type $PROMPT_TYPE \
  --n-sampling $N_SAMPLING \
  --temperature $TEMPERATURE \
  --cuda-visible-devices $CUDA_VISIBLE_DEVICES
```


We also provide wrapper scripts to reproduce the PriFT runs used in the paper. By default these wrappers run `prift_prob` and `prift_mass`:

```bash
bash scripts/reproduce_qwen25_math_1p5b.sh 4
bash scripts/reproduce_qwen25_math_7b.sh 4
bash scripts/reproduce_qwen3_8b_base.sh 4
```

## Acknowledgements

This codebase builds directly on DFT. We thank the authors for providing the repo. 

## Citation

```bibtex
@article{wang2026prift,
  title={PriFT: Prior-Support Guided Supervised Fine-Tuning},
  author={Wang, Ke and Li, Shuangqi and Salzmann, Mathieu and Frossard, Pascal},
  journal={arXiv preprint arXiv:2606.09396},
  year={2026}
}
```
