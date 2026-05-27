<div align="center">

# PriFT: Prior-Support Guided Supervised Fine-Tuning

</div>

PriFT is a token-reweighted supervised fine-tuning framework that derives token
weights from a **frozen pretrained reference model** instead of the online model
being optimized. The pretrained reference gives a *prior-support* signal that is
decoupled from the optimization trajectory, avoiding the self-reinforcing bias of
online reweighting methods such as DFT.

This repository is a fork of **DFT**
([On the Generalization of SFT](https://github.com/yongliang-wu/DFT)), which is in
turn built on [verl](https://github.com/volcengine/verl) for training and
[Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) for evaluation. We keep the
original SFT and DFT training paths fully runnable and add the PriFT methods on top.

## Methods

The training loss is selected through the `loss.method` config field:

| `loss.method` | Per-token weight `m_t` | Source |
|---------------|------------------------|--------|
| `sft`         | `1`                    | —      |
| `dft`         | `sg(p_online(y_t))`    | online model |
| `prift_prob`  | `p_ref(y_t)`           | frozen pretrained reference |
| `prift_mass`  | `1[ u_t >= τ ]`        | frozen pretrained reference |

where, under the frozen reference distribution `p_ref(· | x, y_<t)`,

```
lower_mass_t = sum_{v : p_ref(v) < p_ref(y_t)} p_ref(v)
u_t = lower_mass_t + p_ref(y_t)
```

and `τ = loss.mass_threshold` (default `0.5`).

The reference model is run **online** on every training batch (eval-only, `no_grad`,
detached, never optimized); PriFT does **not** precompute weights offline.

## Configuration

New fields in [`verl/trainer/config/sft_trainer.yaml`](verl/verl/trainer/config/sft_trainer.yaml):

```yaml
loss:
  method: dft                 # {sft, dft, prift_prob, prift_mass}
  reference_model_path: null  # defaults to model.partial_pretrain when null
  mass_threshold: 0.5         # selection threshold for prift_mass
  reference_micro_batch_size: null  # optional micro-batching for the reference forward
```

All original DFT codepaths remain in the repository. PriFT adds new loss modes on
top of the original SFT/DFT trainer instead of replacing the upstream codebase.

## Installation

```bash
conda create -n prift python=3.10 -y
conda activate prift
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

## Quick start

See [`README_REPRODUCE.md`](README_REPRODUCE.md) for exact, end-to-end commands to
reproduce the mathematical-reasoning results on `Qwen2.5-Math-1.5B`,
`Qwen2.5-Math-7B`, and `Qwen3-8B-Base`.

## Acknowledgements

This codebase builds directly on DFT and verl. Please also cite the DFT paper if
you use this code:

```latex
@article{wu2025generalization,
  title={On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification},
  author={Wu, Yongliang and Zhou, Yizhou and Ziheng, Zhou and Peng, Yingzhe and Ye, Xinyu and Hu, Xinting and Zhu, Wenbo and Qi, Lu and Yang, Ming-Hsuan and Yang, Xu},
  journal={arXiv preprint arXiv:2508.05629},
  year={2025}
}
```
