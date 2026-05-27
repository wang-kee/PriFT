# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure token-weighting utilities for DFT and PriFT.

These are factored out of the trainer so the weighting math can be unit-tested
without any distributed/model-loading machinery. All ``*_logits`` arguments are
flattened to ``(N, V)`` and aligned 1:1 with the shifted CE targets
``shift_labels`` of shape ``(N,)``.

The PriFT helpers expose the frozen-reference statistics used by the trainer:

- ``q_t(y_t)`` via :func:`reference_target_probs`
- full ``q_t(v)`` via :func:`reference_full_probs`
- ``lower_mass_t = sum_v q_t(v) 1[q_t(v) < q_t(y_t)]`` via
  :func:`reference_lower_mass`
- ``u_t = lower_mass_t + q_t(y_t)`` via :func:`reference_mass`
"""

import torch


def dft_weights(shift_logits, shift_labels):
    """DFT: m_t = sg(p_online(y_t)), the detached online target-token probability."""
    probs = torch.softmax(shift_logits, dim=-1)
    return probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()


def reference_full_probs(ref_logits):
    """Return the frozen-reference distribution q_t(v) for every token position."""
    return torch.softmax(ref_logits.float(), dim=-1)


def reference_target_probs(ref_logits, shift_labels):
    """Return q_t(y_t), the frozen-reference target-token probability."""
    probs = reference_full_probs(ref_logits)
    return probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)


def prift_prob_weights(ref_logits, shift_labels):
    """PriFT-prob: m_t = p_ref(y_t), the frozen-reference target-token probability.

    The softmax is taken in fp32 for a numerically stable target probability.
    """
    return reference_target_probs(ref_logits, shift_labels).detach()


def reference_lower_mass(ref_logits, shift_labels, vocab_chunk=None):
    """Return lower_mass_t = sum_v q_t(v) 1[q_t(v) < q_t(y_t)].

    Computed in fp32 with a shared log-sum-exp shift for stability. Because every
    term is divided by the same positive normalizer, the strict comparison
    ``q_t(v) < q_t(y_t)`` is equivalent to ``exp(l_v - max) < exp(l_yt - max)``.
    When ``vocab_chunk`` is provided, the computation is streamed over the vocab
    dimension to bound peak memory.
    """
    N, V = ref_logits.shape
    target_logit = ref_logits.gather(1, shift_labels.unsqueeze(-1)).float()  # (N, 1)
    max_logit = ref_logits.max(dim=-1, keepdim=True).values.float()  # (N, 1)
    target_e = torch.exp(target_logit - max_logit)  # (N, 1)

    chunk = vocab_chunk or V
    chunk = max(1, int(chunk))
    denom = torch.zeros((N, 1), device=ref_logits.device, dtype=torch.float32)
    masked = torch.zeros((N, 1), device=ref_logits.device, dtype=torch.float32)
    for c in range(0, V, chunk):
        e = torch.exp(ref_logits[:, c : c + chunk].float() - max_logit)  # (N, chunk)
        denom += e.sum(dim=-1, keepdim=True)
        masked += torch.where(e < target_e, e, torch.zeros_like(e)).sum(dim=-1, keepdim=True)
    return (masked / denom).squeeze(-1)  # (N,)


def reference_mass(ref_logits, shift_labels, vocab_chunk=None):
    """Return PriFT-mass's score u_t = lower_mass_t + q_t(y_t)."""
    lower_mass = reference_lower_mass(ref_logits, shift_labels, vocab_chunk=vocab_chunk)
    target_prob = reference_target_probs(ref_logits, shift_labels)
    return lower_mass + target_prob


def prift_mass_weights(ref_logits, shift_labels, threshold=0.5, vocab_chunk=None):
    """PriFT-mass: m_t = 1[ lower_mass_t + q_t(y_t) >= threshold ]."""
    score = reference_mass(ref_logits, shift_labels, vocab_chunk=vocab_chunk)
    return (score >= threshold).to(torch.float32).detach()
