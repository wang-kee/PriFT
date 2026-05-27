"""Unit tests for the PriFT token-weighting math.

Runnable on CPU with only torch installed:

    python tests/test_prift_weights.py

It imports the *real* functions used by the trainer from
verl/verl/trainer/prift_weights.py so the math is tested directly.
"""

import os
import sys

import torch

# Import the pure weight module directly (no verl/FSDP needed).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl", "verl", "trainer"))
import prift_weights as pw  # noqa: E402


def test_prift_prob_matches_softmax():
    torch.manual_seed(0)
    N, V = 17, 5000
    logits = torch.randn(N, V)
    labels = torch.randint(0, V, (N,))
    w = pw.prift_prob_weights(logits, labels)
    ref = torch.softmax(logits.float(), dim=-1).gather(1, labels.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(w, ref, atol=1e-5), (w - ref).abs().max()
    assert not w.requires_grad


def _naive_lower_mass(logits, labels):
    probs = torch.softmax(logits.float(), dim=-1)
    target_p = probs.gather(1, labels.unsqueeze(-1))  # (N,1)
    return torch.where(probs < target_p, probs, torch.zeros_like(probs)).sum(dim=-1)


def test_reference_stats_match_naive_and_chunking():
    torch.manual_seed(1)
    N, V = 23, 4096
    logits = torch.randn(N, V) * 2.0
    labels = torch.randint(0, V, (N,))
    probs = torch.softmax(logits.float(), dim=-1)
    target = probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
    naive_lower_mass = _naive_lower_mass(logits, labels)
    full_lower_mass = pw.reference_lower_mass(logits, labels, vocab_chunk=None)
    chunked_lower_mass = pw.reference_lower_mass(logits, labels, vocab_chunk=257)
    assert torch.allclose(full_lower_mass, naive_lower_mass, atol=1e-5), (
        full_lower_mass - naive_lower_mass
    ).abs().max()
    assert torch.allclose(chunked_lower_mass, full_lower_mass, atol=1e-5), (
        chunked_lower_mass - full_lower_mass
    ).abs().max()

    full_probs = pw.reference_full_probs(logits)
    target_probs = pw.reference_target_probs(logits, labels)
    assert torch.allclose(full_probs.sum(dim=-1), torch.ones(N), atol=1e-5)
    assert torch.allclose(target_probs, target, atol=1e-5)

    score = pw.reference_mass(logits, labels)
    assert torch.allclose(score, naive_lower_mass + target, atol=1e-5)
    assert (score > 0).all() and (score <= 1.0 + 1e-5).all()


def test_prift_mass_thresholding():
    torch.manual_seed(3)
    N, V = 50, 2000
    logits = torch.randn(N, V)
    labels = torch.randint(0, V, (N,))
    score = pw.reference_mass(logits, labels)
    for tau in (0.0, 0.3, 0.5, 0.9):
        mask = pw.prift_mass_weights(logits, labels, threshold=tau)
        expected = (score >= tau).float()
        assert torch.equal(mask, expected)
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})


def test_dft_weights_detached():
    torch.manual_seed(4)
    N, V = 10, 100
    logits = torch.randn(N, V, requires_grad=True)
    labels = torch.randint(0, V, (N,))
    w = pw.dft_weights(logits, labels)
    assert not w.requires_grad
    ref = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(w, ref.detach(), atol=1e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
