"""File-based PriFT trainer smoke check.

This avoids importing the full training stack and instead verifies that the
repo-local trainer and weight modules exist, expose the expected symbols, and
contain the teacher-forced reference-forward hooks in source.
"""

import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER_FILE = os.path.join(REPO_ROOT, "verl", "verl", "trainer", "fsdp_dft_trainer.py")
WEIGHTS_FILE = os.path.join(REPO_ROOT, "verl", "verl", "trainer", "prift_weights.py")

print("TRAINER_FILE=" + TRAINER_FILE)
assert os.path.isfile(TRAINER_FILE), TRAINER_FILE
print("PRIFT_WEIGHTS_FILE=" + WEIGHTS_FILE)
assert os.path.isfile(WEIGHTS_FILE), WEIGHTS_FILE

# Import the pure weight module directly from file; this only depends on torch.
spec = importlib.util.spec_from_file_location("prift_weights_local", WEIGHTS_FILE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

for fn in (
    "dft_weights",
    "reference_full_probs",
    "reference_target_probs",
    "reference_lower_mass",
    "reference_mass",
    "prift_prob_weights",
    "prift_mass_weights",
):
    assert hasattr(module, fn), "missing " + fn
    print("HAS " + fn)

src = open(TRAINER_FILE).read()
for hook in ("_build_reference_model", "_reference_weights", "_token_weights", '"method", "dft"'):
    assert hook in src, "missing " + hook
for pattern in (
    "targets = input_ids[:, 1:]",
    "input_ids=input_ids[s:e]",
    "attention_mask=attention_mask[s:e]",
    "position_ids=position_ids[s:e]",
    "use_cache=False",
):
    assert pattern in src, "missing teacher-forced reference pattern: " + pattern

print("OK: trainer and PriFT helpers are present in repo source.")
