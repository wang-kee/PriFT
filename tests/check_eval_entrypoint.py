"""Lightweight source-level smoke check for scripts/eval_math.sh.

This avoids importing vLLM or running any model while still verifying that the
public evaluation entrypoint exposes the intended arguments and checkpoint
resolution behavior.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script = os.path.join(REPO_ROOT, "scripts", "eval_math.sh")
src = open(script).read()

for token in (
    "--model",
    "--output-dir",
    "--prompt-type",
    "--n-sampling",
    "--temperature",
    "--cuda-visible-devices",
    "resolve_model_path",
    "global_step_",
    "summary_metrics.json",
):
    assert token in src, f"missing eval entrypoint token: {token}"

print("OK: eval entrypoint exposes explicit args and checkpoint resolution.")
