"""Phase 2: Surrogate-force HMC using canonical batch-fused LEDH target.

Uses the proper batch-native NeuTra target API instead of the single-cloud lane.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import json
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / ".claude/worktrees/ledh-canonical-rebuild"))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    make_canonical_neutra_target,
)

DTYPE = tf.float64
THETA_TRUE = np.array([0.72, 0.55, 0.35, 0.35, 0.45])

print("=" * 80)
print("Phase 2: Surrogate-Force HMC (Batch-Native LEDH)")
print("=" * 80)
print()

# Build canonical NeuTra target (batch-native, fused)
print("Building canonical LEDH target...")
t0 = time.time()
target = make_canonical_neutra_target(
    "lgssm",
    particle_count=1008,
    noise_seed=140000,
    substeps=12,
)
print(f"  Built in {time.time()-t0:.3f}s")
print(f"  Model: {target.model_id}")
print(f"  Data: {target.data_id}")
print()

# Test batch evaluation
print("Testing batch evaluation...")
theta_batch = tf.constant(np.stack([THETA_TRUE, THETA_TRUE]), DTYPE)  # [2, 5]
directions = tf.eye(5, dtype=DTYPE)
directions_batch = tf.stack([directions[0], directions[1]])  # [2, 5]

t0 = time.time()
values, scores, diagnostics = target.batch_value_score(theta_batch, directions_batch)
t1 = time.time()

print(f"  Batch call completed in {t1-t0:.3f}s")
print(f"  Values shape: {values.shape}")
print(f"  Scores shape: {scores.shape}")
print(f"  Values: {values.numpy()}")
print(f"  Scores: {scores.numpy()}")
print()

# Now build surrogate-force adapter
print("Building surrogate-force dual adapter...")

# The adapter: value uses exact config, gradient uses damped config
# But we need to call batch_value_score with different configs...
# Let me check if we can pass config parameters

print("\nChecking batch_value_score signature...")
import inspect
sig = inspect.signature(target.batch_value_score)
print(f"  Signature: {sig}")
print()

# For now, test if HMC can use the target directly
print("Testing HMC initialization with exact target...")

def log_prob_fn(theta):
    """Target log probability for HMC."""
    # HMC needs log prob, we have negative log likelihood
    # batch_value_score returns (values, scores, diagnostics)
    # We need just the value

    # Handle both batched and unbatched
    if len(theta.shape) == 1:
        theta_batch = theta[None, :]  # [1, 5]
    else:
        theta_batch = theta

    # Dummy directions (not used for value-only)
    directions = tf.zeros_like(theta_batch)

    values, _, _ = target.batch_value_score(theta_batch, directions)

    if len(theta.shape) == 1:
        return values[0]
    else:
        return values

# Test
print("  Single theta evaluation...")
t0 = time.time()
val = log_prob_fn(tf.constant(THETA_TRUE, DTYPE))
print(f"  Time: {time.time()-t0:.3f}s")
print(f"  Value: {float(val.numpy()):.6f}")

print("\n  Batched theta evaluation (2 chains)...")
t0 = time.time()
vals = log_prob_fn(theta_batch)
print(f"  Time: {time.time()-t0:.3f}s")
print(f"  Values: {vals.numpy()}")

print("\nDone - batch-native target works!")
print()
print("Next step: Implement dual adapter using batch_value_score")
print("  with different ridge/damping configs for value vs gradient")
