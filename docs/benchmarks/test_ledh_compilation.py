"""Test LEDH filter compilation time."""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import time
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / ".claude/worktrees/ledh-canonical-rebuild"))

import tensorflow as tf
from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import diagonal_lgssm_any_dim

DTYPE = tf.float64
DIM = 3
HORIZON = 50
PARTICLES = 1008

print("LEDH filter compilation test")
print("=" * 60)

# Setup
theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)
obs_matrix = tf.eye(DIM, dtype=DTYPE)
observations = tf.constant(np.random.randn(HORIZON, DIM), DTYPE)

# Pre-compute constants
sqrt_d = np.sqrt(DIM)
basis_np = np.array([[sqrt_d, 0, 0], [-sqrt_d, 0, 0], [0, sqrt_d, 0],
                     [0, -sqrt_d, 0], [0, 0, sqrt_d], [0, 0, -sqrt_d]])
reset_design = tf.constant(np.tile(basis_np, (PARTICLES // 6, 1)), DTYPE)
initial_covs = tf.constant(np.stack([np.eye(DIM)] * PARTICLES), DTYPE)
initial_noise = tf.constant(np.random.randn(PARTICLES, DIM), DTYPE)
transition_noise = tf.constant(np.random.randn(HORIZON, PARTICLES, DIM), DTYPE)

print("\nTest: Single LEDH filter call (value only)")
model, _ = diagonal_lgssm_any_dim(theta, dim=DIM, obs_matrix=obs_matrix)

print("  Calling canonical_value_and_analytical_score...")
t0 = time.time()
value, _ = canonical_value_and_analytical_score(
    model, theta, initial_noise, initial_covs, transition_noise, observations,
    flow_substeps=12, with_score=False,
    reset_policy="contract_e", reset_design=reset_design,
    reset_epsilon=1.0, reset_sinkhorn_steps=8, reset_balance_steps=8,
    reset_ridge=1e-5, correction_steps=1, correction_lm_damping=1e-5,
    pairwise_steps=1, annealed_stages=1, annealed_seed=17,
)
t1 = time.time()
print(f"  First call: {t1-t0:.3f}s")
print(f"  Value: {float(value.numpy()):.6f}")

print("\n  Second call (should be cached)...")
t0 = time.time()
value2, _ = canonical_value_and_analytical_score(
    model, theta, initial_noise, initial_covs, transition_noise, observations,
    flow_substeps=12, with_score=False,
    reset_policy="contract_e", reset_design=reset_design,
    reset_epsilon=1.0, reset_sinkhorn_steps=8, reset_balance_steps=8,
    reset_ridge=1e-5, correction_steps=1, correction_lm_damping=1e-5,
    pairwise_steps=1, annealed_stages=1, annealed_seed=17,
)
t1 = time.time()
print(f"  Second call: {t1-t0:.3f}s")

print("\nTest: LEDH with score (1 direction)")
print("  Building model with direction...")
model2, set_dir = diagonal_lgssm_any_dim(theta, dim=DIM, obs_matrix=obs_matrix)
set_dir(tf.constant([1.0, 0, 0, 0, 0], DTYPE))

print("  Calling with with_score=True...")
t0 = time.time()
value3, score = canonical_value_and_analytical_score(
    model2, theta, initial_noise, initial_covs, transition_noise, observations,
    flow_substeps=12, with_score=True,
    reset_policy="contract_e", reset_design=reset_design,
    reset_epsilon=1.0, reset_sinkhorn_steps=8, reset_balance_steps=8,
    reset_ridge=1e-5, correction_steps=1, correction_lm_damping=1e-5,
    pairwise_steps=1, annealed_stages=1, annealed_seed=17,
)
t1 = time.time()
print(f"  Time: {t1-t0:.3f}s")
print(f"  Score: {float(score[0].numpy()):.6f}")

print("\nDone")
