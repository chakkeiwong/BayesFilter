"""Minimal test to isolate the compilation performance issue."""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / ".claude/worktrees/ledh-canonical-rebuild"))

import tensorflow as tf
from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import diagonal_lgssm_any_dim

DTYPE = tf.float64

print("Minimal compilation test")
print("=" * 60)

# Test 1: Build model once
print("\nTest 1: Build model (outside @tf.custom_gradient)")
t0 = time.time()
theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)
obs_matrix = tf.eye(3, dtype=DTYPE)
model, set_dir = diagonal_lgssm_any_dim(theta, dim=3, obs_matrix=obs_matrix)
t1 = time.time()
print(f"  Time: {t1-t0:.3f}s")

# Test 2: Build model inside custom_gradient
print("\nTest 2: Build model inside @tf.custom_gradient")

@tf.custom_gradient
def test_fn(theta_inner):
    print("  [tracing forward]")
    model_fwd, _ = diagonal_lgssm_any_dim(theta_inner, dim=3, obs_matrix=obs_matrix)
    value = tf.reduce_sum(theta_inner)  # Dummy

    def grad_fn(upstream):
        print("  [tracing gradient]")
        # This is the problem: building model in gradient
        model_grad, set_dir_grad = diagonal_lgssm_any_dim(theta_inner, dim=3, obs_matrix=obs_matrix)

        # Simulate 5 direction calls
        scores = []
        for direction in range(5):
            one_hot = tf.one_hot(direction, 5, dtype=DTYPE)
            set_dir_grad(one_hot)
            # Dummy score computation
            score = tf.reduce_sum(theta_inner * one_hot)
            scores.append(score)

        return upstream * tf.stack(scores)

    return value, grad_fn

print("  Calling test_fn(theta)...")
t0 = time.time()
result = test_fn(theta)
t1 = time.time()
print(f"  Time (first call, triggers tracing): {t1-t0:.3f}s")

print("\n  Computing gradient (triggers gradient tracing)...")
t0 = time.time()
with tf.GradientTape() as tape:
    tape.watch(theta)
    val = test_fn(theta)
grad = tape.gradient(val, theta)
t1 = time.time()
print(f"  Time: {t1-t0:.3f}s")
print(f"  Gradient shape: {grad.shape}")

print("\n  Second call (should be cached)...")
t0 = time.time()
result2 = test_fn(theta)
t1 = time.time()
print(f"  Time: {t1-t0:.3f}s")

print("\nDone")
