"""Step 3: Three-arm comparison with reduced particle count.

Compare on equal footing with N=252:
- Arm A: Exact force (λ=1e-5, δ=1e-5 for both value and gradient)
- Arm B: Damped force (λ=1e-3, δ=1e-3 for both value and gradient)
- Arm C: Surrogate-force (exact value, damped gradient)

Success: Arm C posterior matches Arm A, acceptance comparable to Arm B.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import time
import gc
import json

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / ".claude/worktrees/ledh-canonical-rebuild"))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
)
from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    make_canonical_neutra_target,
)

gc.collect()

DTYPE = tf.float64
THETA_TRUE = np.array([0.72, 0.55, 0.35, 0.35, 0.45])
N_PARTICLES = 252

print("=" * 70)
print("Step 3: Three-Arm Comparison")
print("=" * 70)
print(f"\nParticles: N={N_PARTICLES}")
print()

# Build base target and models
print("Building models...")
base_target = make_canonical_neutra_target("lgssm", particle_count=N_PARTICLES, noise_seed=140000, substeps=12)
base_model = base_target.fused_model

# Exact model (λ=1e-5, δ=1e-5)
exact_model = base_model

# Damped model (λ=1e-3, δ=1e-3)
damped_model = PerPointScoreModel(
    transition_mean_fn=base_model.transition_mean_fn,
    transition_mean_tangent_fn=base_model.transition_mean_tangent_fn,
    observation_fn=base_model.observation_fn,
    observation_jacobian_fn=base_model.observation_jacobian_fn,
    observation_tangent_fn=base_model.observation_tangent_fn,
    process_covariance=base_model.process_covariance + 1e-3 * tf.eye(3, dtype=DTYPE),
    observation_covariance=base_model.observation_covariance + 1e-3 * tf.eye(3, dtype=DTYPE),
)

print("  Arm A (exact):  λ=1e-5, δ=1e-5 for value+gradient")
print("  Arm B (damped): λ=1e-3, δ=1e-3 for value+gradient")
print("  Arm C (surrogate): exact value, damped gradient")
print()

# Target functions
def make_target_fn(value_model, gradient_model, name):
    """Create target with specified value and gradient models."""
    def target_fn(theta):
        theta = tf.convert_to_tensor(theta, DTYPE)
        if len(theta.shape) == 1:
            theta_batch = tf.reshape(theta, [1, -1])
            is_scalar = True
        else:
            theta_batch = theta
            is_scalar = False

        @tf.custom_gradient
        def value_with_grad(theta_b):
            directions_dummy = tf.zeros_like(theta_b)
            vals, _, _ = canonical_batch_fused_value_score(
                value_model, theta_b, directions_dummy,
                base_target.initial_states, base_target.initial_covariances,
                base_target.noises, base_target.observations,
                substeps=base_target.substeps,
            )

            def grad_fn(dy):
                batch_size = tf.shape(theta_b)[0]
                param_dim = int(theta_b.shape[1])
                gradients = []
                for p in range(param_dim):
                    direction = tf.zeros([batch_size, param_dim], DTYPE)
                    indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], p)], axis=1)
                    direction = tf.tensor_scatter_nd_update(direction, indices, tf.ones([batch_size], DTYPE))
                    _, scores, _ = canonical_batch_fused_value_score(
                        gradient_model, theta_b, direction,
                        base_target.initial_states, base_target.initial_covariances,
                        base_target.noises, base_target.observations,
                        substeps=base_target.substeps,
                    )
                    gradients.append(scores)
                grad_batch = tf.stack(gradients, axis=1)
                return dy[:, None] * grad_batch if len(dy.shape) > 0 else grad_batch[0]

            return vals, grad_fn

        result = value_with_grad(theta_batch)
        if is_scalar:
            return -result[0]
        return -result

    return target_fn

# Build three targets
arm_a_fn = make_target_fn(exact_model, exact_model, "Arm A")
arm_b_fn = make_target_fn(damped_model, damped_model, "Arm B")
arm_c_fn = make_target_fn(exact_model, damped_model, "Arm C")

# HMC settings
print("HMC settings:")
num_burnin = 50
num_results = 50
num_chains = 2
print(f"  Chains: {num_chains}")
print(f"  Burnin: {num_burnin}")
print(f"  Samples: {num_results}")
print(f"  Leapfrog: 3")
print()

initial_state = tf.constant(
    np.random.RandomState(42).randn(num_chains, 5) * 0.1 + THETA_TRUE,
    DTYPE
)

# Run all three arms
results = {}

for arm_name, target_fn in [("Arm A", arm_a_fn), ("Arm B", arm_b_fn), ("Arm C", arm_c_fn)]:
    print(f"\nRunning {arm_name}...")

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_fn,
        step_size=0.01,
        num_leapfrog_steps=3,
    )

    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * num_burnin),
        target_accept_prob=0.65,
    )

    t0 = time.time()
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin,
        current_state=initial_state,
        kernel=adaptive_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        seed=42 + hash(arm_name) % 1000,
    )
    elapsed = time.time() - t0

    acceptance = float(tf.reduce_mean(tf.cast(trace, tf.float32)))
    means = tf.reduce_mean(samples, axis=[0, 1]).numpy()
    stds = tf.math.reduce_std(samples, axis=[0, 1]).numpy()

    results[arm_name] = {
        'time_min': elapsed / 60,
        'acceptance': acceptance,
        'means': means.tolist(),
        'stds': stds.tolist(),
        'samples': samples.numpy().tolist(),
    }

    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Acceptance: {acceptance:.3f}")
    print(f"  Means: {means}")

# Analysis
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\nAcceptance rates:")
for arm in ["Arm A", "Arm B", "Arm C"]:
    print(f"  {arm}: {results[arm]['acceptance']:.3f}")

print("\nParameter estimates:")
param_names = ['phi1', 'phi2', 'phi3', 'q', 'r']
for i, name in enumerate(param_names):
    print(f"\n  {name} (true={THETA_TRUE[i]:.4f}):")
    for arm in ["Arm A", "Arm B", "Arm C"]:
        m = results[arm]['means'][i]
        s = results[arm]['stds'][i]
        print(f"    {arm}: {m:.4f} ± {s:.4f}")

# Posterior agreement (Arm C vs Arm A)
print("\nPosterior agreement (Arm C vs Arm A):")
mean_diff = np.array(results["Arm C"]['means']) - np.array(results["Arm A"]['means'])
print(f"  Mean difference: {mean_diff}")
print(f"  RMS difference: {np.sqrt(np.mean(mean_diff**2)):.4f}")

# Save results
output_file = "/tmp/three_arm_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_file}")

print("\n" + "=" * 70)
print("SUCCESS: Three-arm comparison complete!")
print("=" * 70)
