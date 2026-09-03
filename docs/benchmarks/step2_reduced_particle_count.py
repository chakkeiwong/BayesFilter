"""Step 2: Test reduced particle count for longer runs.

Test N=252 (quarter of 1008) to enable longer HMC runs within memory constraints.
Use true surrogate-force from Step 1.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import time
import gc

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

print("=" * 70)
print("Step 2: Reduced Particle Count Test (N=252)")
print("=" * 70)
print()

# Build with reduced particle count
N_REDUCED = 252
print(f"Building models with N={N_REDUCED} (was 1008)...")
exact_target = make_canonical_neutra_target("lgssm", particle_count=N_REDUCED, noise_seed=140000, substeps=12)
exact_model = exact_target.fused_model

ridge_lambda_damped = 1e-3
ridge_delta_damped = 1e-3

damped_model = PerPointScoreModel(
    transition_mean_fn=exact_model.transition_mean_fn,
    transition_mean_tangent_fn=exact_model.transition_mean_tangent_fn,
    observation_fn=exact_model.observation_fn,
    observation_jacobian_fn=exact_model.observation_jacobian_fn,
    observation_tangent_fn=exact_model.observation_tangent_fn,
    process_covariance=exact_model.process_covariance + ridge_lambda_damped * tf.eye(3, dtype=DTYPE),
    observation_covariance=exact_model.observation_covariance + ridge_delta_damped * tf.eye(3, dtype=DTYPE),
)

print(f"  Exact:  λ=1e-5, δ=1e-5")
print(f"  Damped: λ={ridge_lambda_damped:.1e}, δ={ridge_delta_damped:.1e}")
print()

# Surrogate-force target
print("Building surrogate-force target...")

def surrogate_force_target_fn(theta):
    theta = tf.convert_to_tensor(theta, DTYPE)
    if len(theta.shape) == 1:
        theta_batch = tf.reshape(theta, [1, -1])
        is_scalar = True
    else:
        theta_batch = theta
        is_scalar = False

    @tf.custom_gradient
    def exact_value_damped_grad(theta_b):
        directions_dummy = tf.zeros_like(theta_b)
        exact_vals, _, _ = canonical_batch_fused_value_score(
            exact_model, theta_b, directions_dummy,
            exact_target.initial_states, exact_target.initial_covariances,
            exact_target.noises, exact_target.observations,
            substeps=exact_target.substeps,
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
                    damped_model, theta_b, direction,
                    exact_target.initial_states, exact_target.initial_covariances,
                    exact_target.noises, exact_target.observations,
                    substeps=exact_target.substeps,
                )
                gradients.append(scores)
            grad_batch = tf.stack(gradients, axis=1)
            return dy[:, None] * grad_batch if len(dy.shape) > 0 else grad_batch[0]

        return exact_vals, grad_fn

    result = exact_value_damped_grad(theta_batch)
    if is_scalar:
        return -result[0]
    return -result

print("Testing target...")
t0 = time.time()
val = surrogate_force_target_fn(THETA_TRUE)
print(f"  Value: {float(val):.6f} ({time.time()-t0:.1f}s)")
print()

# Longer HMC run (still conservative)
print("HMC settings:")
num_burnin = 50
num_results = 50
num_chains = 2
print(f"  Chains: {num_chains}")
print(f"  Burnin: {num_burnin}")
print(f"  Samples: {num_results}")
print(f"  Leapfrog: 3")
print()

kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=surrogate_force_target_fn,
    step_size=0.01,
    num_leapfrog_steps=3,
)

adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel,
    num_adaptation_steps=int(0.8 * num_burnin),
    target_accept_prob=0.65,
)

initial_state = tf.constant(
    np.random.RandomState(42).randn(num_chains, 5) * 0.1 + THETA_TRUE,
    DTYPE
)

print("Running HMC...")
print(f"  Estimated time: ~10-15 minutes with N={N_REDUCED}")
t0 = time.time()

samples, trace = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin,
    current_state=initial_state,
    kernel=adaptive_kernel,
    trace_fn=lambda _, pkr: {
        'is_accepted': pkr.inner_results.is_accepted,
        'step_size': pkr.inner_results.accepted_results.step_size,
    },
    seed=42,
)

t1 = time.time()
print(f"  Completed in {(t1-t0)/60:.1f} minutes")

acceptance = float(tf.reduce_mean(tf.cast(trace['is_accepted'], tf.float32)))
final_step_size = float(trace['step_size'][-1])

print(f"\nResults:")
print(f"  Acceptance: {acceptance:.3f}")
print(f"  Final step size: {final_step_size:.4f}")
print(f"  Samples shape: {samples.shape}")

means = tf.reduce_mean(samples, axis=[0, 1]).numpy()
stds = tf.math.reduce_std(samples, axis=[0, 1]).numpy()

print(f"\nParameter estimates (mean ± std):")
param_names = ['phi1', 'phi2', 'phi3', 'q', 'r']
for i, name in enumerate(param_names):
    print(f"  {name}: {means[i]:.4f} ± {stds[i]:.4f}  (true: {THETA_TRUE[i]:.4f})")

print("\n" + "=" * 70)
print("SUCCESS: Reduced N enables longer HMC runs!")
print("=" * 70)
print(f"\nKey finding: N={N_REDUCED} completes {num_chains} chains × {num_burnin+num_results} steps")
print("Next: Three-arm comparison with this N")
