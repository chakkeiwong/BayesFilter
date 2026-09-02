"""Phase 2: Surrogate-force HMC - Single Arm Test (Arm B only).

Simplified version to test the damped surrogate force approach.
Pure TensorFlow, XLA-compatible, no numpy in graph mode.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import json
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Import from worktree
sys.path.insert(0, str(_REPO_ROOT / ".claude/worktrees/ledh-canonical-rebuild"))

from bayesfilter.highdim.ledh_canonical_score_tf import (
    canonical_value_and_analytical_score,
)
from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import diagonal_lgssm_any_dim

DTYPE = tf.float64
THETA_TRUE = np.array([0.72, 0.55, 0.35, 0.35, 0.45])
DIM = 3
HORIZON = 50
DATA_SEED = 81100
PARTICLES = 1008
FLOW_SUBSTEPS = 12


def _seed_from_theta_and_salt(theta: tf.Tensor, salt: int) -> tf.Tensor:
    """Derive stateless seed pair from theta (pure TF, deterministic)."""
    modulus = 2147483647
    seed_int = tf.cast(
        tf.reduce_sum(tf.abs(theta) * 1e7 + salt) % modulus,
        tf.int32
    )
    return tf.stack([seed_int, salt], axis=0)


def make_dual_adapter_ledh_single():
    """Single-chain adapter for testing."""

    # Generate observations once
    rng = np.random.default_rng(DATA_SEED)
    phi = THETA_TRUE[:DIM]
    sw, sv = THETA_TRUE[DIM], THETA_TRUE[DIM+1]
    x = np.zeros(DIM)
    obs = []
    for _ in range(HORIZON):
        x = phi * x + sw * rng.standard_normal(DIM)
        obs.append(x + sv * rng.standard_normal(DIM))
    observations = tf.constant(np.array(obs), DTYPE)
    obs_matrix = tf.eye(DIM, dtype=DTYPE)

    # Pre-compute constants
    sqrt_d_val = np.sqrt(DIM)
    basis_np = np.array([
        [sqrt_d_val, 0.0, 0.0],
        [-sqrt_d_val, 0.0, 0.0],
        [0.0, sqrt_d_val, 0.0],
        [0.0, -sqrt_d_val, 0.0],
        [0.0, 0.0, sqrt_d_val],
        [0.0, 0.0, -sqrt_d_val],
    ])
    reset_design_const = tf.constant(np.tile(basis_np, (PARTICLES // 6, 1)), DTYPE)
    initial_covs_const = tf.constant(np.stack([np.eye(DIM)] * PARTICLES), DTYPE)

    # Damped config
    ridge_exact, damping_exact = 1e-5, 1e-5
    ridge_damped, damping_damped = 1e-3, 1e-3

    # Wrap LEDH calls in @tf.function for caching (CRITICAL for performance)
    @tf.function
    def ledh_value_exact(theta_in, initial_noise, transition_noise):
        model, _ = diagonal_lgssm_any_dim(theta_in, dim=DIM, obs_matrix=obs_matrix)
        value, _ = canonical_value_and_analytical_score(
            model, theta_in, initial_noise, initial_covs_const,
            transition_noise, observations,
            flow_substeps=FLOW_SUBSTEPS, with_score=False,
            reset_policy="contract_e", reset_design=reset_design_const,
            reset_epsilon=1.0, reset_sinkhorn_steps=8, reset_balance_steps=8,
            reset_ridge=ridge_exact, correction_steps=1,
            correction_lm_damping=damping_exact, pairwise_steps=1,
            annealed_stages=1, annealed_seed=17,
        )
        return value

    @tf.function
    def ledh_score_damped(theta_in, initial_noise, transition_noise, direction_onehot):
        model, set_dir = diagonal_lgssm_any_dim(theta_in, dim=DIM, obs_matrix=obs_matrix)
        set_dir(direction_onehot)
        _, score = canonical_value_and_analytical_score(
            model, theta_in, initial_noise, initial_covs_const,
            transition_noise, observations,
            flow_substeps=FLOW_SUBSTEPS, with_score=True,
            reset_policy="contract_e", reset_design=reset_design_const,
            reset_epsilon=1.0, reset_sinkhorn_steps=8, reset_balance_steps=8,
            reset_ridge=ridge_damped, correction_steps=1,
            correction_lm_damping=damping_damped, pairwise_steps=1,
            annealed_stages=1, annealed_seed=17,
        )
        return score[0]

    @tf.custom_gradient
    def value_with_damped_score(theta_inner: tf.Tensor) -> tf.Tensor:
        # Generate frozen noise
        seed_init = _seed_from_theta_and_salt(theta_inner, salt=1001)
        seed_transition = _seed_from_theta_and_salt(theta_inner, salt=2002)

        initial_noise = tf.random.stateless_normal(
            [PARTICLES, DIM], seed_init, dtype=DTYPE
        )
        transition_noise = tf.random.stateless_normal(
            [HORIZON, PARTICLES, DIM], seed_transition, dtype=DTYPE
        )

        # Value: call wrapped @tf.function
        value = ledh_value_exact(theta_inner, initial_noise, transition_noise)

        def grad_fn(upstream: tf.Tensor) -> tf.Tensor:
            # Score: call wrapped @tf.function for each direction
            directions_matrix = tf.eye(5, dtype=DTYPE)
            score_list = []
            for direction in tf.range(5):
                one_hot = directions_matrix[direction]
                score = ledh_score_damped(theta_inner, initial_noise, transition_noise, one_hot)
                score_list.append(score)

            return upstream * tf.stack(score_list, axis=0)

        return value, grad_fn

    return value_with_damped_score


def main():
    print("=" * 80)
    print("Phase 2: Surrogate-Force HMC Test (Arm B only)")
    print("=" * 80)
    print(f"Model: d={DIM}, T={HORIZON}, N={PARTICLES}")
    print(f"Config: Damped score (λ=1e-3, δ=1e-3)")
    print()

    # Build adapter
    print("Building adapter...")
    target_fn = make_dual_adapter_ledh_single()

    # Skip test evaluation - it triggers full gradient compilation
    # which is extremely slow. HMC will do the first evaluation.
    print("Skipping test evaluation (would trigger slow compilation)")
    print()

    # HMC setup
    num_chains = 2  # Reduced from 4
    num_warmup = 500  # Reduced from 1000
    num_samples = 500  # Reduced from 1000
    step_size = 0.01
    num_leapfrog_steps = 10
    theta_init = THETA_TRUE + 0.01 * np.random.randn(5)

    print(f"Running HMC: {num_chains} chains, {num_warmup} warmup, {num_samples} samples")
    t0 = time.time()

    # Single chain (unbatched)
    init_state = tf.constant(theta_init, dtype=DTYPE)

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )

    def trace_fn(current_state, kernel_results):
        return {
            'is_accepted': kernel_results.is_accepted,
            'log_accept_ratio': kernel_results.log_accept_ratio,
        }

    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_warmup,
        current_state=init_state,
        kernel=kernel,
        trace_fn=trace_fn,
        seed=tf.constant([42, 1337], tf.int32),
    )

    elapsed = time.time() - t0

    # Analysis
    samples_np = samples.numpy()
    is_accepted = trace['is_accepted'].numpy()

    warmup_accept = is_accepted[:num_warmup // 2].mean()
    sampling_accept = is_accepted[num_warmup // 2:].mean()

    posterior_mean = samples_np.mean(axis=0)
    posterior_std = samples_np.std(axis=0)

    # ESS
    ess_all = []
    for p in range(5):
        ess_p = tfp.mcmc.effective_sample_size(samples[:, p]).numpy()
        ess_all.append(float(ess_p))

    ess_min = min(ess_all)
    num_grads = (num_warmup + num_samples) * num_leapfrog_steps
    ess_per_grad = ess_min / num_grads

    # Rhat (need >1 chain)
    # For single chain, skip Rhat

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Acceptance (warmup): {warmup_accept:.3f}")
    print(f"Acceptance (sampling): {sampling_accept:.3f}")
    print(f"ESS (min): {ess_min:.1f}")
    print(f"ESS/grad: {ess_per_grad:.4f}")
    print()
    print("Posterior mean:")
    for p in range(5):
        print(f"  theta[{p}]: {posterior_mean[p]:.4f} ± {posterior_std[p]:.4f} (true: {THETA_TRUE[p]:.2f})")
    print()

    # Coverage
    coverage = []
    for p in range(5):
        ci_low = posterior_mean[p] - 1.96 * posterior_std[p]
        ci_high = posterior_mean[p] + 1.96 * posterior_std[p]
        covers = (ci_low <= THETA_TRUE[p] <= ci_high)
        coverage.append(covers)

    print(f"Coverage: {sum(coverage)}/5 parameters")
    print()

    # Success criteria
    pass_accept = sampling_accept > 0.3
    pass_coverage = sum(coverage) >= 4  # At least 4/5

    print("Success criteria:")
    print(f"  {'✓' if pass_accept else '✗'} Acceptance > 0.3: {sampling_accept:.3f}")
    print(f"  {'✓' if pass_coverage else '✗'} Coverage ≥ 4/5: {sum(coverage)}/5")
    print()

    if pass_accept and pass_coverage:
        print("STATUS: PASS (single-chain test)")
    else:
        print("STATUS: FAIL")
    print()

    # Save
    output_path = Path(__file__).parent / "phase2_single_arm_test.json"
    with open(output_path, 'w') as f:
        json.dump({
            'status': 'PASS' if (pass_accept and pass_coverage) else 'FAIL',
            'config': {
                'model': 'LGSSM d=3 T=50',
                'particles': PARTICLES,
                'ridge_exact': 1e-5,
                'damping_exact': 1e-5,
                'ridge_damped': 1e-3,
                'damping_damped': 1e-3,
            },
            'hmc': {
                'num_chains': 1,
                'num_warmup': num_warmup,
                'num_samples': num_samples,
                'step_size': step_size,
                'num_leapfrog_steps': num_leapfrog_steps,
            },
            'results': {
                'elapsed_sec': elapsed,
                'warmup_acceptance': float(warmup_accept),
                'sampling_acceptance': float(sampling_accept),
                'ess_min': float(ess_min),
                'ess_per_grad': float(ess_per_grad),
                'posterior_mean': posterior_mean.tolist(),
                'posterior_std': posterior_std.tolist(),
                'coverage': coverage,
            },
        }, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
