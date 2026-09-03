"""Phase 2: Surrogate-force HMC on LGSSM d=3 T=50 (XLA-compatible).

Three-arm comparison:
- Arm A (baseline): Exact score as force (λ=1e-5, δ=1e-5)
- Arm B (surrogate): Damped score as force (λ=1e-3, δ=1e-3)
- Arm C (fallback): Intermediate (λ=1e-4, δ=1e-4)

Pure TensorFlow implementation compatible with XLA/JIT compilation.
No numpy conversions, no Python loops inside TF functions.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import json
import time
from typing import Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Import from worktree (canonical LEDH)
sys.path.insert(0, str(_REPO_ROOT / ".claude/worktrees/ledh-canonical-rebuild"))

from bayesfilter.highdim.ledh_canonical_score_tf import (
    canonical_value_and_analytical_score,
    NonlinearScoreModel,
)
from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import diagonal_lgssm_any_dim

DTYPE = tf.float64

# Historical fixture: benchmark_lgssm_exact_oracle_m3_T50
THETA_TRUE = np.array([0.72, 0.55, 0.35, 0.35, 0.45])
DIM = 3
HORIZON = 50
DATA_SEED = 81100
PARTICLES = 1008
FLOW_SUBSTEPS = 12


def _seed_from_theta_and_salt(theta: tf.Tensor, salt: int) -> tf.Tensor:
    """Derive stateless seed pair from theta (pure TF, deterministic)."""
    # Hash theta values into integer seed
    modulus = 2147483647
    seed_int = tf.cast(
        tf.reduce_sum(tf.abs(theta) * 1e7 + salt) % modulus,
        tf.int32
    )
    return tf.stack([seed_int, salt], axis=0)


def make_dual_adapter_ledh(
    observations: tf.Tensor,
    obs_matrix: tf.Tensor,
    ridge_exact: float,
    damping_exact: float,
    ridge_damped: float,
    damping_damped: float,
):
    """Create dual adapter with frozen noise generation.

    Pure TF implementation: generates noise from theta using stateless random,
    passes same noise to both value (exact config) and score (damped config).
    """

    # Pre-compute constants OUTSIDE the loop (cannot be in map_fn body)
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

    @tf.custom_gradient
    def value_with_damped_score(theta_inner: tf.Tensor) -> tf.Tensor:
        """Single unbatched theta evaluation."""
        # Generate frozen noise from theta (pure TF, stateless)
        seed_init = _seed_from_theta_and_salt(theta_inner, salt=1001)
        seed_transition = _seed_from_theta_and_salt(theta_inner, salt=2002)

        initial_noise = tf.random.stateless_normal(
            [PARTICLES, DIM],
            seed_init,
            dtype=DTYPE
        )
        transition_noise = tf.random.stateless_normal(
            [HORIZON, PARTICLES, DIM],
            seed_transition,
            dtype=DTYPE
        )

        # Build model (exact config)
        model_exact, set_direction = diagonal_lgssm_any_dim(
            theta_inner, dim=DIM, obs_matrix=obs_matrix
        )

        # Value: exact config
        value, _ = canonical_value_and_analytical_score(
            model_exact,
            theta_inner,
            initial_noise,
            initial_covs_const,
            transition_noise,
            observations,
            flow_substeps=FLOW_SUBSTEPS,
            with_score=False,
            reset_policy="contract_e",
            reset_design=reset_design_const,
            reset_epsilon=1.0,
            reset_sinkhorn_steps=8,
            reset_balance_steps=8,
            reset_ridge=ridge_exact,
            correction_steps=1,
            correction_lm_damping=damping_exact,
            pairwise_steps=1,
            annealed_stages=1,
            annealed_seed=17,
        )

        def grad_fn(upstream: tf.Tensor) -> tf.Tensor:
            """Score gradient using damped config (same frozen noise)."""
            # Build model (damped config) - need fresh direction closure
            model_damped, set_direction_damped = diagonal_lgssm_any_dim(
                theta_inner, dim=DIM, obs_matrix=obs_matrix
            )

            # Compute score for all 5 directions
            scores = []
            for direction in range(5):
                one_hot = tf.one_hot(direction, 5, dtype=DTYPE)
                set_direction_damped(one_hot)

                _, score = canonical_value_and_analytical_score(
                    model_damped,
                    theta_inner,
                    initial_noise,  # SAME noise
                    initial_covs_const,
                    transition_noise,  # SAME noise
                    observations,
                    flow_substeps=FLOW_SUBSTEPS,
                    with_score=True,
                    reset_policy="contract_e",
                    reset_design=reset_design_const,
                    reset_epsilon=1.0,
                    reset_sinkhorn_steps=8,
                    reset_balance_steps=8,
                    reset_ridge=ridge_damped,
                    correction_steps=1,
                    correction_lm_damping=damping_damped,
                    pairwise_steps=1,
                    annealed_stages=1,
                    annealed_seed=17,
                )
                scores.append(score[0])

            full_score = tf.stack(scores, axis=0)
            return upstream * full_score

        return value, grad_fn

    # Single-theta evaluation (unbatched)
    def log_prob_unbatched(theta_single: tf.Tensor) -> tf.Tensor:
        """Evaluate single unbatched theta [5]."""
        return value_with_damped_score(theta_single)

    # Vectorized wrapper for batched HMC input
    def log_prob_fn(theta: tf.Tensor) -> tf.Tensor:
        """Handle both batched [num_chains, 5] and unbatched [5] input."""
        if len(theta.shape) == 1:
            # Unbatched: direct call
            return log_prob_unbatched(theta)
        else:
            # Batched: use map_fn (not pfor/vectorized_map which is restricted)
            return tf.map_fn(
                log_prob_unbatched,
                theta,
                dtype=DTYPE,
                parallel_iterations=1,  # Sequential to avoid pfor
            )

    return log_prob_fn


def generate_observations(seed: int) -> np.ndarray:
    """Generate observation path using historical fixture."""
    rng = np.random.default_rng(seed)
    phi = THETA_TRUE[:DIM]
    sw, sv = THETA_TRUE[DIM], THETA_TRUE[DIM+1]
    x = np.zeros(DIM)
    obs = []
    for _ in range(HORIZON):
        x = phi * x + sw * rng.standard_normal(DIM)
        obs.append(x + sv * rng.standard_normal(DIM))
    return np.array(obs)


def run_hmc_arm(
    target_log_prob_fn,
    theta_init: np.ndarray,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
    step_size: float,
    num_leapfrog_steps: int,
) -> dict:
    """Run HMC and collect metrics."""

    t0 = time.time()

    # Convert initial state to TF
    init_state = tf.constant(
        np.stack([theta_init] * num_chains),
        dtype=DTYPE
    )

    # HMC kernel
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )

    # Trace function
    def trace_fn(current_state, kernel_results):
        return {
            'is_accepted': kernel_results.is_accepted,
            'log_accept_ratio': kernel_results.log_accept_ratio,
        }

    # Sample
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_warmup,
        current_state=init_state,
        kernel=kernel,
        trace_fn=trace_fn,
        seed=tf.constant([42, 1337], tf.int32),
    )

    elapsed = time.time() - t0

    # Convert to numpy for analysis
    samples_np = samples.numpy()  # [num_samples, num_chains, 5]
    is_accepted = trace['is_accepted'].numpy()  # [num_samples, num_chains]

    # Compute metrics
    warmup_accept = is_accepted[:num_warmup // 2, :].mean()
    sampling_accept = is_accepted[num_warmup // 2:, :].mean()

    # ESS (per parameter)
    ess_all = []
    for p in range(5):
        ess_p = tfp.mcmc.effective_sample_size(
            samples[:, :, p]
        ).numpy()
        ess_all.append(ess_p.mean())

    ess_min = min(ess_all)
    ess_mean = np.mean(ess_all)

    # ESS per gradient
    num_grads = (num_warmup + num_samples) * num_leapfrog_steps
    ess_per_grad = ess_min / num_grads

    # Posterior statistics
    posterior_mean = samples_np.reshape(-1, 5).mean(axis=0)
    posterior_std = samples_np.reshape(-1, 5).std(axis=0)

    # Coverage check
    coverage = []
    for p in range(5):
        ci_low = posterior_mean[p] - 1.96 * posterior_std[p]
        ci_high = posterior_mean[p] + 1.96 * posterior_std[p]
        covers = (ci_low <= THETA_TRUE[p] <= ci_high)
        coverage.append(covers)

    # Rhat
    rhat = []
    for p in range(5):
        rhat_p = tfp.mcmc.potential_scale_reduction(
            samples[:, :, p]
        ).numpy()
        rhat.append(float(rhat_p))

    return {
        'elapsed_sec': elapsed,
        'warmup_acceptance': float(warmup_accept),
        'sampling_acceptance': float(sampling_accept),
        'ess_min': float(ess_min),
        'ess_mean': float(ess_mean),
        'ess_per_grad': float(ess_per_grad),
        'posterior_mean': posterior_mean.tolist(),
        'posterior_std': posterior_std.tolist(),
        'coverage': coverage,
        'rhat': rhat,
        'samples': samples_np.tolist(),  # Store for inspection
    }


def main():
    print("=" * 80)
    print("Phase 2: Surrogate-Force HMC on LGSSM d=3 T=50")
    print("=" * 80)
    print()

    # Generate observations
    print("Generating observations...")
    obs_np = generate_observations(DATA_SEED)
    observations = tf.constant(obs_np, DTYPE)
    obs_matrix = tf.eye(DIM, dtype=DTYPE)

    print(f"Model: d={DIM}, T={HORIZON}, N={PARTICLES}")
    print(f"True theta: {THETA_TRUE.tolist()}")
    print()

    # HMC configuration
    num_chains = 4
    num_warmup = 1000
    num_samples = 1000
    step_size = 0.01
    num_leapfrog_steps = 10
    theta_init = THETA_TRUE + 0.01 * np.random.randn(5)

    # Three arms
    arms = {
        'A_exact': {
            'ridge_exact': 1e-5,
            'damping_exact': 1e-5,
            'ridge_damped': 1e-5,  # Same as value
            'damping_damped': 1e-5,  # Same as value
            'label': 'Exact (baseline)',
        },
        'B_damped': {
            'ridge_exact': 1e-5,
            'damping_exact': 1e-5,
            'ridge_damped': 1e-3,  # Stronger ridge
            'damping_damped': 1e-3,  # Stronger damping
            'label': 'Damped (surrogate)',
        },
        'C_intermediate': {
            'ridge_exact': 1e-5,
            'damping_exact': 1e-5,
            'ridge_damped': 1e-4,  # Intermediate
            'damping_damped': 1e-4,  # Intermediate
            'label': 'Intermediate (fallback)',
        },
    }

    results = {}

    for arm_name, config in arms.items():
        print(f"Running Arm {arm_name}: {config['label']}")
        print(f"  Ridge: value={config['ridge_exact']:.0e}, score={config['ridge_damped']:.0e}")
        print(f"  Damping: value={config['damping_exact']:.0e}, score={config['damping_damped']:.0e}")

        # Build adapter
        target_fn = make_dual_adapter_ledh(
            observations,
            obs_matrix,
            config['ridge_exact'],
            config['damping_exact'],
            config['ridge_damped'],
            config['damping_damped'],
        )

        # Run HMC
        result = run_hmc_arm(
            target_fn,
            theta_init,
            num_chains,
            num_warmup,
            num_samples,
            step_size,
            num_leapfrog_steps,
        )

        results[arm_name] = {
            'config': config,
            'metrics': result,
        }

        print(f"  Elapsed: {result['elapsed_sec']:.1f}s")
        print(f"  Acceptance (warmup): {result['warmup_acceptance']:.3f}")
        print(f"  Acceptance (sampling): {result['sampling_acceptance']:.3f}")
        print(f"  ESS/grad (min): {result['ess_per_grad']:.4f}")
        print(f"  Rhat (max): {max(result['rhat']):.4f}")
        print(f"  Coverage: {sum(result['coverage'])}/5 parameters")
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()

    header = f"{'Arm':<15} {'Accept':>8} {'ESS/grad':>10} {'Rhat_max':>9} {'Coverage':>9} {'Status':>10}"
    print(header)
    print("-" * len(header))

    for arm_name in ['A_exact', 'B_damped', 'C_intermediate']:
        r = results[arm_name]['metrics']
        accept = r['sampling_acceptance']
        ess_grad = r['ess_per_grad']
        rhat_max = max(r['rhat'])
        coverage = sum(r['coverage'])

        # Pass criteria
        pass_accept = accept > 0.3
        pass_rhat = rhat_max < 1.05
        pass_coverage = coverage == 5
        pass_all = pass_accept and pass_rhat and pass_coverage
        status = "PASS" if pass_all else "FAIL"

        print(f"{results[arm_name]['config']['label']:<15} {accept:>8.3f} {ess_grad:>10.4f} {rhat_max:>9.4f} {coverage:>9}/5 {status:>10}")

    print()

    # Mean shift analysis
    if 'A_exact' in results and 'B_damped' in results:
        mean_a = np.array(results['A_exact']['metrics']['posterior_mean'])
        mean_b = np.array(results['B_damped']['metrics']['posterior_mean'])
        shift = np.linalg.norm(mean_b - mean_a)

        print("Mean shift (B vs A):")
        print(f"  ||μ_B - μ_A|| = {shift:.6f}")
        print(f"  Veto threshold: 0.18")
        print(f"  Status: {'PASS' if shift <= 0.18 else 'FAIL'}")
        print()

    # Save results
    output_path = Path(__file__).parent / "phase2_lgssm_three_arm_v3.json"
    with open(output_path, 'w') as f:
        json.dump({
            'study': 'Phase 2: Surrogate-Force HMC on LGSSM',
            'date': '2026-08-30',
            'fixture': {
                'model': 'diagonal AR(1) LGSSM',
                'dim': DIM,
                'horizon': HORIZON,
                'particles': PARTICLES,
                'theta_true': THETA_TRUE.tolist(),
                'data_seed': DATA_SEED,
            },
            'hmc_config': {
                'num_chains': num_chains,
                'num_warmup': num_warmup,
                'num_samples': num_samples,
                'step_size': step_size,
                'num_leapfrog_steps': num_leapfrog_steps,
            },
            'results': results,
        }, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()

    b_metrics = results['B_damped']['metrics']
    b_accept = b_metrics['sampling_acceptance']
    b_ess_grad = b_metrics['ess_per_grad']
    b_rhat_max = max(b_metrics['rhat'])
    b_coverage = sum(b_metrics['coverage'])

    mean_a = np.array(results['A_exact']['metrics']['posterior_mean'])
    mean_b = np.array(results['B_damped']['metrics']['posterior_mean'])
    mean_shift = np.linalg.norm(mean_b - mean_a)

    pass_criteria = [
        ('Acceptance > 0.3', b_accept > 0.3),
        ('Rhat < 1.05', b_rhat_max < 1.05),
        ('Coverage = 5/5', b_coverage == 5),
        ('Mean shift ≤ 0.18', mean_shift <= 0.18),
    ]

    all_pass = all(p[1] for p in pass_criteria)

    for criterion, passed in pass_criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {criterion}")

    print()
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print()

    if all_pass:
        print("Bounded claim:")
        print("  'Surrogate-force HMC with damped score produces deterministic")
        print("   mechanics and acceptable mixing on the LGSSM diagnostic fixture.")
        print("   The chain samples the executed pseudo-posterior.'")
    else:
        print("Surrogate-force approach did not meet success criteria.")
        print("See intermediate arm (C) or diagnose failures.")
    print()


if __name__ == '__main__':
    main()
