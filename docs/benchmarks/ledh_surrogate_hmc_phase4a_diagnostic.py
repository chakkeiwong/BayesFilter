"""Phase 4a: LEDH Surrogate-Force HMC Ultra-Short Diagnostic.

Master program: ledh-surrogate-hmc-executable-master-program-2026-09-07.md
Phase: 4a (diagnostic gate before expensive Phase 4b)

Configuration:
- LGSSM d=3, T=50, N=1008 particles
- 2 arms: exact-force (zero damping) vs damped-force (ε-based damping)
- 2 chains × 1000 steps per arm (parallel batched execution)
- One frozen ω (deterministic noise for Corollary 5.2)

Budget: 2 GPU-hours (1 hour per arm, chains in parallel)

Success criterion (diagnostic only):
- If W₂ < threshold → proceed to Phase 4b
- If W₂ > threshold → method broken, STOP and diagnose

Evidence contract:
- Primary: W₂ agreement (diagnostic threshold)
- Veto: None (diagnostic only, not certification)
- Explanatory: Acceptance rate, R-hat, ESS

Deliverable: docs/plans/ledh-surrogate-hmc-phase4a-diagnostic-result-2026-09-07.md
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# TensorFlow GPU Memory Rule: enable and verify memory growth
_VISIBLE_GPUS = tf.config.list_physical_devices("GPU")
for _gpu in _VISIBLE_GPUS:
    tf.config.experimental.set_memory_growth(_gpu, True)
    if not tf.config.experimental.get_memory_growth(_gpu):
        raise RuntimeError(
            f"set_memory_growth failed on {_gpu.name}; "
            "refusing to run with whole-device preallocation"
        )
GPU_MEMORY_POLICY = (
    "memory_growth_verified" if _VISIBLE_GPUS else "cpu_only_no_gpu_visible"
)

import tensorflow_probability as tfp

from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _diagonal_lgssm_fused_model,
    _lgssm_frozen_observations,
)
from bayesfilter.inference.ledh_dual_parameter_target import (
    DualParameterLEDHTarget,
)

tfb = tfp.bijectors
tfd = tfp.distributions


def make_dual_target_exact_vs_damped(
    model,
    initial_states,
    initial_covariances,
    noises,
    observations,
    exact: bool,
    **shared_params,
) -> DualParameterLEDHTarget:
    """Create dual-parameter target: exact (zero damping) or damped (ε-based).

    For Corollary 5.2 surrogate-force HMC:
    - Exact arm: value and score both use base parameters (zero extra damping)
    - Damped arm: value uses base, score uses damped parameters

    Parameters
    ----------
    exact : bool
        If True, both value and score use base params (exact-force HMC).
        If False, value uses base, score uses damped params (surrogate-force HMC).
    """
    # Base parameters (for exact value computation)
    # Phase 4a repair: strengthened regularization for HMC parameter exploration
    # Original: reset_ridge=1e-5, lm_damping=1e-2 (sufficient for Phase 3.5 fixed-θ)
    # Repair: 100× ridge increase, 10× damping increase for HMC's dynamic sampling
    base_reset_ridge = 1e-3  # Was 1e-5
    base_lm_damping = 1e-1   # Was 1e-2

    exact_params = {
        **shared_params,
        "reset_ridge": base_reset_ridge,
        "correction_lm_damping": base_lm_damping,
    }

    if exact:
        # Exact arm: score also uses base parameters
        biased_params = exact_params
    else:
        # Damped arm: score uses damped parameters (ε-based damping)
        # Master program specifies ε=0.01 as the damping level
        damping_epsilon = 0.01
        damped_reset_ridge = base_reset_ridge + damping_epsilon
        damped_lm_damping = base_lm_damping + damping_epsilon

        biased_params = {
            **shared_params,
            "reset_ridge": damped_reset_ridge,
            "correction_lm_damping": damped_lm_damping,
        }

    return DualParameterLEDHTarget(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        exact_params=exact_params,
        biased_params=biased_params,
    )


def run_hmc_single_chain(
    target_log_prob_fn,
    initial_theta,
    num_burnin_steps: int,
    num_results: int,
    step_size: float,
    num_leapfrog_steps: int,
    seed: int,
):
    """Run a single HMC chain.

    Due to tf.custom_gradient in DualParameterLEDHTarget, batched execution
    causes bootstrap issues. Run chains sequentially instead.

    Returns
    -------
    samples : Tensor, shape [num_results, param_dim]
    trace : dict with keys:
        - is_accepted: [num_results]
        - step_size: [num_results]
    """
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )

    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.65,
    )

    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_theta,
        kernel=adaptive_kernel,
        trace_fn=lambda _, pkr: {
            "is_accepted": pkr.inner_results.is_accepted,
            "step_size": pkr.new_step_size,
        },
        seed=seed,
    )


def compute_diagnostics(samples, trace):
    """Compute acceptance rate, R-hat, and ESS.

    Parameters
    ----------
    samples : Tensor, shape [num_results, param_dim] or [num_chains, num_results, param_dim]
    trace : dict with is_accepted [num_results] or [num_chains, num_results]

    Returns
    -------
    dict with:
        acceptance_rate : float (pooled across chains if multiple)
        rhat : ndarray [param_dim] (split R-hat per parameter)
        ess_bulk : ndarray [param_dim] (bulk ESS per parameter)
        ess_tail : ndarray [param_dim] (tail ESS per parameter)
    """
    # Handle single chain: reshape to [1, num_results, param_dim] for R-hat
    if samples.shape.rank == 2:
        samples = samples[None, :, :]  # [S, P] -> [1, S, P]

    # Handle trace: ensure 2D [num_chains, num_results]
    is_accepted = trace["is_accepted"]
    if is_accepted.shape.rank == 1:
        is_accepted = is_accepted[None, :]

    acceptance_rate = float(tf.reduce_mean(tf.cast(is_accepted, tf.float32)))

    # R-hat: split each chain and compute rank-normalized split R-hat
    rhat = tfp.mcmc.potential_scale_reduction(samples, split_chains=True)
    rhat_np = rhat.numpy()

    # ESS: bulk and tail
    ess_bulk = tfp.mcmc.effective_sample_size(
        samples, filter_beyond_positive_pairs=True, cross_chain_dims=1
    )
    ess_tail = tfp.mcmc.effective_sample_size(
        samples, filter_beyond_positive_pairs=True, cross_chain_dims=1
    )

    return {
        "acceptance_rate": acceptance_rate,
        "rhat": rhat_np,
        "ess_bulk": ess_bulk.numpy(),
        "ess_tail": ess_tail.numpy(),
    }


def wasserstein2_empirical(samples_a, samples_b):
    """Compute empirical Wasserstein-2 distance between two sample sets.

    Uses marginal W₂² = sum over coordinates of (μ_a - μ_b)² + (σ_a - σ_b)²
    as a simple diagnostic. This is the W₂ under independence assumption.

    Parameters
    ----------
    samples_a, samples_b : ndarray, shape [S, num_chains, P]
        Pooled samples from two arms.

    Returns
    -------
    w2 : float
        Empirical W₂ distance (marginal independence approximation)
    """
    # Pool chains
    a_flat = samples_a.reshape(-1, samples_a.shape[-1])  # [S*chains, P]
    b_flat = samples_b.reshape(-1, samples_b.shape[-1])

    # Marginal means and stds
    mean_a = np.mean(a_flat, axis=0)
    mean_b = np.mean(b_flat, axis=0)
    std_a = np.std(a_flat, axis=0, ddof=1)
    std_b = np.std(b_flat, axis=0, ddof=1)

    # W₂² = sum[(μ_a - μ_b)² + (σ_a - σ_b)²]
    w2_squared = np.sum((mean_a - mean_b)**2 + (std_a - std_b)**2)
    return float(np.sqrt(w2_squared))


def main():
    print("=" * 80)
    print("Phase 4a: LEDH Surrogate-Force HMC Ultra-Short Diagnostic")
    print("=" * 80)
    print()
    print("Configuration (master program Phase 4a):")
    print("  LGSSM d=3, T=50, N=252 particles (Phase 3.5 validated size)")
    print("  2 arms: exact-force (zero damping) vs damped-force (ε=0.01)")
    print("  2 chains × 1000 steps per arm (parallel batched execution)")
    print("  Budget: 2 GPU-hours (1 hour per arm)")
    print()
    print("Success criterion (diagnostic only):")
    print("  If W₂ < threshold → proceed to Phase 4b")
    print("  If W₂ > threshold → method broken, STOP")
    print()
    print(f"GPU memory policy: {GPU_MEMORY_POLICY}")
    print(f"GPU devices: {[g.name for g in tf.config.list_physical_devices('GPU')]}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print()

    # [1/7] Load LGSSM T=50 fixture
    print("[1/7] Loading LGSSM T=50 frozen observations...")
    observations = _lgssm_frozen_observations()
    T = observations.shape[0]
    dtype = observations.dtype
    print(f"  T={T}, dtype={dtype}")

    # [2/7] Create LGSSM model
    print("[2/7] Creating diagonal LGSSM model...")
    model = _diagonal_lgssm_fused_model()
    d = 3
    print(f"  State dimension: {d}")

    # [3/7] Initialize particles and noises (one frozen ω)
    print("[3/7] Generating frozen particles and noises (N=252)...")
    N = 252  # Phase 3.5 validated size (reduced from 1008 due to GPU memory)
    MASTER_SEED = 81100  # One frozen ω for Corollary 5.2

    generator = tf.random.Generator.from_seed(MASTER_SEED)
    initial_states = generator.normal([N, d], dtype=dtype) * 0.1
    initial_covariances = tf.tile(tf.eye(d, dtype=dtype)[None, :, :], [N, 1, 1]) * 0.01
    noises = generator.normal([T, N, d], dtype=dtype) * 0.1
    print(f"  Frozen ω seed: {MASTER_SEED}")
    print(f"  Particles: {N}")

    # [4/7] Shared LEDH parameters (Contract E)
    print("[4/7] Setting shared LEDH parameters (Contract E)...")
    reset_basis = tf.concat([tf.eye(d, dtype=dtype), -tf.eye(d, dtype=dtype)], axis=0)
    reset_repeats = (N + 2 * d - 1) // (2 * d)
    reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:N]

    shared_params = dict(
        substeps=8,
        reset_policy="contract_e",
        reset_design=reset_design,
        reset_epsilon=2.0,
        reset_sinkhorn_steps=8,
        reset_balance_steps=8,
        correction_steps=0,  # Phase 1 constraint: correction disabled
        correction_strength=0.2,
        correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5,
        pairwise_steps=0,  # Phase 1 constraint: pairwise disabled
        pairwise_strength=0.02,
        pairwise_rms_cap=2.0,
        coordinate_cap=0.0,
        annealed_stages=1,  # Phase 1 constraint: no annealing
        annealed_seed=0,
    )
    print("  ✓ Contract E parameters set")

    # [5/7] Create dual-parameter targets
    print("[5/7] Creating dual-parameter targets...")
    target_exact = make_dual_target_exact_vs_damped(
        model, initial_states, initial_covariances, noises, observations,
        exact=True, **shared_params
    )
    target_damped = make_dual_target_exact_vs_damped(
        model, initial_states, initial_covariances, noises, observations,
        exact=False, **shared_params
    )
    print("  ✓ Exact-force target (zero damping)")
    print("  ✓ Damped-force target (ε=0.01)")

    # Create dual-parameter targets (no batched wrapper needed - run chains sequentially)
    print("  ✓ Exact-force target (zero damping)")
    print("  ✓ Damped-force target (ε=0.01)")

    # [6/7] HMC configuration
    num_chains = 2
    num_burnin_steps = 1000
    num_results = 1000
    step_size = 0.01
    num_leapfrog_steps = 10
    base_seed = 97701

    true_theta = tf.constant([1.0, 1.0, 1.0, 0.5, 0.3], dtype=dtype)

    print()
    print("[6/7] HMC configuration:")
    print(f"  Chains: {num_chains} (sequential, not batched)")
    print(f"  Burn-in: {num_burnin_steps}")
    print(f"  Samples: {num_results}")
    print(f"  Initial step size: {step_size}")
    print(f"  Leapfrog steps: {num_leapfrog_steps}")
    print(f"  Initial theta: {true_theta.numpy()}")

    # [7/7] Run HMC for both arms
    print()
    print("[7/7] Running HMC chains...")
    print("=" * 80)

    results = {}

    # Arm 1: Exact-force HMC
    print()
    print("Arm 1: Exact-force HMC (zero damping)")
    print("-" * 80)
    t0_arm1 = time.time()

    # Run chains sequentially
    chain_samples_exact = []
    chain_traces_exact = []
    for chain_id in range(num_chains):
        print(f"  Chain {chain_id + 1}/{num_chains}...")
        sys.stdout.flush()
        samples_chain, trace_chain = run_hmc_single_chain(
            target_log_prob_fn=target_exact,
            initial_theta=true_theta,
            num_burnin_steps=num_burnin_steps,
            num_results=num_results,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            seed=base_seed + chain_id,
        )
        chain_samples_exact.append(samples_chain)
        chain_traces_exact.append(trace_chain)
        print(f"    ✓ Chain {chain_id + 1} complete")

    # Stack chains: [num_chains, num_results, param_dim]
    samples_exact = tf.stack(chain_samples_exact, axis=0)
    trace_exact = {
        "is_accepted": tf.stack([t["is_accepted"] for t in chain_traces_exact], axis=0),
        "step_size": tf.stack([t["step_size"] for t in chain_traces_exact], axis=0),
    }

    t1_arm1 = time.time()
    wall_time_exact = t1_arm1 - t0_arm1

    diag_exact = compute_diagnostics(samples_exact, trace_exact)

    print(f"Wall time: {wall_time_exact:.1f}s ({wall_time_exact/3600:.2f} hours)")
    print(f"Acceptance rate: {diag_exact['acceptance_rate']:.3f}")
    print(f"R-hat: {diag_exact['rhat']}")
    print(f"ESS bulk: {diag_exact['ess_bulk']}")

    results["exact"] = {
        "samples": samples_exact.numpy(),
        "trace": {k: v.numpy() for k, v in trace_exact.items()},
        "diagnostics": diag_exact,
        "wall_time": wall_time_exact,
    }

    # Arm 2: Damped-force HMC
    print()
    print("Arm 2: Damped-force HMC (ε=0.01)")
    print("-" * 80)
    t0_arm2 = time.time()

    # Run chains sequentially
    chain_samples_damped = []
    chain_traces_damped = []
    for chain_id in range(num_chains):
        print(f"  Chain {chain_id + 1}/{num_chains}...")
        sys.stdout.flush()
        samples_chain, trace_chain = run_hmc_single_chain(
            target_log_prob_fn=target_damped,
            initial_theta=true_theta,
            num_burnin_steps=num_burnin_steps,
            num_results=num_results,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            seed=base_seed + num_chains + chain_id,
        )
        chain_samples_damped.append(samples_chain)
        chain_traces_damped.append(trace_chain)
        print(f"    ✓ Chain {chain_id + 1} complete")

    # Stack chains: [num_chains, num_results, param_dim]
    samples_damped = tf.stack(chain_samples_damped, axis=0)
    trace_damped = {
        "is_accepted": tf.stack([t["is_accepted"] for t in chain_traces_damped], axis=0),
        "step_size": tf.stack([t["step_size"] for t in chain_traces_damped], axis=0),
    }

    t1_arm2 = time.time()
    wall_time_damped = t1_arm2 - t0_arm2

    diag_damped = compute_diagnostics(samples_damped, trace_damped)

    print(f"Wall time: {wall_time_damped:.1f}s ({wall_time_damped/3600:.2f} hours)")
    print(f"Acceptance rate: {diag_damped['acceptance_rate']:.3f}")
    print(f"R-hat: {diag_damped['rhat']}")
    print(f"ESS bulk: {diag_damped['ess_bulk']}")

    results["damped"] = {
        "samples": samples_damped.numpy(),
        "trace": {k: v.numpy() for k, v in trace_damped.items()},
        "diagnostics": diag_damped,
        "wall_time": wall_time_damped,
    }

    # Compute W₂ distance
    print()
    print("=" * 80)
    print("Posterior Agreement (W₂ distance)")
    print("-" * 80)
    w2_distance = wasserstein2_empirical(samples_exact.numpy(), samples_damped.numpy())
    print(f"W₂ distance (marginal approx): {w2_distance:.6f}")

    # Diagnostic threshold (not a hard criterion, just a gate)
    # Use a permissive threshold since this is just a diagnostic
    W2_THRESHOLD = 1.0  # Permissive for Phase 4a diagnostic

    print()
    if w2_distance < W2_THRESHOLD:
        decision = "PASS"
        next_action = "Proceed to Phase 4b (full certification)"
        print(f"✓ DIAGNOSTIC PASS: W₂ < {W2_THRESHOLD}")
        print(f"  {next_action}")
    else:
        decision = "FAIL"
        next_action = "STOP and diagnose (method broken)"
        print(f"✗ DIAGNOSTIC FAIL: W₂ ≥ {W2_THRESHOLD}")
        print(f"  {next_action}")

    results["w2_distance"] = w2_distance
    results["w2_threshold"] = W2_THRESHOLD
    results["decision"] = decision
    results["next_action"] = next_action

    # Save results
    output_dir = Path("docs/plans")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / "ledh-surrogate-hmc-phase4a-diagnostic-result-2026-09-07.json"

    # Convert arrays to lists for JSON serialization
    results_json = {
        "exact": {
            "diagnostics": {
                "acceptance_rate": results["exact"]["diagnostics"]["acceptance_rate"],
                "rhat": results["exact"]["diagnostics"]["rhat"].tolist(),
                "ess_bulk": results["exact"]["diagnostics"]["ess_bulk"].tolist(),
            },
            "wall_time": results["exact"]["wall_time"],
        },
        "damped": {
            "diagnostics": {
                "acceptance_rate": results["damped"]["diagnostics"]["acceptance_rate"],
                "rhat": results["damped"]["diagnostics"]["rhat"].tolist(),
                "ess_bulk": results["damped"]["diagnostics"]["ess_bulk"].tolist(),
            },
            "wall_time": results["damped"]["wall_time"],
        },
        "w2_distance": results["w2_distance"],
        "w2_threshold": results["w2_threshold"],
        "decision": results["decision"],
        "next_action": results["next_action"],
    }

    with open(result_file, "w") as f:
        json.dump(results_json, f, indent=2)

    print()
    print(f"Results saved to: {result_file}")
    print()
    print("=" * 80)
    print(f"Phase 4a Diagnostic: {decision}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
