"""Phase 3 Task 3.2: LEDH surrogate-force HMC damping calibration.

Following ledh-surrogate-hmc-phase3-task3.2-calibration-plan-2026-09-11.md:
- 4 arms: 1×, 10×, 100×, 1000× damping ratios
- N = 1008 particles (per plan)
- 2 chains × 500 warmup + 500 samples per arm (screening run)
- Target: acceptance ≥ 0.15, ESS/grad ≥ 0.3× baseline
"""

import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
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


def make_dual_parameter_target(
    model,
    initial_states,
    initial_covariances,
    noises,
    observations,
    damping_ratio: float,
    base_reset_ridge: float,
    base_lm_damping: float,
    **shared_params,
) -> DualParameterLEDHTarget:
    """Create dual-parameter target with specified damping ratio."""
    damped_reset_ridge = base_reset_ridge * damping_ratio
    damped_lm_damping = base_lm_damping * damping_ratio

    # Merge shared params with exact/biased specific params
    exact_params = {
        **shared_params,
        "reset_ridge": base_reset_ridge,
        "correction_lm_damping": base_lm_damping,
    }

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


def run_hmc_chain_eager(
    target_log_prob_fn,
    initial_theta,
    num_burnin_steps: int,
    num_results: int,
    step_size: float,
    num_leapfrog_steps: int,
    seed: int,
):
    """Run one HMC chain in eager mode (no tf.function to avoid graph explosion)."""
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

    # Run in EAGER mode to avoid massive graph
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
    """Compute acceptance rate and ESS."""
    acceptance_rate = float(tf.reduce_mean(tf.cast(trace["is_accepted"], tf.float32)))

    # ESS per parameter
    ess = tfp.mcmc.effective_sample_size(samples, filter_beyond_positive_pairs=True)
    ess_per_param = float(tf.reduce_mean(ess))

    return {
        "acceptance_rate": acceptance_rate,
        "ess_per_param": ess_per_param,
    }


def main():
    print("=" * 70)
    print("LEDH Surrogate-Force HMC Damping Calibration")
    print("Following: ledh-surrogate-hmc-phase3-task3.2-calibration-plan")
    print("=" * 70)

    # Load LGSSM T=50 fixture
    print("\n[1/6] Loading LGSSM T=50 fixture...")
    observations = _lgssm_frozen_observations()
    print(f"  Observations shape: {observations.shape}")
    print(f"  Observations dtype: {observations.dtype}")

    # Create model
    print("\n[2/6] Creating LGSSM model...")
    model = _diagonal_lgssm_fused_model()
    print(f"  Model type: {type(model).__name__}")

    # Fixed parameters per calibration plan
    d = 3
    N = 1008  # Per plan specification
    T = observations.shape[0]
    dtype = observations.dtype

    print(f"  State dim: {d}")
    print(f"  Particles: {N}")
    print(f"  Horizon: {T}")

    # Initialize particles (match dtype)
    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=dtype) * 0.1
    initial_covariances = tf.tile(
        tf.eye(d, dtype=dtype)[None, :, :], [N, 1, 1]
    ) * 0.01

    # Frozen noises (match dtype) - shape [T, N, d] for per-point model
    noises = generator.normal([T, N, d], dtype=dtype) * 0.1

    # Shared parameters (per calibration plan)
    reset_basis = tf.concat(
        [tf.eye(d, dtype=dtype), -tf.eye(d, dtype=dtype)],
        axis=0,
    )
    reset_repeats = (N + 2 * d - 1) // (2 * d)
    reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:N]

    shared_params = dict(
        substeps=8,
        reset_policy="contract_e",
        reset_design=reset_design,
        reset_epsilon=2.0,
        reset_sinkhorn_steps=8,
        reset_balance_steps=8,
        correction_steps=4,
        correction_strength=0.2,
        correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5,
        pairwise_steps=4,
        pairwise_strength=0.02,
        pairwise_rms_cap=2.0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )

    # Create targets for 4-arm sweep
    print("\n[3/6] Creating dual-parameter targets for 4 arms...")

    damping_ratios = [1.0, 10.0, 100.0, 1000.0]
    base_reset_ridge = 1e-5
    base_lm_damping = 1e-2

    targets = {}
    for ratio in damping_ratios:
        targets[ratio] = make_dual_parameter_target(
            model=model,
            initial_states=initial_states,
            initial_covariances=initial_covariances,
            noises=noises,
            observations=observations,
            damping_ratio=ratio,
            base_reset_ridge=base_reset_ridge,
            base_lm_damping=base_lm_damping,
            **shared_params,
        )
        print(f"  ✓ {ratio:.0f}× damping target")

    # HMC parameters (per plan: screening run, not convergence study)
    num_chains = 2
    num_burnin_steps = 500
    num_results = 500
    step_size = 0.01
    num_leapfrog_steps = 10
    base_seed = 97701

    # True theta (LGSSM parameters)
    true_theta = tf.constant([1.0, 1.0, 1.0, 0.5, 0.3], dtype=dtype)

    print(f"\n[4/6] HMC configuration:")
    print(f"  Chains: {num_chains}")
    print(f"  Burn-in: {num_burnin_steps}")
    print(f"  Samples: {num_results}")
    print(f"  Initial step size: {step_size}")
    print(f"  Leapfrog steps: {num_leapfrog_steps}")
    print(f"  Note: EAGER mode (no tf.function graph compilation)")

    # Run calibration sweep
    print(f"\n[5/6] Running 4-arm calibration sweep...")

    results = {}
    for ratio in damping_ratios:
        print(f"\n  --- {ratio:.0f}× damping ---")
        target = targets[ratio]

        arm_results = {
            "samples": [],
            "traces": [],
            "wall_times": [],
            "diagnostics": [],
        }

        for chain_idx in range(num_chains):
            chain_seed = base_seed + int(ratio * 10) + chain_idx
            print(f"    Chain {chain_idx + 1}/{num_chains} (seed={chain_seed})...", end=" ", flush=True)

            # Initial state (perturbed true theta)
            init_generator = tf.random.Generator.from_seed(chain_seed)
            initial_theta = true_theta + init_generator.normal([5], dtype=dtype) * 0.1

            # Run HMC (eager mode)
            t0 = time.time()
            samples, trace = run_hmc_chain_eager(
                target_log_prob_fn=target,
                initial_theta=initial_theta,
                num_burnin_steps=num_burnin_steps,
                num_results=num_results,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
                seed=chain_seed,
            )
            wall_time = time.time() - t0

            # Compute diagnostics
            diagnostics = compute_diagnostics(samples, trace)

            arm_results["samples"].append(samples)
            arm_results["traces"].append(trace)
            arm_results["wall_times"].append(wall_time)
            arm_results["diagnostics"].append(diagnostics)

            print(f"accept={diagnostics['acceptance_rate']:.3f}, "
                  f"ESS={diagnostics['ess_per_param']:.0f}, "
                  f"time={wall_time:.1f}s")

        results[ratio] = arm_results

    # Aggregate and report
    print(f"\n[6/6] Summary:")
    print("\n  Damping  Accept  ESS/param  ESS Ratio  Wall Time")
    print("  -------  ------  ---------  ---------  ---------")

    baseline_ess = None
    for ratio in damping_ratios:
        arm = results[ratio]
        mean_accept = np.mean([d["acceptance_rate"] for d in arm["diagnostics"]])
        mean_ess = np.mean([d["ess_per_param"] for d in arm["diagnostics"]])
        mean_time = np.mean(arm["wall_times"])

        if baseline_ess is None:
            baseline_ess = mean_ess

        ess_ratio = mean_ess / baseline_ess if baseline_ess > 0 else 0.0

        status = "✓" if mean_accept >= 0.15 and ess_ratio >= 0.3 else "✗"

        print(f"  {ratio:6.0f}×  {mean_accept:.3f}  {mean_ess:9.0f}     {ess_ratio:.2f}    {mean_time:7.1f}s  {status}")

    print("\n  Criteria:")
    print("    Acceptance rate: ≥ 0.15")
    print("    ESS/grad ratio: ≥ 0.3× baseline")

    # Save results
    output_dir = Path("docs/plans/artifacts/ledh-surrogate-hmc-damping-calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "calibration_results.npz"
    np.savez(
        output_file,
        damping_ratios=damping_ratios,
        **{
            f"samples_{int(ratio)}x": np.array([s.numpy() for s in results[ratio]["samples"]])
            for ratio in damping_ratios
        },
        **{
            f"diagnostics_{int(ratio)}x": {
                "acceptance_rate": [d["acceptance_rate"] for d in results[ratio]["diagnostics"]],
                "ess_per_param": [d["ess_per_param"] for d in results[ratio]["diagnostics"]],
                "wall_time": results[ratio]["wall_times"],
            }
            for ratio in damping_ratios
        },
    )

    print(f"\n  ✓ Results saved to {output_file}")

    print("\n" + "=" * 70)
    print("✓ CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
