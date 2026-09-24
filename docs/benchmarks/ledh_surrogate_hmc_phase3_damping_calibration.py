#!/usr/bin/env python
"""Phase 3 Task 3.2: Damping calibration curve for surrogate-force HMC.

Measures acceptance rate and ESS/gradient across damping ratios to find the
optimal balance between mixing quality and computational cost.

Authority: Phase 3 Task 3.2 of ledh-surrogate-hmc-unified-program-2026-09-06.md
Plan: docs/plans/ledh-surrogate-hmc-phase3-task3.2-calibration-plan-2026-09-11.md
Date: 2026-09-12
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.ledh_dual_parameter_target import (
    make_dual_parameter_target,
)
from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _lgssm_frozen_observations,
    _diagonal_lgssm_fused_model,
)


def main():
    print("=" * 70)
    print("LEDH Surrogate-Force HMC: Phase 3 Task 3.2 Damping Calibration")
    print("=" * 70)

    # Load LGSSM T=50 fixture
    print("\n[1/7] Loading LGSSM T=50 fixture...")
    observations = _lgssm_frozen_observations()
    print(f"  Observations shape: {observations.shape}")
    print(f"  Observations dtype: {observations.dtype}")

    # Create model
    print("\n[2/7] Creating LGSSM model...")
    model = _diagonal_lgssm_fused_model()
    print(f"  Model type: {type(model).__name__}")

    # Fixed parameters for calibration
    d = 3
    N = 1008  # Per calibration plan
    T = observations.shape[0]
    dtype = observations.dtype  # Match observations dtype

    # Initialize particles
    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=dtype) * 0.1
    initial_covariances = tf.tile(
        tf.eye(d, dtype=dtype)[None, :, :], [N, 1, 1]
    ) * 0.01

    # Frozen noises
    noises = generator.normal([T, N, d], dtype=dtype) * 0.1

    # True theta (5 params: 3 AR coefficients, 2 covariance scales)
    theta_true = tf.constant([0.5, 0.3, 0.2, 0.1, 0.05], dtype=dtype)

    print(f"  State dim: {d}")
    print(f"  Particles: {N}")
    print(f"  Horizon: {T}")
    print(f"  Theta dim: {theta_true.shape[0]}")

    # Shared parameters (per calibration plan)
    shared_params = dict(
        substeps=8,
        reset_policy="contract_e",
        reset_design=tf.eye(d, dtype=dtype),
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

    # Damping ratios to test
    damping_ratios = [1.0, 10.0, 100.0, 1000.0]

    print(f"\n[3/7] Creating targets for {len(damping_ratios)} damping ratios...")

    targets = {}
    for ratio in damping_ratios:
        target = make_dual_parameter_target(
            model=model,
            initial_states=initial_states,
            initial_covariances=initial_covariances,
            noises=noises,
            observations=observations,
            damping_ratio=ratio,
            base_reset_ridge=1e-5,
            base_lm_damping=1e-2,
            **shared_params,
        )
        targets[ratio] = target
        print(f"  ✓ {ratio}× damping")

    # HMC configuration
    num_chains = 2
    num_warmup = 500
    num_samples = 500
    step_size = 0.01
    hmc_seed = 97701

    print(f"\n[4/7] HMC configuration:")
    print(f"  Chains: {num_chains}")
    print(f"  Warmup: {num_warmup}")
    print(f"  Samples: {num_samples}")
    print(f"  Step size: {step_size}")
    print(f"  Seed: {hmc_seed}")

    # Run calibration sweep
    print(f"\n[5/7] Running HMC calibration sweep...")
    results = {}

    for ratio in damping_ratios:
        print(f"\n  --- Damping ratio: {ratio}× ---")
        target = targets[ratio]

        # Wrap target for TFP HMC
        @tf.function
        def neg_log_prob_fn(theta_batch):
            values, _ = target(theta_batch)
            return -values

        @tf.function
        def grad_neg_log_prob_fn(theta_batch):
            _, scores = target(theta_batch)
            return -scores

        # Initial state for chains
        initial_state = tf.stack([theta_true + 0.01 * generator.normal([5], dtype=dtype)
                                   for _ in range(num_chains)], axis=0)

        # HMC kernel
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=neg_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=10,
            grad_target_log_prob_fn=grad_neg_log_prob_fn,
        )

        # Adaptive step size (only during warmup)
        adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            kernel,
            num_adaptation_steps=int(0.8 * num_warmup),
            target_accept_prob=0.65,
        )

        # Run HMC
        print(f"    Running {num_warmup + num_samples} steps...")
        start_time = time.time()

        @tf.function
        def run_chain():
            return tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_warmup,
                current_state=initial_state,
                kernel=adaptive_kernel,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                seed=hmc_seed,
            )

        samples, is_accepted = run_chain()
        elapsed = time.time() - start_time

        print(f"    Completed in {elapsed:.1f}s")

        # Compute diagnostics
        acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy()
        print(f"    Acceptance rate: {acceptance_rate:.3f}")

        # ESS per parameter
        ess_per_param = tfp.mcmc.effective_sample_size(samples).numpy()
        mean_ess = np.mean(ess_per_param)
        print(f"    Mean ESS: {mean_ess:.1f}")

        # ESS per gradient evaluation
        num_gradients = num_chains * (num_warmup + num_samples) * 10  # 10 leapfrog steps
        ess_per_gradient = mean_ess / num_gradients
        print(f"    ESS per gradient: {ess_per_gradient:.6f}")

        # R-hat (convergence diagnostic)
        rhat = tfp.mcmc.potential_scale_reduction(samples).numpy()
        max_rhat = np.max(rhat)
        print(f"    Max R-hat: {max_rhat:.4f}")

        # Check for divergences (proxy: very low acceptance)
        divergences = tf.reduce_sum(tf.cast(~is_accepted, tf.int32)).numpy()
        print(f"    Divergences (rejections): {divergences}/{is_accepted.shape[0] * is_accepted.shape[1]}")

        # Record results
        results[ratio] = {
            'acceptance_rate': float(acceptance_rate),
            'ess_per_param': ess_per_param.tolist(),
            'mean_ess': float(mean_ess),
            'ess_per_gradient': float(ess_per_gradient),
            'num_gradients': int(num_gradients),
            'rhat': rhat.tolist(),
            'max_rhat': float(max_rhat),
            'divergences': int(divergences),
            'total_steps': int(is_accepted.shape[0] * is_accepted.shape[1]),
            'elapsed_seconds': float(elapsed),
        }

    # Analysis and selection
    print(f"\n[6/7] Analyzing results...")
    print(f"\n  Summary:")
    print(f"  {'Ratio':<10} {'Accept':<10} {'ESS/grad':<12} {'Max R-hat':<12} {'Status'}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*20}")

    baseline_ess_per_grad = results[1.0]['ess_per_gradient']

    for ratio in damping_ratios:
        r = results[ratio]
        accept = r['acceptance_rate']
        ess_grad = r['ess_per_gradient']
        rhat = r['max_rhat']

        # Check criteria
        pass_accept = accept >= 0.15
        pass_ess = ess_grad >= 0.3 * baseline_ess_per_grad
        pass_rhat = rhat <= 1.1

        if pass_accept and pass_ess and pass_rhat:
            status = "✓ PASS"
        else:
            reasons = []
            if not pass_accept:
                reasons.append("accept")
            if not pass_ess:
                reasons.append("ESS")
            if not pass_rhat:
                reasons.append("R-hat")
            status = f"✗ FAIL ({', '.join(reasons)})"

        print(f"  {ratio:<10.0f} {accept:<10.3f} {ess_grad:<12.6f} {rhat:<12.4f} {status}")

    # Select optimal damping
    passing = [
        r for r in damping_ratios
        if results[r]['acceptance_rate'] >= 0.15
        and results[r]['ess_per_gradient'] >= 0.3 * baseline_ess_per_grad
        and results[r]['max_rhat'] <= 1.1
    ]

    print(f"\n  Passing damping ratios: {passing if passing else 'None'}")

    if passing:
        selected = max(passing)  # Coarsest (largest) damping
        print(f"  ✓ Selected damping: {selected}×")
        print(f"    (Coarsest damping that passes all criteria)")
    else:
        selected = None
        print(f"  ✗ No damping ratio passes all criteria")
        print(f"    Recommendation: Relax criteria or try intermediate ratios")

    # Save results
    print(f"\n[7/7] Saving results...")
    artifact_dir = Path("docs/benchmarks/artifacts/ledh-damping-calibration-lgssm-t50-20260912")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    result_data = {
        'selected_damping': selected,
        'damping_ratios': damping_ratios,
        'results': results,
        'criteria': {
            'min_acceptance': 0.15,
            'min_ess_per_grad_fraction': 0.3,
            'max_rhat': 1.1,
        },
        'baseline_ess_per_grad': baseline_ess_per_grad,
    }

    result_path = artifact_dir / "result.json"
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"  ✓ Saved: {result_path}")

    # Manifest
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).parent.parent.parent,
            text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    manifest = {
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_commit': git_commit,
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
        'seeds': {
            'fixture': 81100,
            'hmc': hmc_seed,
        },
        'config': {
            'num_chains': num_chains,
            'num_warmup': num_warmup,
            'num_samples': num_samples,
            'step_size': step_size,
            'particles': N,
            'horizon': T,
        },
    }

    manifest_path = artifact_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ Saved: {manifest_path}")

    # Summary
    print("\n" + "=" * 70)
    if selected:
        print(f"✓ CALIBRATION COMPLETE")
        print(f"  Selected damping: {selected}×")
        print(f"  Acceptance: {results[selected]['acceptance_rate']:.3f}")
        print(f"  ESS/grad: {results[selected]['ess_per_gradient']:.6f}")
        print(f"  Max R-hat: {results[selected]['max_rhat']:.4f}")
        print(f"\nNext: Proceed to Phase 4 with {selected}× damping")
        return 0
    else:
        print(f"✗ NO DAMPING RATIO PASSED")
        print(f"  All tested ratios failed one or more criteria")
        print(f"\nNext: Review results and decide:")
        print(f"  - Try intermediate ratios (30×, 50×)")
        print(f"  - Relax acceptance floor (e.g., 0.10)")
        print(f"  - Abandon surrogate-force approach for this model")
        return 1


if __name__ == "__main__":
    sys.exit(main())
