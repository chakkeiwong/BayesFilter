#!/usr/bin/env python
"""Spot-check dual-parameter adapter before full calibration.

Verifies:
1. Adapter produces finite values and scores
2. Values identical across damping ratios (exact params used)
3. Scores differ across damping ratios (biased params differ)

Authority: Phase 3 Task 3.2 spot-check requirement
Date: 2026-09-11
"""

import numpy as np
import tensorflow as tf

from bayesfilter.inference.ledh_dual_parameter_target import (
    make_dual_parameter_target,
)
from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _lgssm_frozen_observations,
    _diagonal_lgssm_fused_model,
)


def main():
    print("=" * 70)
    print("Dual-Parameter Adapter Spot-Check")
    print("=" * 70)

    # Load LGSSM T=50 fixture
    print("\n[1/5] Loading LGSSM T=50 fixture...")
    observations = _lgssm_frozen_observations()
    print(f"  Observations shape: {observations.shape}")
    print(f"  Observations dtype: {observations.dtype}")

    # Create model
    print("\n[2/5] Creating LGSSM model...")
    model = _diagonal_lgssm_fused_model()
    print(f"  Model type: {type(model).__name__}")

    # Fixed parameters for spot-check
    d = 3
    N = 252  # Smaller for spot-check
    T = observations.shape[0]
    dtype = observations.dtype  # Use float64 to match observations

    # Initialize particles (match dtype)
    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=dtype) * 0.1
    initial_covariances = tf.tile(
        tf.eye(d, dtype=dtype)[None, :, :], [N, 1, 1]
    ) * 0.01

    # Frozen noises (match dtype) - shape [T, N, d] for per-point model
    # Per-point model expects theta_rows [M, P], so noises stay [T, N, d]
    noises = generator.normal([T, N, d], dtype=dtype) * 0.1

    # True theta (5 params: 3 AR coefficients, 2 covariance scales)
    theta_true = tf.constant([0.5, 0.3, 0.2, 0.1, 0.05], dtype=dtype)

    print(f"  State dim: {d}")
    print(f"  Particles: {N}")
    print(f"  Horizon: {T}")
    print(f"  Theta dim: {theta_true.shape[0]}")

    # Shared parameters (use matched dtype for reset_design)
    # Design must be [N, d] tiled basis (canonical pattern from NeuTra factory)
    reset_basis = tf.concat(
        [tf.eye(d, dtype=dtype), -tf.eye(d, dtype=dtype)],
        axis=0,
    )
    reset_repeats = (N + 2 * d - 1) // (2 * d)
    reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:N]

    shared_params = dict(
        substeps=8,
        reset_policy="contract_e",
        reset_design=reset_design,  # [N, d] shape
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

    # Create targets with different damping ratios
    print("\n[3/5] Creating dual-parameter targets...")

    target_1x = make_dual_parameter_target(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        damping_ratio=1.0,
        base_reset_ridge=1e-5,
        base_lm_damping=1e-2,
        **shared_params,
    )
    print("  ✓ 1× target (baseline)")

    target_100x = make_dual_parameter_target(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        damping_ratio=100.0,
        base_reset_ridge=1e-5,
        base_lm_damping=1e-2,
        **shared_params,
    )
    print("  ✓ 100× target (damped)")

    # Evaluate at theta_true
    print("\n[4/5] Evaluating at true theta...")

    print("  Computing 1× (baseline)...")
    value_1x, score_1x = target_1x(theta_true)
    print(f"    Value: {value_1x.numpy():.6f}")
    print(f"    Score L2: {tf.norm(score_1x).numpy():.6f}")

    print("  Computing 100× (damped)...")
    value_100x, score_100x = target_100x(theta_true)
    print(f"    Value: {value_100x.numpy():.6f}")
    print(f"    Score L2: {tf.norm(score_100x).numpy():.6f}")

    # Verification checks
    print("\n[5/5] Running verification checks...")

    passed = True

    # Check 1: Finite values
    if not tf.math.is_finite(value_1x):
        print("  ✗ FAIL: 1× value is not finite")
        passed = False
    else:
        print("  ✓ PASS: 1× value is finite")

    if not tf.math.is_finite(value_100x):
        print("  ✗ FAIL: 100× value is not finite")
        passed = False
    else:
        print("  ✓ PASS: 100× value is finite")

    # Check 2: Finite scores
    if not tf.reduce_all(tf.math.is_finite(score_1x)):
        print("  ✗ FAIL: 1× score contains non-finite values")
        passed = False
    else:
        print("  ✓ PASS: 1× score is finite")

    if not tf.reduce_all(tf.math.is_finite(score_100x)):
        print("  ✗ FAIL: 100× score contains non-finite values")
        passed = False
    else:
        print("  ✓ PASS: 100× score is finite")

    # Check 3: Values should be identical (both use exact params)
    value_diff = abs(value_1x.numpy() - value_100x.numpy())
    if value_diff > 1e-6:
        print(f"  ✗ FAIL: Values differ by {value_diff:.2e} (expected identical)")
        passed = False
    else:
        print(f"  ✓ PASS: Values identical (diff={value_diff:.2e})")

    # Check 4: Scores should differ (biased params differ)
    score_diff = tf.norm(score_100x - score_1x).numpy()
    if score_diff < 1e-6:
        print(f"  ✗ FAIL: Scores too similar (diff={score_diff:.2e})")
        print("         Damping may not be working correctly")
        passed = False
    else:
        print(f"  ✓ PASS: Scores differ (L2 diff={score_diff:.6f})")

        # Show relative change
        score_1x_norm = tf.norm(score_1x).numpy()
        rel_change = (score_diff / score_1x_norm) * 100
        print(f"         Relative score change: {rel_change:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    if passed:
        print("✓ ALL CHECKS PASSED")
        print("Adapter is working correctly. Safe to proceed with calibration.")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("Do not proceed with calibration until adapter is fixed.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
