"""Phase 3 seed-policy verification tests for LEDH surrogate-force HMC.

V1: Determinism — same master_seed → bitwise identical forces
V2: Reversibility (involution) — forward-backward-forward = identity
V3: No call-count dependence — force(θ, call=N) = force(θ, call=N+M)

These tests verify the Corollary 5.2 premise that F is a deterministic function
of θ alone, with no dependence on trajectory history, call order, or hidden state.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.ledh_dual_force_adapter import (
    LEDHDualForceAdapter,
    create_simple_lgssm_fixture,
)


def test_v1_determinism_same_seed_gives_bitwise_identical_force():
    """
    V1: Determinism check.

    Two adapters with the same master_seed must produce bitwise identical
    forces at the same θ. This verifies ω is frozen and F depends only on θ.
    """
    observations = create_simple_lgssm_fixture(T=4, seed=8001)

    adapter1 = LEDHDualForceAdapter(
        observations, num_particles=8, master_seed=42
    )
    adapter2 = LEDHDualForceAdapter(
        observations, num_particles=8, master_seed=42
    )

    theta = tf.constant([0.5, 0.2, -0.1, np.log(0.5), np.log(0.5)], dtype=tf.float32)

    force1 = adapter1.exact_force(theta)
    force2 = adapter2.exact_force(theta)

    # Bitwise equality
    np.testing.assert_array_equal(
        force1.numpy(),
        force2.numpy(),
        err_msg="V1 FAIL: Same master_seed produced different forces",
    )

    print("✓ V1 PASS: Determinism verified (bitwise equality)")


def test_v1_different_seed_gives_different_force():
    """
    V1 control: Different seeds must produce different forces.

    If this fails, the seed isn't actually affecting ω.
    """
    observations = create_simple_lgssm_fixture(T=4, seed=8002)

    adapter1 = LEDHDualForceAdapter(
        observations, num_particles=8, master_seed=42
    )
    adapter2 = LEDHDualForceAdapter(
        observations, num_particles=8, master_seed=43
    )

    theta = tf.constant([0.5, 0.2, -0.1, np.log(0.5), np.log(0.5)], dtype=tf.float32)

    force1 = adapter1.exact_force(theta)
    force2 = adapter2.exact_force(theta)

    assert not np.allclose(
        force1.numpy(), force2.numpy(), rtol=1e-10, atol=1e-12
    ), "V1 control FAIL: Different seeds gave identical forces"

    print("✓ V1 control: Different seeds produce different forces")


def test_v2_reversibility_forward_backward_forward_is_identity():
    """
    V2: Reversibility (involution) check.

    Standard HMC correctness: momentum flip must reverse the trajectory.
    Forward L steps, flip momentum, backward L steps, flip again → should
    return to the starting point within integrator tolerance.

    This is strictly stronger than V1 — V1 only tests F at the same θ twice
    in a row, while V2 tests F along an entire trajectory.
    """
    observations = create_simple_lgssm_fixture(T=4, seed=8003)

    adapter = LEDHDualForceAdapter(
        observations, num_particles=8, master_seed=42
    )

    # Initial state
    theta0 = tf.constant([0.5, 0.2, -0.1, np.log(0.5), np.log(0.5)], dtype=tf.float32)
    momentum0 = tf.constant([0.5, -0.3, 0.2, 0.1, -0.15], dtype=tf.float32)

    step_size = 0.01
    num_steps = 10

    # Helper: single leapfrog step
    def leapfrog_step(theta, momentum, force_fn):
        force = force_fn(theta)
        momentum = momentum + 0.5 * step_size * force

        for _ in range(num_steps - 1):
            theta = theta + step_size * momentum
            force = force_fn(theta)
            momentum = momentum + step_size * force

        theta = theta + step_size * momentum
        force = force_fn(theta)
        momentum = momentum + 0.5 * step_size * force

        return theta, momentum

    # Forward
    theta_fwd, momentum_fwd = leapfrog_step(theta0, momentum0, adapter.exact_force)

    # Reverse momentum, backward
    theta_bwd, momentum_bwd = leapfrog_step(
        theta_fwd, -momentum_fwd, adapter.exact_force
    )

    # Reverse momentum again
    theta_final = theta_bwd
    momentum_final = -momentum_bwd

    # Check: should equal (theta0, momentum0) within integrator tolerance
    np.testing.assert_allclose(
        theta_final.numpy(),
        theta0.numpy(),
        rtol=1e-6,
        atol=1e-8,
        err_msg="V2 FAIL: Reversibility violated (theta mismatch)",
    )

    np.testing.assert_allclose(
        momentum_final.numpy(),
        momentum0.numpy(),
        rtol=1e-6,
        atol=1e-8,
        err_msg="V2 FAIL: Reversibility violated (momentum mismatch)",
    )

    print("✓ V2 PASS: Reversibility verified (involution within tolerance)")


def test_v3_no_call_count_dependence():
    """
    V3: No call-count dependence.

    Force at the same θ must be identical regardless of how many times the
    force function has been called previously. This verifies no hidden state
    (call counter, cached clouds, accumulated float state).
    """
    observations = create_simple_lgssm_fixture(T=4, seed=8004)

    adapter = LEDHDualForceAdapter(
        observations, num_particles=8, master_seed=42
    )

    theta = tf.constant([0.5, 0.2, -0.1, np.log(0.5), np.log(0.5)], dtype=tf.float32)

    # Call 1
    adapter.reset_call_count()
    force_call1 = adapter.exact_force(theta)
    assert adapter.get_call_count() == 1

    # Call force at 50 other θ values to change internal state if any
    for i in range(50):
        theta_other = tf.constant(
            [0.5 + 0.01 * i, 0.2, -0.1, np.log(0.5) - 0.01 * i, np.log(0.5)],
            dtype=tf.float32,
        )
        _ = adapter.exact_force(theta_other)

    # Call at original θ again (call #52)
    force_call52 = adapter.exact_force(theta)
    assert adapter.get_call_count() == 52

    # Should be bitwise identical
    np.testing.assert_allclose(
        force_call1.numpy(),
        force_call52.numpy(),
        rtol=1e-12,
        atol=1e-14,
        err_msg="V3 FAIL: Force depends on call count",
    )

    print("✓ V3 PASS: No call-count dependence (force is stateless)")


def test_exact_and_damped_forces_differ_by_damping_factor():
    """
    Sanity check: damped force = exact force / (1 + epsilon).

    Not a Corollary 5.2 premise, but verifies the adapter is wired correctly.
    """
    observations = create_simple_lgssm_fixture(T=4, seed=8005)

    epsilon = 0.01
    adapter = LEDHDualForceAdapter(
        observations,
        num_particles=8,
        master_seed=42,
        damping_epsilon=epsilon,
    )

    theta = tf.constant([0.5, 0.2, -0.1, np.log(0.5), np.log(0.5)], dtype=tf.float32)

    exact = adapter.exact_force(theta)
    damped = adapter.damped_force(theta)

    expected_damped = exact / (1.0 + epsilon)

    np.testing.assert_allclose(
        damped.numpy(),
        expected_damped.numpy(),
        rtol=1e-10,
        atol=1e-12,
        err_msg="Sanity check FAIL: damped force formula is wrong",
    )

    print("✓ Sanity check: damped = exact / (1 + epsilon)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
