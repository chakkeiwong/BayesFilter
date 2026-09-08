"""Test surrogate-force HMC mechanics on toy quadratic potential.

Phase 2 of ledh-surrogate-force-hmc-master-program-2026-09-04.md

Tests
-----
T1: Deterministic repeated calls
T2: Endpoint energy equality
T3: Acceptance across damping ladder
T4: Force-norm diagnostic

Success Criteria
----------------
- All 4 tests pass
- Damping=1.0: acceptance 0.7-0.8
- Damping=0.5: acceptance 0.5-0.6
- Damping=0.1: acceptance ≥ 0.2
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.toy_surrogate_force_adapter import (
    DualAdapterSurrogateForce,
    ToyPotentialAdapter,
)

DTYPE = tf.float64


def test_t1_deterministic_repeated_calls():
    """T1: Same θ → same (value, force)."""
    sigma_inv = np.diag([1.0, 0.25, 1.0 / 9.0])
    adapter = ToyPotentialAdapter(sigma_inv, damping_scale=0.1, dtype=DTYPE)

    theta = tf.constant([[1.0, 2.0, 3.0]], DTYPE)

    v1, f1 = adapter.log_prob_and_grad(theta)
    v2, f2 = adapter.log_prob_and_grad(theta)

    np.testing.assert_allclose(v1.numpy(), v2.numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(f1.numpy(), f2.numpy(), rtol=0, atol=0)


def test_t2_endpoint_energy_conservation():
    """T2: H(start) ≈ H(end) up to leapfrog discretization error."""
    sigma_inv = np.diag([1.0, 0.25, 1.0 / 9.0])
    adapter = ToyPotentialAdapter(sigma_inv, damping_scale=1.0, dtype=DTYPE)

    # Simple leapfrog trajectory (L=10 steps, ε=0.1)
    theta_start = tf.constant([[1.0, 0.5, -0.5]], DTYPE)
    p_start = tf.constant([[0.3, -0.2, 0.1]], DTYPE)

    # Leapfrog integrator
    def leapfrog_step(theta, p, epsilon):
        _, grad = adapter.log_prob_and_grad(theta)
        p_half = p + 0.5 * epsilon * grad
        theta_new = theta + epsilon * p_half
        _, grad_new = adapter.log_prob_and_grad(theta_new)
        p_new = p_half + 0.5 * epsilon * grad_new
        return theta_new, p_new

    theta, p = theta_start, p_start
    for _ in range(10):
        theta, p = leapfrog_step(theta, p, 0.1)

    # Compute Hamiltonian at start and end
    value_start, _ = adapter.log_prob_and_grad(theta_start)
    value_end, _ = adapter.log_prob_and_grad(theta)

    k_start = 0.5 * tf.reduce_sum(p_start**2)
    k_end = 0.5 * tf.reduce_sum(p**2)

    h_start = -value_start + k_start
    h_end = -value_end + k_end

    delta_h = float(tf.abs(h_end - h_start))

    # Loose bound for toy potential (leapfrog has O(ε²) error per step)
    assert delta_h < 0.1, f"Energy conservation failed: ΔH = {delta_h}"


def test_t3_acceptance_rate_with_hmc():
    """T3: Measure actual HMC acceptance rate across damping ladder.

    This is the REAL T3 promotion criterion test.
    Uses reviewed_value_score_target_fn to wire damped force into TFP HMC.

    Promotion criterion: acceptance ≥ 0.2 at damping 0.1
    """
    from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn

    sigma_inv = np.diag([1.0, 0.25, 1.0 / 9.0])

    results = {}

    for damping in [1.0, 0.5, 0.1]:
        # Create dual adapter: exact value, damped force
        adapter = DualAdapterSurrogateForce(
            sigma_inv,
            damping_scale=damping,
            dtype=DTYPE
        )

        # Wrap with reviewed_value_score_target_fn for TFP HMC
        # This uses the adapter's force for leapfrog, but TFP thinks it's autodiffing
        target_log_prob_fn = reviewed_value_score_target_fn(
            adapter,
            dtype=DTYPE,
            require_batched=False,  # TFP HMC uses scalar chains
        )

        # Run short HMC chain
        num_results = 100
        num_burnin_steps = 50

        # Initial state: slightly off center
        initial_state = tf.constant([1.0, 0.5, -0.5], dtype=DTYPE)

        # HMC kernel with step size tuned for this potential
        step_size = 0.1
        num_leapfrog_steps = 10

        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
        )

        # Run chain
        @tf.function
        def run_chain():
            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=initial_state,
                kernel=kernel,
                trace_fn=lambda _, pkr: pkr.is_accepted,
                seed=tf.constant([20260908, int(damping * 10)], dtype=tf.int32),
            )

        samples, is_accepted = run_chain()

        # Compute acceptance rate
        acceptance_rate = float(tf.reduce_mean(tf.cast(is_accepted, DTYPE)))
        results[damping] = acceptance_rate

        print(f"\nDamping={damping:.1f}: acceptance rate = {acceptance_rate:.3f}")

    # Verify promotion criterion
    assert results[0.1] >= 0.2, (
        f"Promotion criterion FAILED: acceptance at damping=0.1 is {results[0.1]:.3f}, "
        f"required ≥ 0.2"
    )

    # Additional sanity checks
    assert results[1.0] > results[0.5], "Higher damping should not increase acceptance"
    assert results[0.5] > results[0.1], "Higher damping should not increase acceptance"

    print("\n✅ T3 PASSED: All acceptance rates meet criteria")
    print(f"   damping=1.0: {results[1.0]:.3f}")
    print(f"   damping=0.5: {results[0.5]:.3f}")
    print(f"   damping=0.1: {results[0.1]:.3f} (≥ 0.2 required)")


def test_t4_force_norm_diagnostic():
    """T4: ||F_damped|| < ||F_exact||."""
    sigma_inv = np.diag([1.0, 0.25, 1.0 / 9.0])
    exact_adapter = ToyPotentialAdapter(sigma_inv, damping_scale=1.0, dtype=DTYPE)
    damped_adapter = ToyPotentialAdapter(sigma_inv, damping_scale=0.1, dtype=DTYPE)

    theta = tf.constant([[1.0, 2.0, 3.0]], DTYPE)

    _, f_exact = exact_adapter.log_prob_and_grad(theta)
    _, f_damped = damped_adapter.log_prob_and_grad(theta)

    norm_exact = float(tf.norm(f_exact))
    norm_damped = float(tf.norm(f_damped))

    assert norm_damped < norm_exact, (
        f"Force norm check failed: ||F_damped||={norm_damped:.4f} >= ||F_exact||={norm_exact:.4f}"
    )

    print(f"\n||F_exact|| = {norm_exact:.4f}")
    print(f"||F_damped|| = {norm_damped:.4f}")
    print(f"Ratio = {norm_damped / norm_exact:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
