"""HMC integration test for LEDH graceful failure.

Tests that HMC with LEDH target function handles pathological proposals
gracefully by returning -inf with zero gradient, allowing MH rejection.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def test_hmc_graceful_failure_pathological_target():
    """Test HMC continues after encountering pathological proposals."""
    dtype = tf.float64

    # Create a target that returns -inf for negative parameters
    # This simulates LEDH canonical score returning -inf for invalid states
    @tf.custom_gradient
    def pathological_target(theta):
        # Valid region: theta > 0
        # Invalid region: theta <= 0 (returns -inf)
        is_valid = theta[0] > 0.0

        # In valid region: simple quadratic centered at 2.0
        valid_value = -0.5 * tf.square(theta[0] - 2.0)

        # Sentinel value for invalid region
        invalid_value = tf.constant(float("-inf"), dtype)

        value = tf.where(is_valid, valid_value, invalid_value)

        def grad_fn(dy):
            # Valid gradient: -(theta - 2.0)
            valid_grad = -(theta[0] - 2.0)
            # Invalid gradient: zero
            invalid_grad = tf.constant(0.0, dtype)
            gradient = tf.where(is_valid, valid_grad, invalid_grad)
            return dy * tf.stack([gradient])

        return value, grad_fn

    # HMC setup
    initial_state = tf.constant([3.0], dtype=dtype)  # Start in valid region

    # Single HMC step with large step size to potentially hit invalid region
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=pathological_target,
        step_size=0.5,
        num_leapfrog_steps=10,
    )

    # Run one step
    next_state, kernel_results = kernel.one_step(
        initial_state,
        tfp.mcmc.HamiltonianMonteCarlo.bootstrap_results(kernel, initial_state)
    )

    # Should complete without crashing
    assert not tf.math.is_nan(next_state[0]).numpy(), "HMC should not produce NaN"

    # If proposal went invalid, it should be rejected and state should stay same
    # or move to a different valid point
    assert next_state[0].numpy() > 0.0 or tf.equal(next_state[0], initial_state[0]).numpy(), \
        "HMC should reject invalid proposals or stay in valid region"


def test_dual_parameter_target_handles_neg_inf():
    """Test that custom gradient propagates -inf correctly."""
    dtype = tf.float64

    @tf.custom_gradient
    def target_with_custom_grad(theta):
        # Return -inf for theta < 0
        is_valid = theta[0] >= 0.0
        value = tf.where(is_valid, -tf.square(theta[0] - 1.0), tf.constant(float("-inf"), dtype))

        def grad_fn(dy):
            # Zero gradient when invalid, surrogate gradient when valid
            surrogate_grad = -2.0 * (theta[0] - 1.0)
            gradient = tf.where(is_valid, surrogate_grad, tf.constant(0.0, dtype))
            return dy * tf.stack([gradient])

        return value, grad_fn

    # Test with valid parameter
    theta_valid = tf.constant([1.5], dtype=dtype)
    with tf.GradientTape() as tape:
        tape.watch(theta_valid)
        value = target_with_custom_grad(theta_valid)
    grad = tape.gradient(value, theta_valid)

    assert not tf.math.is_inf(value).numpy(), "Valid parameter should give finite value"
    assert not tf.math.is_nan(grad[0]).numpy(), "Valid parameter should give finite gradient"

    # Test with invalid parameter (should return -inf with zero gradient)
    theta_invalid = tf.constant([-0.5], dtype=dtype)
    with tf.GradientTape() as tape:
        tape.watch(theta_invalid)
        value = target_with_custom_grad(theta_invalid)
    grad = tape.gradient(value, theta_invalid)

    assert tf.math.is_inf(value).numpy() and value.numpy() < 0, \
        "Invalid parameter should give -inf"
    assert abs(grad[0].numpy()) < 1e-10, \
        f"Invalid parameter should give zero gradient, got {grad[0].numpy()}"


def test_metropolis_hastings_rejection_of_invalid():
    """Test that MH correctly rejects proposals with -inf log probability."""
    dtype = tf.float64

    # Current state: valid with log_prob = -1.0
    current_log_prob = tf.constant(-1.0, dtype=dtype)

    # Proposed state: invalid with log_prob = -inf
    proposed_log_prob = tf.constant(float("-inf"), dtype=dtype)

    # MH acceptance log probability: log(min(1, exp(proposed - current)))
    log_accept_prob = proposed_log_prob - current_log_prob

    # Should be -inf, giving acceptance probability = 0
    assert tf.math.is_inf(log_accept_prob).numpy() and log_accept_prob.numpy() < 0, \
        "MH log acceptance probability should be -inf"

    # Accept probability
    accept_prob = tf.exp(log_accept_prob)
    assert accept_prob.numpy() == 0.0, \
        f"MH should reject with probability 1, got acceptance {accept_prob.numpy()}"


if __name__ == "__main__":
    test_hmc_graceful_failure_pathological_target()
    test_dual_parameter_target_handles_neg_inf()
    test_metropolis_hastings_rejection_of_invalid()
    print("All HMC graceful failure tests passed!")
