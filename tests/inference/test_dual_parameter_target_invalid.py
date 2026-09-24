"""Unit tests for dual-parameter target -inf handling.

Tests that the dual-parameter target custom gradient correctly handles
-inf from canonical score computation by returning -inf with zero gradient.
"""

import numpy as np
import tensorflow as tf

from bayesfilter.inference.ledh_dual_parameter_target import (
    DualParameterLEDHTarget,
)
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import PerPointScoreModel


class MockInvalidScoreModel(PerPointScoreModel):
    """Mock model that returns -inf for specific parameter values."""

    def __init__(self, invalid_threshold: float = 10.0):
        self.invalid_threshold = invalid_threshold

    def transition_log_density(self, state, next_state, noise, theta):
        return tf.constant(0.0, dtype=state.dtype)

    def observation_log_density(self, state, observation, theta):
        return tf.constant(0.0, dtype=state.dtype)

    def transition_score(self, state, next_state, noise, theta):
        return tf.zeros_like(theta)

    def observation_score(self, state, observation, theta):
        return tf.zeros_like(theta)


def mock_canonical_batch_fused_value_score_invalid(
    model, theta, directions, initial_states, initial_covariances,
    noises, observations, **params
):
    """Mock that returns -inf for theta with any component > threshold."""
    dtype = theta.dtype
    batch_size = tf.shape(theta)[0]
    param_dim = tf.shape(theta)[1]
    num_directions = tf.shape(directions)[1]

    # Check if any theta component exceeds threshold
    is_invalid = tf.reduce_any(theta > 10.0, axis=1)

    # Return -inf for invalid, finite value for valid
    valid_value = tf.constant(-5.0, dtype)
    invalid_value = tf.constant(float("-inf"), dtype)
    values = tf.where(is_invalid, invalid_value, valid_value)

    # Zero scores
    scores = tf.zeros([batch_size, param_dim], dtype=dtype)

    # Scalar validity flag (always True in this mock - failure is via -inf)
    valid = tf.constant(True)

    return values, scores, valid


class TestDualParameterTargetInvalid(tf.test.TestCase):
    """Test dual-parameter target -inf handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.dtype = tf.float64
        self.param_dim = 3
        self.horizon = 2
        self.state_dim = 2

        # Create mock data
        self.initial_states = tf.zeros([1, self.state_dim], dtype=self.dtype)
        self.initial_covariances = tf.eye(
            self.state_dim, dtype=self.dtype
        )[None, :, :]
        self.noises = tf.zeros([self.horizon, self.state_dim], dtype=self.dtype)
        self.observations = tf.zeros(
            [self.horizon, self.state_dim], dtype=self.dtype
        )

        # Mock model
        self.model = MockInvalidScoreModel()

    def test_invalid_parameter_returns_neg_inf(self):
        """Test that invalid parameters return -inf."""
        # Patch the canonical_batch_fused_value_score function
        import bayesfilter.inference.ledh_dual_parameter_target as target_module
        original_fn = target_module.canonical_batch_fused_value_score
        target_module.canonical_batch_fused_value_score = (
            mock_canonical_batch_fused_value_score_invalid
        )

        try:
            target = DualParameterLEDHTarget(
                model=self.model,
                initial_states=self.initial_states,
                initial_covariances=self.initial_covariances,
                noises=self.noises,
                observations=self.observations,
                exact_params={"substeps": 1},
                biased_params={"substeps": 1},
            )

            # Valid parameter (all components < 10)
            theta_valid = tf.constant([1.0, 2.0, 3.0], dtype=self.dtype)
            value_valid = target(theta_valid)

            # Should be finite
            self.assertFalse(tf.math.is_inf(value_valid).numpy())
            self.assertAlmostEqual(value_valid.numpy(), -5.0, places=6)

            # Invalid parameter (one component > 10)
            theta_invalid = tf.constant([1.0, 15.0, 3.0], dtype=self.dtype)
            value_invalid = target(theta_invalid)

            # Should be -inf
            self.assertTrue(tf.math.is_inf(value_invalid).numpy())
            self.assertTrue(value_invalid.numpy() < 0)

        finally:
            # Restore original function
            target_module.canonical_batch_fused_value_score = original_fn

    def test_invalid_parameter_returns_zero_gradient(self):
        """Test that invalid parameters return zero gradient."""
        # Patch the canonical_batch_fused_value_score function
        import bayesfilter.inference.ledh_dual_parameter_target as target_module
        original_fn = target_module.canonical_batch_fused_value_score
        target_module.canonical_batch_fused_value_score = (
            mock_canonical_batch_fused_value_score_invalid
        )

        try:
            target = DualParameterLEDHTarget(
                model=self.model,
                initial_states=self.initial_states,
                initial_covariances=self.initial_covariances,
                noises=self.noises,
                observations=self.observations,
                exact_params={"substeps": 1},
                biased_params={"substeps": 1},
            )

            # Invalid parameter
            theta_invalid = tf.constant([1.0, 15.0, 3.0], dtype=self.dtype)

            with tf.GradientTape() as tape:
                tape.watch(theta_invalid)
                value = target(theta_invalid)

            gradient = tape.gradient(value, theta_invalid)

            # Value should be -inf
            self.assertTrue(tf.math.is_inf(value).numpy())

            # Gradient should be zero (not NaN, not inf)
            self.assertFalse(tf.reduce_any(tf.math.is_nan(gradient)).numpy())
            self.assertFalse(tf.reduce_any(tf.math.is_inf(gradient)).numpy())
            self.assertAllClose(
                gradient.numpy(),
                np.zeros(self.param_dim),
                atol=1e-10
            )

        finally:
            # Restore original function
            target_module.canonical_batch_fused_value_score = original_fn

    def test_valid_parameter_has_nonzero_gradient(self):
        """Test that valid parameters can have non-zero gradients."""
        # Patch to return non-zero biased scores
        def mock_with_scores(
            model, theta, directions, initial_states, initial_covariances,
            noises, observations, **params
        ):
            dtype = theta.dtype
            batch_size = tf.shape(theta)[0]
            param_dim = tf.shape(theta)[1]

            is_invalid = tf.reduce_any(theta > 10.0, axis=1)
            valid_value = tf.constant(-5.0, dtype)
            invalid_value = tf.constant(float("-inf"), dtype)
            values = tf.where(is_invalid, invalid_value, valid_value)

            # Non-zero scores for valid parameters (simulating biased force)
            scores = tf.ones([batch_size, param_dim], dtype=dtype) * 0.5
            scores = tf.where(
                is_invalid[:, None],
                tf.zeros([batch_size, param_dim], dtype=dtype),
                scores
            )

            valid = tf.constant(True)
            return values, scores, valid

        import bayesfilter.inference.ledh_dual_parameter_target as target_module
        original_fn = target_module.canonical_batch_fused_value_score
        target_module.canonical_batch_fused_value_score = mock_with_scores

        try:
            target = DualParameterLEDHTarget(
                model=self.model,
                initial_states=self.initial_states,
                initial_covariances=self.initial_covariances,
                noises=self.noises,
                observations=self.observations,
                exact_params={"substeps": 1},
                biased_params={"substeps": 1},
            )

            # Valid parameter
            theta_valid = tf.constant([1.0, 2.0, 3.0], dtype=self.dtype)

            with tf.GradientTape() as tape:
                tape.watch(theta_valid)
                value = target(theta_valid)

            gradient = tape.gradient(value, theta_valid)

            # Value should be finite
            self.assertFalse(tf.math.is_inf(value).numpy())

            # Gradient should be non-zero (from biased scores)
            self.assertFalse(tf.reduce_any(tf.math.is_nan(gradient)).numpy())
            self.assertGreater(tf.reduce_max(tf.abs(gradient)).numpy(), 0.0)

        finally:
            # Restore original function
            target_module.canonical_batch_fused_value_score = original_fn


if __name__ == "__main__":
    tf.test.main()
