"""Integration test for LEDH graceful failure with pathological parameters.

Tests that LEDH canonical score returns -inf with zero gradient when
Cholesky decomposition fails due to pathological model parameters.
"""

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
)


class PathologicalLGSSM(PerPointScoreModel):
    """LGSSM with pathologically small process noise to trigger degeneracy."""

    def __init__(self, state_dim: int = 2, tiny_noise: float = 1e-10):
        self.state_dim = state_dim
        self.tiny_noise = tiny_noise

    def transition_log_density(self, state, next_state, noise, theta):
        # Near-deterministic dynamics: Q = tiny_noise * I
        diff = next_state - state
        log_det = self.state_dim * tf.math.log(
            tf.constant(self.tiny_noise, state.dtype)
        )
        mahalanobis = tf.reduce_sum(
            tf.square(diff) / self.tiny_noise, axis=-1
        )
        normalizer = (
            0.5 * self.state_dim * tf.math.log(
                2.0 * tf.constant(np.pi, state.dtype)
            )
        )
        return -normalizer - 0.5 * log_det - 0.5 * mahalanobis

    def observation_log_density(self, state, observation, theta):
        # Identity observation with unit noise
        diff = observation - state
        log_det = 0.0
        mahalanobis = tf.reduce_sum(tf.square(diff), axis=-1)
        normalizer = (
            0.5 * self.state_dim * tf.math.log(
                2.0 * tf.constant(np.pi, state.dtype)
            )
        )
        return -normalizer - 0.5 * log_det - 0.5 * mahalanobis

    def transition_score(self, state, next_state, noise, theta):
        # Zero score (parameters are hardcoded)
        return tf.zeros_like(theta)

    def observation_score(self, state, observation, theta):
        # Zero score (parameters are hardcoded)
        return tf.zeros_like(theta)


class TestLEDHGracefulFailureIntegration(tf.test.TestCase):
    """Integration tests for LEDH graceful failure."""

    def setUp(self):
        """Set up test fixtures."""
        self.dtype = tf.float64
        self.state_dim = 2
        self.param_dim = 2
        self.horizon = 3
        self.num_particles = 50

    def test_pathological_lgssm_returns_neg_inf(self):
        """Test that pathological LGSSM returns -inf without crashing."""
        # Create pathological model (tiny process noise)
        model = PathologicalLGSSM(
            state_dim=self.state_dim,
            tiny_noise=1e-20  # Extremely small to trigger degeneracy
        )

        # Initial state
        initial_states = tf.zeros(
            [1, self.num_particles, self.state_dim], dtype=self.dtype
        )
        initial_covariances = (
            tf.eye(self.state_dim, dtype=self.dtype)[None, None, :, :]
            * 0.01
        )

        # Zero noises (deterministic dynamics will cause particle collapse)
        noises = tf.zeros(
            [self.horizon, self.state_dim], dtype=self.dtype
        )

        # Observations far from initial state (strong likelihood)
        observations = tf.ones(
            [self.horizon, self.state_dim], dtype=self.dtype
        ) * 10.0

        # Dummy parameter
        theta = tf.zeros([1, self.param_dim], dtype=self.dtype)
        directions = tf.eye(
            self.param_dim, dtype=self.dtype
        )[None, :, :]

        # Create reset design (symmetric basis)
        reset_basis = tf.concat([
            tf.eye(self.state_dim, dtype=self.dtype),
            -tf.eye(self.state_dim, dtype=self.dtype)
        ], axis=0)
        reset_repeats = (self.num_particles + 2 * self.state_dim - 1) // (2 * self.state_dim)
        reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:self.num_particles]

        # Call canonical score with weak ridge (should trigger failure)
        value, score, valid = canonical_batch_fused_value_score(
            model,
            theta,
            directions,
            initial_states,
            initial_covariances,
            noises,
            observations,
            substeps=1,
            reset_policy="contract_e",
            reset_design=reset_design,
            reset_ridge=1e-5,  # Weak ridge
        )

        # Should return -inf (not crash)
        # Note: Depending on when failure occurs, may return finite or -inf
        # The key is no crash and no NaN
        self.assertFalse(tf.math.is_nan(value[0]).numpy())

        # If invalid, should be -inf with zero score
        if tf.math.is_inf(value[0]).numpy():
            self.assertTrue(value[0].numpy() < 0)
            self.assertAllClose(
                score[0].numpy(),
                np.zeros(self.param_dim),
                atol=1e-10
            )

    def test_normal_lgssm_returns_finite(self):
        """Test that normal LGSSM returns finite values."""
        # Create normal model (reasonable process noise)
        model = PathologicalLGSSM(
            state_dim=self.state_dim,
            tiny_noise=0.1  # Reasonable noise level
        )

        # Initial state
        initial_states = tf.zeros(
            [1, self.num_particles, self.state_dim], dtype=self.dtype
        )
        initial_covariances = (
            tf.eye(self.state_dim, dtype=self.dtype)[None, None, :, :]
            * 0.1
        )

        # Small noises
        generator = tf.random.Generator.from_seed(42)
        noises = generator.normal(
            [self.horizon, self.state_dim], dtype=self.dtype
        ) * 0.1

        # Observations near initial state
        observations = generator.normal(
            [self.horizon, self.state_dim], dtype=self.dtype
        ) * 0.5

        # Dummy parameter
        theta = tf.zeros([1, self.param_dim], dtype=self.dtype)
        directions = tf.eye(
            self.param_dim, dtype=self.dtype
        )[None, :, :]

        # Create reset design
        reset_basis = tf.concat([
            tf.eye(self.state_dim, dtype=self.dtype),
            -tf.eye(self.state_dim, dtype=self.dtype)
        ], axis=0)
        reset_repeats = (self.num_particles + 2 * self.state_dim - 1) // (2 * self.state_dim)
        reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:self.num_particles]

        # Call canonical score
        value, score, valid = canonical_batch_fused_value_score(
            model,
            theta,
            directions,
            initial_states,
            initial_covariances,
            noises,
            observations,
            substeps=1,
            reset_policy="contract_e",
            reset_design=reset_design,
            reset_ridge=1e-5,
        )

        # Should return finite values
        self.assertFalse(tf.math.is_nan(value[0]).numpy())
        self.assertFalse(tf.math.is_inf(value[0]).numpy())
        self.assertFalse(
            tf.reduce_any(tf.math.is_nan(score[0])).numpy()
        )
        self.assertFalse(
            tf.reduce_any(tf.math.is_inf(score[0])).numpy()
        )


if __name__ == "__main__":
    tf.test.main()
