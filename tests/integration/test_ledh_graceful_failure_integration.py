"""Integration test for LEDH graceful failure with pathological parameters.

Tests that LEDH canonical score returns -inf with zero gradient when
Cholesky decomposition fails due to pathological model parameters.
"""

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    canonical_batch_fused_value_score, PerPointScoreModel,
)


class PathologicalLGSSM(PerPointScoreModel):
    """LGSSM with pathologically small process noise to trigger degeneracy.

    Implements PerPointScoreModel interface for canonical_batch_fused_value_score.
    """

    def __init__(self, state_dim: int = 2, tiny_noise: float = 1e-10, dtype=tf.float64):
        self.state_dim = state_dim
        self.tiny_noise = tiny_noise
        self.dtype = dtype

        # PerPointScoreModel required attributes
        process_covariance = tf.constant(
            tiny_noise * np.eye(state_dim), dtype=dtype
        )
        observation_covariance = tf.constant(
            1.0 * np.eye(state_dim), dtype=dtype
        )

        super().__init__(
            self._transition_mean_fn, self._transition_mean_tangent_fn,
            self._observation_fn, self._observation_jacobian_fn,
            self._observation_tangent_fn, process_covariance,
            observation_covariance)

    def _transition_mean_fn(self, theta_rows, points):
        """Identity dynamics: f(x) = x (no parameter dependence)."""
        return points

    def _transition_mean_tangent_fn(self, theta_rows, points, d_points, d_theta_rows):
        """Tangent of identity dynamics: df/dx @ d_points."""
        return d_points

    def _observation_fn(self, points):
        """Identity observation: h(x) = x."""
        return points

    def _observation_jacobian_fn(self, points):
        """Jacobian of identity observation: H = I."""
        n = tf.shape(points)[0]
        return tf.broadcast_to(
            tf.eye(self.state_dim, dtype=self.dtype)[None, :, :],
            [n, self.state_dim, self.state_dim]
        )

    def _observation_tangent_fn(self, points, d_points):
        """Tangent of identity observation: H @ d_points."""
        return d_points


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
        # Invalid covariance gives an unconditional failure expectation.
        model = PathologicalLGSSM(
            state_dim=self.state_dim,
            tiny_noise=-0.1  # Indefinite process covariance must fail
        )

        # Initial state
        initial_states = tf.zeros(
            [self.num_particles, self.state_dim], dtype=self.dtype
        )
        initial_covariances = (
            tf.eye(self.state_dim, batch_shape=[self.num_particles], dtype=self.dtype)
            * 0.01
        )

        # Zero noises (deterministic dynamics will cause particle collapse)
        noises = tf.zeros(
            [self.horizon, self.num_particles, self.state_dim], dtype=self.dtype
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

        self.assertEqual(float(value[0]), float('-inf'))
        self.assertAllEqual(score[0], tf.zeros([self.param_dim], self.dtype))
        self.assertFalse(bool(valid['value_valid'][0]))
        self.assertFalse(bool(valid['force_valid'][0]))
        self.assertEqual(int(valid['first_failure_time'][0]), -1)
        self.assertEqual(int(valid['first_failure_stage'][0]), 0)

    def test_normal_lgssm_returns_finite(self):
        """Test that normal LGSSM returns finite values."""
        # Create normal model (reasonable process noise)
        model = PathologicalLGSSM(
            state_dim=self.state_dim,
            tiny_noise=0.1  # Reasonable noise level
        )

        # Initial state
        initial_states = tf.zeros(
            [self.num_particles, self.state_dim], dtype=self.dtype
        )
        initial_covariances = (
            tf.eye(self.state_dim, batch_shape=[self.num_particles], dtype=self.dtype)
            * 0.1
        )

        # Small noises
        generator = tf.random.Generator.from_seed(42)
        noises = generator.normal(
            [self.horizon, self.num_particles, self.state_dim], dtype=self.dtype
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
