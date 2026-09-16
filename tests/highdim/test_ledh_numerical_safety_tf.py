"""Unit tests for numerical safety utilities.

Tests graceful failure handling for Cholesky decomposition and other
numerically unstable operations that can occur during HMC parameter exploration.
"""

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky


class TestSafeCholesky(tf.test.TestCase):
    """Test safe Cholesky decomposition with NaN detection."""

    def test_well_conditioned_matrix(self):
        """Safe Cholesky should succeed on well-conditioned PD matrices."""
        # Create well-conditioned matrix: κ ≈ 10
        matrix = tf.constant(
            [[4.0, 1.0], [1.0, 3.0]], dtype=tf.float64
        )

        valid, chol = safe_cholesky(matrix)

        self.assertTrue(valid.numpy())
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())

        # Verify L @ L^T = matrix
        reconstructed = tf.matmul(chol, chol, transpose_b=True)
        self.assertAllClose(matrix, reconstructed, rtol=1e-10)

    def test_identity_matrix(self):
        """Safe Cholesky should handle identity matrix."""
        matrix = tf.eye(5, dtype=tf.float64)

        valid, chol = safe_cholesky(matrix)

        self.assertTrue(valid.numpy())
        self.assertAllClose(chol, matrix, rtol=1e-10)

    def test_ill_conditioned_but_valid(self):
        """Safe Cholesky should succeed on ill-conditioned but PD matrices."""
        # κ ≈ 10^8 but still positive definite
        matrix = tf.constant(
            [[1e8, 0.0], [0.0, 1.0]], dtype=tf.float64
        )

        valid, chol = safe_cholesky(matrix)

        self.assertTrue(valid.numpy())
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())

    def test_extremely_ill_conditioned(self):
        """Safe Cholesky should fail gracefully on extremely ill-conditioned matrices."""
        # κ > 10^15 - numerically singular in float64
        matrix = tf.constant(
            [[1e16, 1e16 - 1.0], [1e16 - 1.0, 1e16]], dtype=tf.float64
        )

        valid, chol = safe_cholesky(matrix)

        # Should detect failure
        self.assertFalse(valid.numpy())
        # Should return zeros, not NaN
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())
        self.assertAllClose(chol, tf.zeros_like(chol))

    def test_indefinite_matrix(self):
        """Safe Cholesky should fail gracefully on indefinite matrices."""
        # Negative eigenvalues
        matrix = tf.constant(
            [[1.0, 2.0], [2.0, 1.0]], dtype=tf.float64
        )

        valid, chol = safe_cholesky(matrix)

        # Should detect failure
        self.assertFalse(valid.numpy())
        # Should return zeros, not NaN
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())
        self.assertAllClose(chol, tf.zeros_like(chol))

    def test_singular_matrix(self):
        """Safe Cholesky should fail gracefully on singular matrices."""
        # Zero eigenvalue
        matrix = tf.constant(
            [[1.0, 1.0], [1.0, 1.0]], dtype=tf.float64
        )

        valid, chol = safe_cholesky(matrix)

        # Should detect failure
        self.assertFalse(valid.numpy())
        # Should return zeros, not NaN
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())
        self.assertAllClose(chol, tf.zeros_like(chol))

    def test_negative_definite_matrix(self):
        """Safe Cholesky should fail gracefully on negative definite matrices."""
        # All negative eigenvalues
        matrix = tf.constant(
            [[-4.0, -1.0], [-1.0, -3.0]], dtype=tf.float64
        )

        valid, chol = safe_cholesky(matrix)

        # Should detect failure
        self.assertFalse(valid.numpy())
        # Should return zeros, not NaN
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())
        self.assertAllClose(chol, tf.zeros_like(chol))

    def test_zero_matrix(self):
        """Safe Cholesky should fail gracefully on zero matrix."""
        matrix = tf.zeros([3, 3], dtype=tf.float64)

        valid, chol = safe_cholesky(matrix)

        # Should detect failure
        self.assertFalse(valid.numpy())
        # Should return zeros, not NaN
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())
        self.assertAllClose(chol, tf.zeros_like(chol))

    def test_batched_matrices(self):
        """Safe Cholesky should handle batched matrices correctly."""
        # Batch of 3 matrices: [valid, invalid, valid]
        matrices = tf.constant([
            [[4.0, 1.0], [1.0, 3.0]],  # Well-conditioned
            [[1.0, 2.0], [2.0, 1.0]],  # Indefinite
            [[9.0, 3.0], [3.0, 2.0]],  # Well-conditioned
        ], dtype=tf.float64)

        valid, chol = safe_cholesky(matrices)

        # First matrix: valid
        self.assertTrue(valid[0].numpy())
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol[0])).numpy())

        # Second matrix: invalid
        self.assertFalse(valid[1].numpy())
        self.assertAllClose(chol[1], tf.zeros_like(chol[1]))

        # Third matrix: valid
        self.assertTrue(valid[2].numpy())
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol[2])).numpy())

    def test_float32_dtype(self):
        """Safe Cholesky should work with float32."""
        matrix = tf.constant(
            [[4.0, 1.0], [1.0, 3.0]], dtype=tf.float32
        )

        valid, chol = safe_cholesky(matrix)

        self.assertTrue(valid.numpy())
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())
        self.assertEqual(chol.dtype, tf.float32)

    def test_phase4a_pathological_case(self):
        """Test case resembling Phase 4a failure: gap covariance with small ridge."""
        # Simulate target_cov - plus_cov with particle degeneracy
        # This can produce negative eigenvalues even with small ridge
        np.random.seed(42)
        d = 10

        # Degenerate particle covariance (one particle has weight ≈ 1)
        weights = np.array([0.987] + [0.013 / (d - 1)] * (d - 1))
        particles = np.random.randn(d, d) * 0.1
        particles[0] = np.zeros(d)  # Dominant particle at origin

        # Target covariance (poorly estimated from degenerate weights)
        target_cov = np.cov(particles.T, aweights=weights)

        # Plus covariance (after Sinkhorn, different from target)
        plus_cov = np.cov(particles.T) * 0.9

        # Gap with insufficient ridge
        ridge = 1e-5
        gap = target_cov - plus_cov + ridge * np.eye(d)

        matrix = tf.constant(gap, dtype=tf.float64)
        valid, chol = safe_cholesky(matrix)

        # May or may not be valid depending on exact eigenvalues
        # But must NOT produce NaN in output
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())

        if not valid.numpy():
            # If invalid, should return zeros
            self.assertAllClose(chol, tf.zeros_like(chol))

    def test_xla_compilation(self):
        """Safe Cholesky should compile with XLA."""
        @tf.function(jit_compile=True)
        def compiled_safe_cholesky(matrix):
            return safe_cholesky(matrix)

        matrix = tf.constant(
            [[4.0, 1.0], [1.0, 3.0]], dtype=tf.float64
        )

        # Should not raise compilation error
        valid, chol = compiled_safe_cholesky(matrix)

        self.assertTrue(valid.numpy())
        self.assertFalse(tf.reduce_any(tf.math.is_nan(chol)).numpy())


if __name__ == "__main__":
    tf.test.main()
