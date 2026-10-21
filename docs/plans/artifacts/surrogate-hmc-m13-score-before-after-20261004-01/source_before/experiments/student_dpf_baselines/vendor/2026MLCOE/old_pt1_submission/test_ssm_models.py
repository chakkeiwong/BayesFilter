"""
test_ssm_models.py
==================
Unit tests for State Space Models (SSM) implemented in ssm_models.py.

Tests cover:
1. Data generation shapes and types.
2. Consistency of measurement functions.
3. Correctness of Jacobian implementations (comparing analytical vs autodiff).

Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import unittest
import numpy as np
import tensorflow as tf

# Import your modules
from ssm_models import StochasticVolatilityModel, RangeBearingModel, AcousticTrackingSSM

DTYPE = tf.float64

class TestSSMModels(tf.test.TestCase):
    """
    Test suite for State Space Models.
    Inherits from tf.test.TestCase for TensorFlow assertions.
    """

    def setUp(self):
        """Initialize models before each test."""
        self.sv_model = StochasticVolatilityModel()
        self.rb_model = RangeBearingModel()
        self.acoustic_model = AcousticTrackingSSM()

    # ==========================================================================
    # Stochastic Volatility Model Tests
    # ==========================================================================
    def test_sv_generation_shape(self):
        """Test if SV model generates data with correct shapes."""
        T = 50
        x, y = self.sv_model.generate(T=T)

        self.assertEqual(x.shape, (T, 1), "State X shape mismatch for SV Model.")
        self.assertEqual(y.shape, (T, 1), "Observation Y shape mismatch for SV Model.")
        self.assertEqual(x.dtype, DTYPE, "State X dtype mismatch.")

    def test_sv_jacobian(self):
        """
        Verify the analytical Jacobian of h(x) matches TF auto-diff.
        h(x) = log(beta^2) + x - offset
        """
        x_val = tf.constant([[0.5]], dtype=DTYPE)

        # 1. Analytical Jacobian
        jac_analytical = self.sv_model.h_jacobian(x_val)

        # 2. Auto-diff Jacobian
        with tf.GradientTape() as tape:
            tape.watch(x_val)
            y_val = self.sv_model.h_func(x_val)
        jac_autodiff = tape.jacobian(y_val, x_val)

        # Squeeze to compare simple scalars/matrices
        self.assertAllClose(jac_analytical, tf.reshape(jac_autodiff, shape=jac_analytical.shape),
                            msg="SV Model H Jacobian mismatch.")

    # ==========================================================================
    # Range Bearing Model Tests
    # ==========================================================================
    def test_rb_generation_shape(self):
        """Test if Range Bearing model generates correct shapes."""
        T = 50
        x, y = self.rb_model.generate(T=T)

        # State dim is 4 (x, y, vx, vy), Obs dim is 2 (r, theta)
        self.assertEqual(x.shape, (T, 4), "State X shape mismatch for RB Model.")
        self.assertEqual(y.shape, (T, 2), "Observation Y shape mismatch for RB Model.")

    def test_rb_measurement_consistency(self):
        """Ensure h_func returns correct range and bearing logic."""
        # Define a state at (3, 4) with arbitrary velocity
        x_input = tf.constant([[3.0, 4.0, 1.0, 1.0]], dtype=DTYPE)
        y_out = self.rb_model.h_func(x_input)

        expected_r = 5.0  # sqrt(3^2 + 4^2)
        expected_theta = np.arctan2(4.0, 3.0)

        self.assertAllClose(y_out[0, 0], expected_r, msg="Range calculation incorrect.")
        self.assertAllClose(y_out[0, 1], expected_theta, msg="Theta calculation incorrect.")

    # ==========================================================================
    # Acoustic Tracking SSM Tests
    # ==========================================================================
    def test_acoustic_jacobian_correctness(self):
        """
        CRITICAL: Test if manual Jacobian in AcousticTrackingSSM matches AutoDiff.
        The manual implementation involves complex broadcasting and slicing.
        """
        # Create a batch of 1 particle
        x_val = tf.random.normal((1, 16), dtype=DTYPE)

        # 1. Manual Implementation
        jac_manual = self.acoustic_model.jacobian_h(x_val)

        # 2. AutoDiff Implementation
        with tf.GradientTape() as tape:
            tape.watch(x_val)
            y_val = self.acoustic_model.h_func(x_val)

        # Jacobian output shape will be [Batch, Obs_Dim, Batch, State_Dim]
        # We need to extract the diagonal block for the batch
        jac_autodiff = tape.jacobian(y_val, x_val)
        jac_autodiff = jac_autodiff[0, :, 0, :] # Extract [Obs, State] for the single batch item

        # Compare
        # jac_manual might come out stacked; ensure shape matches
        self.assertEqual(jac_manual.shape, jac_autodiff.shape,
                         "Acoustic Jacobian shapes differ.")

        self.assertAllClose(jac_manual, jac_autodiff, rtol=1e-4, atol=1e-4,
                           msg="Acoustic Model manual Jacobian does not match TF AutoDiff.")

if __name__ == "__main__":
    unittest.main()
