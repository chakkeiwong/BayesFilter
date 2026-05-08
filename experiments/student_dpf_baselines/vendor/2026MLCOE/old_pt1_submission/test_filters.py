"""
test_filters.py
===============
Integration tests for filters.py using ssm_models.py.
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import os
# Force CPU usage to avoid Colocation/GPU errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import unittest
import tensorflow as tf
import numpy as np

# Import generic filters
from filters import (KF, EKF, UKF, ESRF, GSMC, BPF, UPF,
                     EDH, LEDH, PFPF_EDH, PFPF_LEDH, KPFF)

# Import Models
from ssm_models import StochasticVolatilityModel, RangeBearingModel, AcousticTrackingSSM

# Configuration
tf.config.set_visible_devices([], 'GPU')
DTYPE = tf.float64

class TestFiltersIntegration(tf.test.TestCase):
    """
    Integration tests for all filter classes against the provided SSMs.
    """

    @classmethod
    def setUpClass(cls):
        """Set up models once."""
        cls.rb_model = RangeBearingModel()
        cls.sv_model = StochasticVolatilityModel()
        cls.ac_model = AcousticTrackingSSM()

    def setUp(self):
        """
        Reset dummy data before each test.
        RESHAPED: All vector data is now Rank 2 (N, 1) to satisfy KF requirements.
        """
        # Range Bearing Data (State dim 4, Obs dim 2)
        # We create (4, 1) and (2, 1) vectors natively.
        self.rb_x0 = tf.constant([[0.0], [0.0], [1.0], [0.0]], dtype=DTYPE)
        self.rb_P0 = tf.eye(4, dtype=DTYPE) * 0.1
        self.rb_z = tf.constant([[5.0], [0.5]], dtype=DTYPE)

        # Acoustic Data (State dim 16, Obs dim 25) -> Rank 1 (kept as is for PFs)
        self.ac_x0 = tf.random.normal((16,), dtype=DTYPE)
        self.ac_P0 = tf.eye(16, dtype=DTYPE)
        self.ac_z = tf.random.normal((25,), dtype=DTYPE)

        # Stochastic Volatility Data (State dim 1, Obs dim 1)
        self.sv_x0 = tf.constant([0.0], dtype=DTYPE)
        self.sv_P0 = tf.eye(1, dtype=DTYPE)
        self.sv_z = tf.constant([1.0], dtype=DTYPE)

    # ==========================================================================
    # 1. Standard Kalman Filters
    # ==========================================================================
    def test_kf_pipeline(self):
        """
        Test standard Linear Kalman Filter.
        Works directly because setUp data is now Rank 2 (N, 1).
        """
        # H must be (Obs, State) -> (2, 4)
        H_dummy = tf.constant([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0]], dtype=DTYPE)

        kf = KF(F=self.rb_model.F,
                H=H_dummy,
                Q=self.rb_model.Q_filter,
                R=self.rb_model.R_filter,
                P0=self.rb_P0,
                x0=self.rb_x0) # Passed directly as (4, 1)

        x_pred, P_pred = kf.predict()
        x_upd, P_upd = kf.update(self.rb_z) # Passed directly as (2, 1)

        self.assertEqual(x_upd.shape, (4, 1))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(x_upd)))

    def test_ekf_rb_model(self):
        """
        Test EKF on Range-Bearing.
        NOTE: EKF expects Rank 1 inputs for z (due to internal flattening),
        so we squeeze the Rank 2 test data.
        """
        ekf = EKF(self.rb_model)
        # Squeeze x0 to (4,) and P0 stays (4,4)
        ekf.init(tf.squeeze(self.rb_x0), self.rb_P0)

        # Squeeze z to (2,)
        est = ekf.step(tf.squeeze(self.rb_z))

        self.assertEqual(est.shape, (4,))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(est)))

    def test_ekf_sv_model(self):
        """Test EKF on Stochastic Volatility."""
        ekf = EKF(self.sv_model)
        ekf.init(self.sv_x0, self.sv_P0)
        est = ekf.step(self.sv_z)
        self.assertEqual(est.shape, (1,))

    def test_ukf_pipeline(self):
        """Test UKF (Uses squeezed inputs)."""
        ukf = UKF(self.rb_model)
        ukf.init(tf.squeeze(self.rb_x0), self.rb_P0)
        est = ukf.step(tf.squeeze(self.rb_z))
        self.assertEqual(est.shape, (4,))

    # ==========================================================================
    # 2. Particle Filters
    # ==========================================================================
    def test_bpf_pipeline(self):
        """Test Bootstrap Particle Filter (Uses squeezed inputs)."""
        bpf = BPF(self.rb_model, N=50)
        bpf.init(tf.squeeze(self.rb_x0), self.rb_P0)
        est = bpf.step(tf.squeeze(self.rb_z))

        self.assertEqual(est.shape, (4,))
        self.assertAllClose(tf.reduce_sum(bpf.W), 1.0)

    def test_upf_pipeline(self):
        """Test Unscented Particle Filter."""
        upf = UPF(self.rb_model, N=10)
        upf.init(tf.squeeze(self.rb_x0), self.rb_P0)
        est = upf.step(tf.squeeze(self.rb_z))
        self.assertEqual(est.shape, (4,))

    def test_esrf_pipeline(self):
        """Test Ensemble Square Root Filter."""
        esrf = ESRF(self.rb_model, N=50)
        esrf.init(tf.squeeze(self.rb_x0), self.rb_P0)
        est = esrf.step(tf.squeeze(self.rb_z))
        self.assertEqual(est.shape, (4,))

    def test_gsmc_pipeline(self):
        """Test Gaussian Sum Monte Carlo."""
        gsmc = GSMC(self.rb_model, N=50)
        gsmc.init(tf.squeeze(self.rb_x0), self.rb_P0)
        est = gsmc.step(tf.squeeze(self.rb_z))
        self.assertEqual(est.shape, (4,))

    # ==========================================================================
    # 3. Particle Flow Filters (EDH/LEDH)
    # ==========================================================================
    def test_edh_pipeline(self):
        """Test EDH (Uses squeezed inputs)."""
        edh = EDH(self.rb_model, N=20)
        edh.init(tf.squeeze(self.rb_x0), self.rb_P0)
        est = edh.step(tf.squeeze(self.rb_z))
        self.assertEqual(est.shape, (4,))

    def test_ledh_pipeline(self):
        """Test LEDH (Uses squeezed inputs)."""
        ledh = LEDH(self.rb_model, N=20)
        ledh.init(tf.squeeze(self.rb_x0), self.rb_P0)
        est = ledh.step(tf.squeeze(self.rb_z))
        self.assertEqual(est.shape, (4,))

    def test_pfpf_edh_pipeline(self):
        """Test PFPF-EDH."""
        pfpf = PFPF_EDH(self.rb_model, N=20)
        pfpf.init(tf.squeeze(self.rb_x0), self.rb_P0)
        est = pfpf.step(tf.squeeze(self.rb_z))
        self.assertEqual(est.shape, (4,))

    def test_pfpf_ledh_acoustic(self):
        """Test PFPF-LEDH on Acoustic model (explicit CPU placement)."""
        with tf.device('/CPU:0'):
            pfpf = PFPF_LEDH(self.ac_model, N=10)
            pfpf.init(self.ac_x0, self.ac_P0)
            est = pfpf.step(self.ac_z)
            self.assertEqual(est.shape, (16,))

    # ==========================================================================
    # 4. Kernel Particle Flow
    # ==========================================================================
    def test_kpff_pipeline(self):
        """Test KPFF."""
        N = 20
        dim = 4
        ensemble = tf.random.normal((N, dim), dtype=DTYPE)

        y_obs = tf.constant([0.5], dtype=DTYPE)
        obs_idx = [0]
        R_var = 0.1

        kpff = KPFF(ensemble=ensemble,
                    y_obs=y_obs,
                    obs_idx=obs_idx,
                    R_var=R_var)

        kpff.update(n_steps=5, dt=0.01)

        new_mean = tf.reduce_mean(kpff.X, axis=0)
        old_mean = tf.reduce_mean(ensemble, axis=0)

        self.assertNotAllClose(new_mean, old_mean)
        self.assertEqual(kpff.X.shape, (N, dim))

if __name__ == "__main__":
    unittest.main()
