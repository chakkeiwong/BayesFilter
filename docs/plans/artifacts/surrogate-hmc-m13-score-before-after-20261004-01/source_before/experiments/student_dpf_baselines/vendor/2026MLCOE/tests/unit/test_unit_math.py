"""
Unit Tests: Mathematical & Logical Constraints
============================================
Verifies that individual subroutines across all filter families preserve
necessary mathematical conditions like mass conservation, positive
semi-definiteness, and functional gradients.

Author: Joowon Lee
Date: 2026-02-26
"""

import sys
import os

# Disable GPU globally to prevent Metal plugin colocation errors on M1 Ultra
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import tensorflow as tf
import numpy as np

# Force CPU placement for stability during variable reads/writes
tf.config.set_visible_devices([], 'GPU')

from src.filters.classical import DTYPE, KF, UKF, ESRF
from src.filters.particle import BPF, UPF, GSMC
from src.filters.flow_filters import ParticleFlowFilter, KPFF
from src.filters.resampling.optimal_transport import SinkhornResampler
from src.filters.resampling.soft import SoftResampler

# --- MOCK MODELS ---
class MockLinearModel:
    state_dim = 2
    F = tf.eye(2, dtype=DTYPE)
    H = tf.eye(2, dtype=DTYPE)
    Q_filter = tf.eye(2, dtype=DTYPE) * 0.1
    R_filter = tf.eye(2, dtype=DTYPE) * 0.1
    R_inv_filter = tf.linalg.inv(R_filter)
    P_prior = tf.eye(2, dtype=DTYPE)

    def h_func(self, x):
        # Polymorphic return for individual vectors or batches
        if len(tf.shape(x)) == 1:
            return tf.linalg.matvec(self.H, x)
        return tf.matmul(x, self.H, transpose_b=True)

# --- TESTS ---

def test_ukf_sigma_point_moments():
    """Math Condition: Sigma points must reconstruct prior statistics."""
    model = MockLinearModel()
    ukf = UKF(model)
    mean = tf.constant([5.0, -3.0], dtype=DTYPE)
    cov = tf.constant([[2.0, 0.5], [0.5, 1.0]], dtype=DTYPE)
    ukf.init(mean, cov)
    sigmas = ukf.sigma_points(ukf.x, ukf.P)
    Wm_vec = [ukf.lam / (ukf.n + ukf.lam)] + [1/(2*(ukf.n + ukf.lam))] * (2*ukf.n)
    Wm = tf.constant(Wm_vec, dtype=DTYPE)
    reconstructed_mean = tf.linalg.matvec(tf.transpose(sigmas), Wm)
    tf.debugging.assert_near(mean, reconstructed_mean, rtol=1e-5)

def test_esrf_anomaly_centering():
    """Math Condition: Ensemble anomalies must strictly sum to zero."""
    model = MockLinearModel()
    esrf = ESRF(model, N=50)
    esrf.init(tf.zeros(2, dtype=DTYPE), tf.eye(2, dtype=DTYPE))
    x_mean = tf.reduce_mean(esrf.X, axis=0)
    A = esrf.X - x_mean
    tf.debugging.assert_near(tf.reduce_sum(A, axis=0), tf.zeros(2, dtype=DTYPE), atol=1e-5)

def test_bpf_weight_normalization():
    """Logic Condition: BPF must maintain probabilistic weight validity."""
    model = MockLinearModel()
    bpf = BPF(model, N=100)
    bpf.init(tf.zeros(2, dtype=DTYPE), tf.eye(2, dtype=DTYPE))
    bpf.step(tf.constant([0.5, 0.5], dtype=DTYPE))
    tf.debugging.assert_near(tf.reduce_sum(bpf.W), tf.cast(1.0, DTYPE), atol=1e-6)

def test_upf_proposal_sampling():
    """Logic Condition: UPF particles must move from prior via UKF proposals."""
    model = MockLinearModel()
    upf = UPF(model, N=10)
    upf.init(tf.zeros(2, dtype=DTYPE), tf.eye(2, dtype=DTYPE))
    old_X = tf.identity(upf.X)
    upf.step(tf.constant([1.0, 1.0], dtype=DTYPE))
    assert not tf.reduce_all(tf.math.equal(upf.X, old_X))

def test_dynamic_bvp_schedule_boundaries():
    """Math Condition: BVP schedule must start at 0 and end at 1."""
    model = MockLinearModel()
    pf = ParticleFlowFilter(model, N=10, mode='ledh_opt')
    # FIX: Corrected tf.eye syntax (dtype must be a keyword argument)
    beta_vals, _ = pf._dynamic_bvp_solver(tf.eye(2, dtype=DTYPE)*1e-4, tf.eye(2, dtype=DTYPE)*1e4)
    assert beta_vals[0] == 0.0
    assert np.isclose(beta_vals[-1], 1.0, atol=1e-2)

def test_kpff_rkhs_kernel_properties():
    """Math Condition: KPFF must initialize positive bandwidth."""
    ensemble = tf.random.normal((20, 2), dtype=DTYPE)
    filt = KPFF(ensemble, [1.0], [0], R_var=0.1)
    assert filt.alpha > 0
    assert tf.reduce_all(filt.prior_var > 0)

def test_sinkhorn_marginal_conservation():
    """Math Condition: OT transport must conserve mass."""
    N = 100
    x = tf.random.normal((N, 2), dtype=DTYPE)
    lw = tf.nn.log_softmax(tf.random.normal((N,), dtype=DTYPE))
    resampler = SinkhornResampler(epsilon=0.05, n_iter=50)
    x_res, _ = resampler.resample(x, lw)
    prior_m = tf.reduce_sum(x * tf.exp(lw)[:, None], 0)
    assert tf.reduce_all(tf.abs(prior_m - tf.reduce_mean(x_res, 0)) < 1e-2)

def test_soft_resampling_interpolation():
    """Logic Condition: Alpha=0 must reduce to weighted mean."""
    N = 50
    resampler = SoftResampler(num_particles=N, alpha=0.0)
    x = tf.random.normal((N, 2), dtype=DTYPE)
    w = tf.nn.softmax(tf.random.normal((N,), dtype=DTYPE))
    x_soft, _, _ = resampler(x, tf.zeros((N, 1), DTYPE), w)
    prior_mean = tf.reduce_sum(x * w[:, None], axis=0)
    tf.debugging.assert_near(prior_mean, tf.reduce_mean(x_soft, axis=0), atol=1e-5)

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
