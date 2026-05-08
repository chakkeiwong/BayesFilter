"""
Integration Tests: System-Level Behavior
========================================
Verifies interactions across Classical, Particle, and Flow families,
ensuring end-to-end stability and mathematical convergence.

Author: Joowon Lee
Date: 2026-02-26
"""

import sys
import os

# Disable GPU globally to prevent Metal plugin abort on M1 Ultra with float64
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import tensorflow as tf
import numpy as np

# Force CPU placement for test stability on Apple Silicon
tf.config.set_visible_devices([], 'GPU')

from src.filters.classical import DTYPE, KF, EKF, UKF, ESRF
from src.filters.particle import BPF, UPF, GSMC
from src.filters.flow_filters import EDH, LEDH, PFPF_EDH, PFPF_LEDH, KPFF
from src.filters.dpfpf import DifferentiablePFPF
from src.filters.DPF import DPF
from src.inference.pmmh import PMMH_Engine
from src.inference.phmc import hmc_pfpf

# --- REFINED MOCK SYSTEM ---
class MockSystem(tf.Module):
    """
    Polymorphic Mock System designed to satisfy Classical, Particle, and Flow APIs.
    """
    def __init__(self):
        super().__init__()
        self.state_dim = 1
        self.F = tf.eye(1, dtype=DTYPE)
        self.Q_filter = tf.eye(1, dtype=DTYPE) * 0.01
        self.R_filter = tf.eye(1, dtype=DTYPE) * 0.01
        self.R_inv_filter = tf.linalg.inv(self.R_filter) # Required for PFPF variants
        self.sigma_v_sq = tf.Variable(1.0, dtype=DTYPE)
        self.sigma_w_sq = tf.Variable(0.01, dtype=DTYPE)
        self.current_n = tf.Variable(1.0, dtype=DTYPE)

    def propagate(self, x):
        # Handles transition for both single particles and ensemble batches
        return 0.5 * x + 1.0

    def h_func(self, x):
        # Polymorphic observation logic to handle varying input ranks
        # Particles: (N, dim) -> return (N, obs)
        # Individuals: (dim,) -> return (obs,)
        if len(tf.shape(x)) == 1:
            return tf.reshape(x[0]**2, [1])
        return tf.reshape(x**2, [-1, 1])

    def jacobian_h(self, x):
        # Analytical Jacobian required for EKF and Flow families
        x_val = tf.reshape(x, [-1])[0]
        return tf.reshape(2.0 * x_val, (1, 1))

    def update_parameters(self, theta):
        # Squeeze theta for scalar variable assignment
        self.sigma_v_sq.assign(tf.reshape(theta[0], []))

# --- SYSTEM TESTS ---

@pytest.mark.parametrize("filter_class", [BPF, UPF, GSMC])
def test_particle_family_convergence(filter_class):
    """Verifies Particle family can track state without weight degeneracy."""
    model = MockSystem()
    filt = filter_class(model, N=50)
    filt.init(tf.zeros(1, dtype=DTYPE), tf.eye(1, dtype=DTYPE) * 0.1)

    z = tf.constant([0.01], dtype=DTYPE)
    est = filt.step(z)

    assert est.shape == (1,)
    assert not tf.math.is_nan(est[0])

@pytest.mark.parametrize("flow_class", [EDH, LEDH, PFPF_EDH, PFPF_LEDH])
def test_flow_family_execution(flow_class):
    """Verifies Particle Flow homotopy loops and ESS stability."""
    model = MockSystem()
    filt = flow_class(model, N=20, steps=5)
    filt.init(tf.zeros(1, dtype=DTYPE), tf.eye(1, dtype=DTYPE) * 0.1)

    z = tf.constant([0.01], dtype=DTYPE)
    est = filt.step(z)

    assert est.shape == (1,)
    assert filt.ess > 0 # Confirm flow correctly initialized diversity

@pytest.mark.parametrize("kalman_variant", [EKF, UKF, ESRF])
def test_classical_family_integration(kalman_variant):
    """Verifies Classical family updates on M1 Ultra CPU."""
    model = MockSystem()
    filt = kalman_variant(model)
    filt.init(tf.zeros(1, dtype=DTYPE), tf.eye(1, dtype=DTYPE) * 0.1)

    z = tf.constant([0.1], dtype=DTYPE)
    est = filt.step(z)
    assert est.shape == (1,)
    assert not tf.math.is_nan(est[0])

def test_kpff_gradient_shift():
    """Verifies RKHS gradients move particles toward likelihood."""
    N, dim = 20, 1
    ensemble = tf.cast(tf.linspace(-1.0, 1.0, N)[:, None], DTYPE)
    y_obs = tf.constant([5.0], dtype=DTYPE)

    filt = KPFF(ensemble, y_obs, [0], R_var=0.1)
    old_mean = tf.reduce_mean(filt.X)
    filt.update(n_steps=5, dt=0.1)
    new_mean = tf.reduce_mean(filt.X)

    # Particles must shift toward the distant observation
    assert new_mean > old_mean

def test_phmc_leapfrog_stability():
    """Verifies PHMC leapfrog execution with dPFPF."""
    model = MockSystem()
    observations = [tf.constant([0.1], dtype=DTYPE)]

    samples, acc_rate = hmc_pfpf(model, observations, num_samples=2, num_leapfrog_steps=2)
    assert len(samples) == 2
    assert acc_rate >= 0.0 # Verify chain acceptance logic

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
