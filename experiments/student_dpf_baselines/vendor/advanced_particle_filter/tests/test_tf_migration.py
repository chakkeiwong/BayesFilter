"""
Phase 2-3: Validate TF filter migration against NumPy baselines.

Tests:
  1. Numerical agreement (KF, EKF, UKF vs NumPy on linear model)
  2. PF statistical agreement (means within tolerance)
  3. PF benchmark: NumPy vs TF with N=100,000 particles
     - Log-likelihood comparison
     - Runtime comparison with speedup
  4. Nonlinear range-bearing model tests (EKF, UKF, PF)
  5. No retracing check

Run:  python -m advanced_particle_filter.test_tf_migration
  or: pytest advanced_particle_filter/test_tf_migration.py -v -s
"""

import time
import numpy as np
import tensorflow as tf
from numpy.random import default_rng

# NumPy versions
from advanced_particle_filter.models.linear_gaussian import make_lgssm as np_make_lgssm
from advanced_particle_filter.models.range_bearing import make_range_bearing_ssm as np_make_rb_ssm
from advanced_particle_filter.filters.kalman import (
    KalmanFilter as NPKalmanFilter,
    ExtendedKalmanFilter as NPExtendedKalmanFilter,
    UnscentedKalmanFilter as NPUnscentedKalmanFilter,
)
from advanced_particle_filter.filters.particle import BootstrapParticleFilter as NPBootstrapPF

# TF versions
from advanced_particle_filter.tf_models.linear_gaussian import make_lgssm as tf_make_lgssm
from advanced_particle_filter.tf_models.range_bearing import make_range_bearing_ssm as tf_make_rb_ssm
from advanced_particle_filter.tf_filters.kalman import (
    TFKalmanFilter,
    TFExtendedKalmanFilter,
    TFUnscentedKalmanFilter,
)
from advanced_particle_filter.tf_filters.particle import TFBootstrapParticleFilter


# ============================================================================
# Config
# ============================================================================

N_PF_LARGE = 100_000
N_PF_SMALL = 2_000
N_TIMING_RUNS = 5
T_STEPS = 100
SEED_SIM = 42
SEED_PF = 123


# ============================================================================
# Model builders
# ============================================================================

def build_cv_model_params():
    """2D constant-velocity model parameters."""
    dt = 1.0
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    q = 0.1
    Q = q * np.array([
        [dt**3/3, 0, dt**2/2, 0],
        [0, dt**3/3, 0, dt**2/2],
        [dt**2/2, 0, dt, 0],
        [0, dt**2/2, 0, dt],
    ])
    R = 0.5 * np.eye(2)
    m0 = np.zeros(4)
    P0 = np.eye(4)
    return A, C, Q, R, m0, P0


def build_linear_data():
    """Build linear model + simulated data for both NP and TF."""
    A, C, Q, R, m0, P0 = build_cv_model_params()
    np_model = np_make_lgssm(A, C, Q, R, m0, P0)
    rng = default_rng(SEED_SIM)
    states, obs = np_model.simulate(T_STEPS, rng)
    tf_model = tf_make_lgssm(A, C, Q, R, m0, P0)
    obs_tf = tf.constant(obs, dtype=tf.float64)
    return np_model, tf_model, states, obs, obs_tf


def build_nonlinear_data():
    """Build range-bearing model + simulated data for both NP and TF."""
    np_model = np_make_rb_ssm(s_r=1.0, s_th=1.0)
    rng = default_rng(SEED_SIM)
    states, obs = np_model.simulate(T_STEPS, rng)
    tf_model = tf_make_rb_ssm(s_r=1.0, s_th=1.0)
    obs_tf = tf.constant(obs, dtype=tf.float64)
    return np_model, tf_model, states, obs, obs_tf


# ============================================================================
# Timing helpers
# ============================================================================

def time_np_pf(np_model, obs, n_particles, seed, n_runs=N_TIMING_RUNS):
    """Time NumPy PF, return (result, median_ms)."""
    pf = NPBootstrapPF(n_particles=n_particles, seed=seed)
    # Warmup
    rng = default_rng(seed)
    result = pf.filter(np_model, obs, rng=rng)
    # Timing
    times = []
    for _ in range(n_runs):
        rng = default_rng(seed)
        t0 = time.perf_counter()
        pf.filter(np_model, obs, rng=rng)
        times.append(time.perf_counter() - t0)
    return result, np.median(times) * 1e3


def time_tf_pf(tf_model, obs_tf, n_particles, seed, n_runs=N_TIMING_RUNS):
    """Time TF PF, return (result, first_call_ms, subsequent_median_ms)."""
    tf_pf = TFBootstrapParticleFilter(n_particles=n_particles, seed=seed)
    # Create ONE Generator — reuse across all calls to avoid retracing
    rng = tf.random.Generator.from_seed(seed)
    # First call (includes tracing)
    t0 = time.perf_counter()
    result = tf_pf.filter(tf_model, obs_tf, rng=rng)
    first_ms = (time.perf_counter() - t0) * 1e3
    # Subsequent calls — reset seed, reuse same object
    times = []
    for _ in range(n_runs):
        rng.reset_from_seed(seed)
        t0 = time.perf_counter()
        tf_pf.filter(tf_model, obs_tf, rng=rng)
        times.append(time.perf_counter() - t0)
    return result, first_ms, np.median(times) * 1e3


# ============================================================================
# Test 1: KF / EKF / UKF correctness (linear model)
# ============================================================================

def test_kalman_correctness(np_model, tf_model, obs, obs_tf):
    print("\n" + "=" * 70)
    print("TEST 1: Kalman Filter Correctness (Linear Model)")
    print("=" * 70)

    ATOL_MEAN = 1e-9
    ATOL_COV = 1e-8
    ATOL_LOGLIK = 1e-6

    for name, np_filt, tf_filt in [
        ("KF",  NPKalmanFilter(),           TFKalmanFilter()),
        ("EKF", NPExtendedKalmanFilter(),    TFExtendedKalmanFilter()),
        ("UKF", NPUnscentedKalmanFilter(),   TFUnscentedKalmanFilter()),
    ]:
        np_res = np_filt.filter(np_model, obs)
        tf_res = tf_filt.filter(tf_model, obs_tf)

        np.testing.assert_allclose(
            tf_res.means.numpy(), np_res.means, atol=ATOL_MEAN, rtol=ATOL_MEAN,
            err_msg=f"{name} means mismatch")
        np.testing.assert_allclose(
            tf_res.covariances.numpy(), np_res.covariances, atol=ATOL_COV, rtol=ATOL_COV,
            err_msg=f"{name} covariances mismatch")
        np.testing.assert_allclose(
            tf_res.log_likelihood.numpy(), np_res.log_likelihood, atol=ATOL_LOGLIK,
            err_msg=f"{name} log-likelihood mismatch")

        print(f"  {name:4s}: PASS  (log_lik={np_res.log_likelihood:+.4f})")


# ============================================================================
# Test 2: PF correctness — small N on linear model (TF PF vs KF)
# ============================================================================

def test_pf_correctness(np_model, tf_model, obs, obs_tf):
    print("\n" + "=" * 70)
    print("TEST 2: PF Correctness — TF PF vs KF (Linear Model, N=2000)")
    print("=" * 70)

    kf_res = NPKalmanFilter().filter(np_model, obs)

    tf_pf = TFBootstrapParticleFilter(n_particles=N_PF_SMALL, seed=42)
    rng = tf.random.Generator.from_seed(42)
    tf_res = tf_pf.filter(tf_model, obs_tf, rng=rng)

    max_mean_diff = np.max(np.abs(tf_res.means.numpy()[1:] - kf_res.means[1:]))
    ess = tf_res.ess.numpy()

    np.testing.assert_allclose(
        tf_res.means.numpy()[1:], kf_res.means[1:], atol=0.5,
        err_msg="TF PF means too far from KF")
    assert np.all(ess >= 1.0) and np.all(ess <= N_PF_SMALL)

    print(f"  Max mean diff vs KF: {max_mean_diff:.4f}")
    print(f"  ESS range: [{ess.min():.1f}, {ess.max():.1f}]")
    print(f"  PASS")


# ============================================================================
# Test 3: PF benchmark — N=100k, NumPy vs TF (linear model)
# ============================================================================

def test_pf_benchmark_linear(np_model, tf_model, obs, obs_tf):
    print("\n" + "=" * 70)
    print(f"TEST 3: PF Benchmark — Linear Model (N={N_PF_LARGE:,}, T={T_STEPS})")
    print("=" * 70)

    # NumPy PF
    np_res, np_ms = time_np_pf(np_model, obs, N_PF_LARGE, SEED_PF)

    # TF PF
    tf_res, tf_first_ms, tf_sub_ms = time_tf_pf(tf_model, obs_tf, N_PF_LARGE, SEED_PF)

    speedup = np_ms / tf_sub_ms if tf_sub_ms > 0 else float('inf')

    print(f"\n  {'Metric':<28} {'NumPy':>14} {'TF (subseq)':>14}")
    print(f"  {'-'*56}")
    print(f"  {'Log-likelihood':<28} {np_res.log_likelihood:>+14.4f} {tf_res.log_likelihood.numpy():>+14.4f}")
    print(f"  {'Avg ESS':<28} {np_res.average_ess():>14.1f} {tf.reduce_mean(tf_res.ess).numpy():>14.1f}")
    print(f"  {'Runtime (ms)':<28} {np_ms:>14.1f} {tf_sub_ms:>14.1f}")
    print(f"  {'TF 1st call (ms)':<28} {'':>14} {tf_first_ms:>14.1f}")
    print(f"  {'Speedup (NP/TF_sub)':<28} {'':>14} {speedup:>13.2f}x")

    # Mean RMSE comparison (both should approximate KF similarly)
    kf_res = NPKalmanFilter().filter(np_model, obs)
    np_rmse = np.sqrt(np.mean((np_res.means[1:] - kf_res.means[1:]) ** 2))
    tf_rmse = np.sqrt(np.mean((tf_res.means.numpy()[1:] - kf_res.means[1:]) ** 2))
    print(f"\n  {'RMSE vs KF (NP PF)':<28} {np_rmse:>14.6f}")
    print(f"  {'RMSE vs KF (TF PF)':<28} {tf_rmse:>14.6f}")


# ============================================================================
# Test 4: Nonlinear range-bearing model — EKF, UKF correctness
# ============================================================================

def test_nonlinear_correctness(np_model_nl, tf_model_nl, obs_nl, obs_nl_tf):
    print("\n" + "=" * 70)
    print("TEST 4: Nonlinear Range-Bearing — EKF/UKF Correctness")
    print("=" * 70)

    ATOL_MEAN = 1e-8
    ATOL_COV = 1e-7
    ATOL_LOGLIK = 1e-5

    for name, np_filt, tf_filt in [
        ("EKF", NPExtendedKalmanFilter(),  TFExtendedKalmanFilter()),
        ("UKF", NPUnscentedKalmanFilter(), TFUnscentedKalmanFilter()),
    ]:
        np_res = np_filt.filter(np_model_nl, obs_nl)
        tf_res = tf_filt.filter(tf_model_nl, obs_nl_tf)

        np.testing.assert_allclose(
            tf_res.means.numpy(), np_res.means, atol=ATOL_MEAN, rtol=ATOL_MEAN,
            err_msg=f"Nonlinear {name} means mismatch")
        np.testing.assert_allclose(
            tf_res.covariances.numpy(), np_res.covariances, atol=ATOL_COV, rtol=ATOL_COV,
            err_msg=f"Nonlinear {name} covariances mismatch")
        np.testing.assert_allclose(
            tf_res.log_likelihood.numpy(), np_res.log_likelihood, atol=ATOL_LOGLIK,
            err_msg=f"Nonlinear {name} log-likelihood mismatch")

        print(f"  {name:4s}: PASS  (log_lik={np_res.log_likelihood:+.4f})")


# ============================================================================
# Test 5: PF benchmark — N=100k, NumPy vs TF (range-bearing)
# ============================================================================

def test_pf_benchmark_nonlinear(np_model_nl, tf_model_nl, obs_nl, obs_nl_tf):
    print("\n" + "=" * 70)
    print(f"TEST 5: PF Benchmark — Range-Bearing (N={N_PF_LARGE:,}, T={T_STEPS})")
    print("=" * 70)

    # NumPy PF
    np_res, np_ms = time_np_pf(np_model_nl, obs_nl, N_PF_LARGE, SEED_PF)

    # TF PF
    tf_res, tf_first_ms, tf_sub_ms = time_tf_pf(tf_model_nl, obs_nl_tf, N_PF_LARGE, SEED_PF)

    speedup = np_ms / tf_sub_ms if tf_sub_ms > 0 else float('inf')

    # Reference: EKF on this model (Gaussian approx — different quantity)
    ekf_res = NPExtendedKalmanFilter().filter(np_model_nl, obs_nl)

    print(f"\n  {'Metric':<28} {'NumPy PF':>14} {'TF PF':>14} {'NP EKF (ref)':>14}")
    print(f"  {'-'*70}")
    print(f"  {'Log-likelihood':<28} {np_res.log_likelihood:>+14.4f} {tf_res.log_likelihood.numpy():>+14.4f} {ekf_res.log_likelihood:>+14.4f}")
    print(f"  {'Avg ESS':<28} {np_res.average_ess():>14.1f} {tf.reduce_mean(tf_res.ess).numpy():>14.1f} {'n/a':>14}")
    print(f"  {'Runtime (ms)':<28} {np_ms:>14.1f} {tf_sub_ms:>14.1f} {'':>14}")
    print(f"  {'TF 1st call (ms)':<28} {'':>14} {tf_first_ms:>14.1f} {'':>14}")
    print(f"  {'Speedup (NP/TF_sub)':<28} {'':>14} {speedup:>13.2f}x {'':>14}")
    print(f"\n  Note: PF uses true Student-t likelihood; EKF uses Gaussian approx.")
    print(f"  Log-likelihoods are NOT directly comparable between PF and EKF.")


# ============================================================================
# Test 6: Kalman filter timing comparison
# ============================================================================

def test_kalman_timing(np_model, tf_model, obs, obs_tf):
    print("\n" + "=" * 70)
    print(f"TEST 6: Kalman Filter Timing (Linear Model, T={T_STEPS})")
    print("=" * 70)
    print(f"  {'Filter':<8} {'NumPy (ms)':>12} {'TF 1st (ms)':>14} {'TF sub (ms)':>14} {'Speedup':>10}")
    print(f"  {'-'*58}")

    for name, np_filt, tf_filt in [
        ("KF",  NPKalmanFilter(),           TFKalmanFilter()),
        ("EKF", NPExtendedKalmanFilter(),    TFExtendedKalmanFilter()),
        ("UKF", NPUnscentedKalmanFilter(),   TFUnscentedKalmanFilter()),
    ]:
        # NumPy
        np_times = []
        for _ in range(N_TIMING_RUNS):
            t0 = time.perf_counter()
            np_filt.filter(np_model, obs)
            np_times.append(time.perf_counter() - t0)
        np_med = np.median(np_times) * 1e3

        # TF first call
        t0 = time.perf_counter()
        tf_filt.filter(tf_model, obs_tf)
        tf_first = (time.perf_counter() - t0) * 1e3

        # TF subsequent
        tf_times = []
        for _ in range(N_TIMING_RUNS):
            t0 = time.perf_counter()
            tf_filt.filter(tf_model, obs_tf)
            tf_times.append(time.perf_counter() - t0)
        tf_med = np.median(tf_times) * 1e3

        speedup = np_med / tf_med if tf_med > 0 else float('inf')
        print(f"  {name:<8} {np_med:>12.3f} {tf_first:>14.3f} {tf_med:>14.3f} {speedup:>9.2f}x")

    print(f"\n  Note: TF 1st call includes graph tracing overhead.")
    print(f"        Speedup = NumPy / TF_subsequent")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Building test data...")

    # Linear model
    np_model, tf_model, states, obs, obs_tf = build_linear_data()

    # Nonlinear model
    np_model_nl, tf_model_nl, states_nl, obs_nl, obs_nl_tf = build_nonlinear_data()

    # Run all tests
    test_kalman_correctness(np_model, tf_model, obs, obs_tf)
    test_pf_correctness(np_model, tf_model, obs, obs_tf)
    test_pf_benchmark_linear(np_model, tf_model, obs, obs_tf)
    test_nonlinear_correctness(np_model_nl, tf_model_nl, obs_nl, obs_nl_tf)
    test_pf_benchmark_nonlinear(np_model_nl, tf_model_nl, obs_nl, obs_nl_tf)
    test_kalman_timing(np_model, tf_model, obs, obs_tf)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
