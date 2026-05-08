"""
Phase 0: Capture NumPy baseline benchmarks.

Saves numerical outputs + runtime + peak memory for each filter (KF, EKF, UKF, PF)
on both a linear and nonlinear model. These are the reference for TF migration validation.

Run:  python -m advanced_particle_filter.benchmark_numpy_baseline
  or: python advanced_particle_filter/benchmark_numpy_baseline.py

Outputs:  numpy_baseline.npz  (arrays + metadata)
"""

import time
import tracemalloc
import numpy as np
from numpy.random import default_rng

from advanced_particle_filter.models.linear_gaussian import make_lgssm
from advanced_particle_filter.models.range_bearing import make_range_bearing_ssm
from advanced_particle_filter.filters.kalman import (
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
)
from advanced_particle_filter.filters.particle import BootstrapParticleFilter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
T_LINEAR = 100
T_NONLINEAR = 100
N_PARTICLES = 100000
N_TIMING_RUNS = 10  # median over this many runs
SEED_SIM = 42
SEED_PF = 123


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_filter(filter_obj, model, observations, n_runs=N_TIMING_RUNS, **kwargs):
    """Return (result, median_seconds, peak_memory_bytes)."""
    # Warmup
    result = filter_obj.filter(model, observations, **kwargs)

    # Timing
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        filter_obj.filter(model, observations, **kwargs)
        times.append(time.perf_counter() - t0)
    median_time = np.median(times)

    # Memory (single run)
    tracemalloc.start()
    filter_obj.filter(model, observations, **kwargs)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, median_time, peak_mem


def time_pf(filter_obj, model, observations, rng_seed, n_runs=N_TIMING_RUNS):
    """PF needs a fresh rng each run for reproducibility measurement."""
    # Warmup + reference result (fixed seed)
    rng = default_rng(rng_seed)
    result = filter_obj.filter(model, observations, rng=rng)

    # Timing (each run uses a fresh rng with same seed for consistency)
    times = []
    for _ in range(n_runs):
        rng = default_rng(rng_seed)
        t0 = time.perf_counter()
        filter_obj.filter(model, observations, rng=rng)
        times.append(time.perf_counter() - t0)
    median_time = np.median(times)

    # Memory
    tracemalloc.start()
    rng = default_rng(rng_seed)
    filter_obj.filter(model, observations, rng=rng)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, median_time, peak_mem


# ---------------------------------------------------------------------------
# Linear Gaussian model
# ---------------------------------------------------------------------------

def build_linear_model():
    """2D constant-velocity model with position observations."""
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
    return make_lgssm(A, C, Q, R, m0, P0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}

    # ---- Linear model ----
    print("=" * 60)
    print("LINEAR GAUSSIAN MODEL")
    print("=" * 60)

    model_lin = build_linear_model()
    rng_sim = default_rng(SEED_SIM)
    states_lin, obs_lin = model_lin.simulate(T_LINEAR, rng_sim)

    results["linear_states"] = states_lin
    results["linear_obs"] = obs_lin

    for name, filt in [
        ("KF", KalmanFilter()),
        ("EKF", ExtendedKalmanFilter()),
        ("UKF", UnscentedKalmanFilter()),
    ]:
        res, t_med, mem = time_filter(filt, model_lin, obs_lin)
        print(f"  {name:4s}  time={t_med*1e3:8.3f} ms  mem={mem/1024:8.1f} KB  "
              f"log_lik={res.log_likelihood:+.4f}")

        results[f"linear_{name}_means"] = res.means
        results[f"linear_{name}_covs"] = res.covariances
        results[f"linear_{name}_loglik"] = np.array(res.log_likelihood)
        results[f"linear_{name}_loglik_inc"] = res.log_likelihood_increments
        results[f"linear_{name}_time_ms"] = np.array(t_med * 1e3)
        results[f"linear_{name}_mem_kb"] = np.array(mem / 1024)

    # PF
    pf = BootstrapParticleFilter(n_particles=N_PARTICLES, seed=SEED_PF)
    res_pf, t_med, mem = time_pf(pf, model_lin, obs_lin, SEED_PF)
    print(f"  PF    time={t_med*1e3:8.3f} ms  mem={mem/1024:8.1f} KB  "
          f"log_lik={res_pf.log_likelihood:+.4f}  avg_ess={res_pf.average_ess():.1f}")

    results["linear_PF_means"] = res_pf.means
    results["linear_PF_covs"] = res_pf.covariances
    results["linear_PF_loglik"] = np.array(res_pf.log_likelihood)
    results["linear_PF_loglik_inc"] = res_pf.log_likelihood_increments
    results["linear_PF_ess"] = res_pf.ess
    results["linear_PF_time_ms"] = np.array(t_med * 1e3)
    results["linear_PF_mem_kb"] = np.array(mem / 1024)

    # ---- Nonlinear model (range-bearing) ----
    print()
    print("=" * 60)
    print("NONLINEAR RANGE-BEARING MODEL (Student-t noise)")
    print("=" * 60)

    model_nl = make_range_bearing_ssm()
    rng_sim2 = default_rng(SEED_SIM)
    states_nl, obs_nl = model_nl.simulate(T_NONLINEAR, rng_sim2)

    results["nonlinear_states"] = states_nl
    results["nonlinear_obs"] = obs_nl

    for name, filt in [
        ("EKF", ExtendedKalmanFilter()),
        ("UKF", UnscentedKalmanFilter()),
    ]:
        res, t_med, mem = time_filter(filt, model_nl, obs_nl)
        print(f"  {name:4s}  time={t_med*1e3:8.3f} ms  mem={mem/1024:8.1f} KB  "
              f"log_lik={res.log_likelihood:+.4f}")

        results[f"nonlinear_{name}_means"] = res.means
        results[f"nonlinear_{name}_covs"] = res.covariances
        results[f"nonlinear_{name}_loglik"] = np.array(res.log_likelihood)
        results[f"nonlinear_{name}_loglik_inc"] = res.log_likelihood_increments
        results[f"nonlinear_{name}_time_ms"] = np.array(t_med * 1e3)
        results[f"nonlinear_{name}_mem_kb"] = np.array(mem / 1024)

    # PF on nonlinear
    pf_nl = BootstrapParticleFilter(n_particles=N_PARTICLES, seed=SEED_PF)
    res_pf_nl, t_med, mem = time_pf(pf_nl, model_nl, obs_nl, SEED_PF)
    print(f"  PF    time={t_med*1e3:8.3f} ms  mem={mem/1024:8.1f} KB  "
          f"log_lik={res_pf_nl.log_likelihood:+.4f}  avg_ess={res_pf_nl.average_ess():.1f}")

    results["nonlinear_PF_means"] = res_pf_nl.means
    results["nonlinear_PF_covs"] = res_pf_nl.covariances
    results["nonlinear_PF_loglik"] = np.array(res_pf_nl.log_likelihood)
    results["nonlinear_PF_loglik_inc"] = res_pf_nl.log_likelihood_increments
    results["nonlinear_PF_ess"] = res_pf_nl.ess
    results["nonlinear_PF_time_ms"] = np.array(t_med * 1e3)
    results["nonlinear_PF_mem_kb"] = np.array(mem / 1024)

    # ---- Save ----
    outpath = "numpy_baseline.npz"
    np.savez(outpath, **results)
    print(f"\nSaved baseline to {outpath}")
    print(f"Keys: {sorted(results.keys())}")


if __name__ == "__main__":
    main()
