
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import tensorflow as tf
import numpy as np
import time
import tracemalloc
import pandas as pd
import gc
import psutil  # NEW: For CPU usage tracking
import os


# Import your modules
from ssm_models import RangeBearingModel
from filters import (
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    BootstrapParticleFilter
)

# Configuration
DTYPE = tf.float64
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

def get_process_stats():
    """Returns peak memory (MB) and average CPU usage (%) since last call."""
    _, peak = tracemalloc.get_traced_memory()
    peak_mb = peak / 10**6

    # Get CPU usage (blocking call for 0.1s to get a sample)
    cpu_pct = psutil.cpu_percent(interval=None)
    return peak_mb, cpu_pct

def rmse(est, truth):
    return np.sqrt(np.mean((est - truth)**2))

def run_experiment():
    print("==========================================================")
    print("      BENCHMARK: High-Load PF & CPU vs GPU Check")
    print("==========================================================")

    # 1. Generate Shared Data
    T = 200
    print(f"Generating {T} steps of Range-Bearing Data...")
    model_ssm = RangeBearingModel(dt=0.1, sigma_q=0.5, sigma_r=0.5)
    x_true, y_obs = model_ssm.generate(T=T)

    # Initial Guess
    x0_true = x_true[0]
    x0_guess = x0_true + tf.random.normal(x0_true.shape, stddev=1.0, dtype=DTYPE)
    P0 = tf.eye(4, dtype=DTYPE) * 2.0
    x0_matrix = tf.reshape(x0_guess, (1, 4))

    # --- RUNNER WRAPPER ---
    def execute_runner(runner_fn, device_name="/GPU:0"):
        """Generic function to run a filter on a specific device."""

        # Force placement on CPU or GPU
        try:
            with tf.device(device_name):
                # Garbage collect and reset stats
                gc.collect()
                tracemalloc.stop()
                tracemalloc.start()
                psutil.cpu_percent(interval=None) # Reset CPU counter

                start_time = time.time()

                # Execute
                estimates = runner_fn()

                elapsed = time.time() - start_time
                _, peak_mem = tracemalloc.get_traced_memory()
                cpu_usage = psutil.cpu_percent(interval=None) # Get usage over the duration

                return estimates, elapsed, peak_mem / 1e6, cpu_usage
        except RuntimeError as e:
            print(f"Device Error ({device_name}): {e}")
            return None, 0, 0, 0

    # --- FILTER DEFINITIONS ---
    def run_ekf():
        ekf = ExtendedKalmanFilter(model_ssm.f, model_ssm.h, model_ssm.Q, model_ssm.R_filter, P0, x0_matrix)
        ests = []
        for t in range(T):
            if t > 0: ekf.predict()
            est, _ = ekf.update(y_obs[t])
            ests.append(est.numpy()[0])
        return np.array(ests)

    def run_ukf():
        ukf = UnscentedKalmanFilter(model_ssm.f, model_ssm.h, model_ssm.Q, model_ssm.R_filter, P0, x0_matrix)
        # Fix t=0
        x_val = ukf.x.read_value()
        P_val = ukf.P.read_value()
        ukf.sigmas_f_var.assign(ukf.generate_sigma_points(x_val, P_val))

        ests = []
        for t in range(T):
            if t > 0: ukf.predict()
            est, _ = ukf.update(y_obs[t])
            ests.append(est.numpy()[0])
        return np.array(ests)

    def run_pf(n_particles):
        pf = BootstrapParticleFilter(model=model_ssm, num_particles=n_particles)
        pf.initialize(mean=x0_guess, cov=P0)
        ests = []
        for t in range(T):
            if t > 0: pf.predict()
            est = pf.update(y_obs[t])
            ests.append(est.numpy())
        return np.array(ests)

    # 2. WARM-UP (Crucial for GPU)
    print("Warm-up phase (initializing Metal kernels)...")
    try:
        with tf.device('/GPU:0'):
            run_pf(n_particles=100)
    except Exception:
        print("GPU Warmup failed, trying CPU...")

    results = []

    # 3. EXPERIMENT CONFIGURATIONS
    # We test EKF/UKF once, then scale PF up massively
    scenarios = [
        ("EKF", "GPU", run_ekf),
        ("UKF", "GPU", run_ukf),
        ("PF (N=500)",   "GPU", lambda: run_pf(500)),
        ("PF (N=10k)",   "GPU", lambda: run_pf(10000)),
        ("PF (N=50k)",   "GPU", lambda: run_pf(50000)),
        ("PF (N=100k)",  "GPU", lambda: run_pf(100000)),

        # RESOURCE COMPARISON: Run the same heavy PF on CPU
        ("PF (N=50k)",   "CPU", lambda: run_pf(50000)),
    ]

    print(f"{'Method':<15} | {'Device':<5} | {'Time (s)':<10} | {'RMSE':<8} | {'Mem (MB)':<10} | {'CPU %':<8}")
    print("-" * 75)

    final_data = []

    for name, device_type, func in scenarios:
        # Map nice name to TF device string
        tf_device = '/CPU:0' if device_type == 'CPU' else '/GPU:0'

        est, t_sec, mem_mb, cpu_pct = execute_runner(func, tf_device)

        if est is not None:
            err = rmse(est, x_true.numpy())

            print(f"{name:<15} | {device_type:<5} | {t_sec:<10.4f} | {err:<8.4f} | {mem_mb:<10.2f} | {cpu_pct:<8.1f}")

            final_data.append({
                "Method": name,
                "Device": device_type,
                "Time": t_sec,
                "RMSE": err,
                "Memory_MB": mem_mb,
                "CPU_Util": cpu_pct
            })
        else:
            print(f"{name} Failed.")

    # 4. Summary
    df = pd.DataFrame(final_data)
    print("\nSummary Table:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_experiment()
