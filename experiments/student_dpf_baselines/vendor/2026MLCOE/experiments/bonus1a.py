"""
Andrieu (2010) Non-linear SSM Benchmark
=======================================
This script evaluates the Particle Flow Particle Filter (PFPF) on the non-linear
benchmark from Andrieu et al. (2010). The model is characterized by a
non-linear transition and a quadratic observation function, posing a significant
challenge for standard Sequential Monte Carlo methods.

Implementation Details:
- Dynamics: Imported from src.models.classic_ssm.
- Filtering: PFPF-LEDH with importance weight tracking via log-determinants.
- Hardware: Optimized for CPU execution on M1 Ultra for numerical stability.

Author: Joowon Lee
Date: 2026-02-27
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =============================================================================
# PROJECT CONFIGURATION & MODULAR IMPORTS
# =============================================================================

# Ensure project root is in path to discover the 'src' package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Modular imports from the unified library
from src.filters.flow_filters import ParticleFlowFilter
from src.models.classic_ssm import AndrieuModel
from src.filters.classical import DTYPE

# Configuration for M1 Ultra stability (Avoids Metal-specific float64 drift)
tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.set_floatx('float64')


# =============================================================================
# 1. SIMULATION RUNNER
# =============================================================================


def run_simulation(T=100, N=100, save_name='pfpf_andrieu_results'):
    # Force CPU to avoid Metal colocation errors on M1 Ultra
    with tf.device('/CPU:0'):
        model = AndrieuModel()

        # Ground Truth Generation
        true_x, obs_y = np.zeros(T), np.zeros(T)
        curr_x = np.random.normal(0, np.sqrt(5))
        for t in range(T):
            # n = t + 1 per paper notation
            curr_x = (curr_x / 2.0) + (25.0 * curr_x / (1.0 + curr_x**2)) + \
                     8.0 * np.cos(1.2 * (t+1)) + np.random.normal(0, np.sqrt(10))
            true_x[t] = curr_x
            obs_y[t] = (curr_x**2 / 20.0) + np.random.normal(0, 1.0)

        # Initialize Invertible PFPF-LEDH
        pfpf = ParticleFlowFilter(model, N=N, mode='ledh', is_pfpf=True)
        pfpf.init(tf.cast([0.0], DTYPE), tf.cast([[5.0]], DTYPE))

        estimates = []
        print("Executing Invertible PFPF-LEDH Estimation (CPU Mode)...")
        for t in range(T):
            z = tf.cast([[obs_y[t]]], DTYPE)

            # Prediction with time-dependent transition
            noise = tf.random.normal((N, 1), stddev=np.sqrt(10), dtype=DTYPE)
            x_pred = model.transition(pfpf.X, t+1) + noise
            pfpf.X.assign(tf.reshape(x_pred, [N, 1]))

            # Flow step with log-det tracking
            est = pfpf.step(z)
            estimates.append(est.numpy()[0])

            if (t+1) % 20 == 0:
                print(f"Step {t+1}/{T} | ESS: {pfpf.ess.numpy():.2f}")

    # Results Calculation and Storage
    est_arr = np.array(estimates)
    rmse_val = np.sqrt(np.mean((true_x - est_arr)**2))

    results_dict = {
        'true_x': true_x,
        'obs_y': obs_y,
        'estimates': est_arr,
        'rmse': rmse_val
    }
    np.save(f"{save_name}.npy", results_dict)
    print(f"\nData saved to {save_name}.npy")

    # Final Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(true_x, color='black', label='Ground Truth', linewidth=1.5, alpha=0.8)
    plt.plot(est_arr, color='red', linestyle='--',
             label=f'PFPF-LEDH (RMSE: {rmse_val:.4f})', linewidth=1.5)
    plt.scatter(range(T), obs_y, color='blue', s=15, alpha=0.2, label='Observations ($Y_n$)')

    plt.title("State Estimation: Non-linear SSM (Andrieu et al. 2010)")
    plt.xlabel("Time Step ($n$)")
    plt.ylabel("Latent State ($X_n$)")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(f"{save_name}.png", bbox_inches='tight', dpi=300)
    print(f"Plot saved to {save_name}.png")
    plt.show()

if __name__ == "__main__":
    run_simulation()
