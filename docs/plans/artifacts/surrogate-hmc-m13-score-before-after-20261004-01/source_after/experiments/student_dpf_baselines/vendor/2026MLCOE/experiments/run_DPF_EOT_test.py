"""
EOT Regularization Trade-off Benchmark
======================================
This script analyzes the fundamental bias-variance trade-off in Entropic Optimal
Transport (EOT) resampling. By varying the entropy parameter (epsilon) across
orders of magnitude, we evaluate:

1. Barycentric Shrinkage: How high regularization biases the particle
   distribution toward the weighted mean, reducing variance but increasing RMSE.
2. Numerical Stability: Identifying the 'cliff' where low epsilon values
   lead to gradient instability.
3. Computational Efficiency: Benchmarking Sinkhorn iteration convergence speed.

The experiment uses a bimodal dynamics model with a stochastic jump component
to specifically penalize over-regularized flows that fail to capture mode-switches.

Author: Joowon Lee
Date: 2026-02-27
"""

import os
import sys
import time
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

# Import the DPF framework from the filters directory
from src.filters.DPF import DPF

# High-precision CPU execution for numerical stability in Sinkhorn loops
tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.set_floatx('float32')



class BimodalDynamics(tf.Module):
    """Dynamics with a mode-switch to penalize over-regularization."""
    def __init__(self, phi=0.9, sigma=0.5):
        self.phi, self.sigma = phi, sigma

    def __call__(self, x_prev, h_prev, noise):
        # 5% probability of jumping to a secondary state
        jump = tf.where(tf.random.uniform(tf.shape(x_prev)) > 0.95, 4.0, 0.0)
        x_curr = self.phi * x_prev + jump + self.sigma * noise
        return x_curr, h_prev

class SharpLikelihood(tf.Module):
    """High-precision likelihood to reveal the bias of barycentric shrinkage."""
    def __call__(self, x, y):
        return -0.5 * (tf.square(y - x) / 0.01)

def run_stochastic_tradeoff_benchmark(num_trials=10, T=50, N=256, save_prefix='eot_tradeoff'):
    """Benchmark EOT across epsilons and save results to disk."""
    DATA_FILE = f'{save_prefix}_data.npy'
    PLOT_FILE = f'{save_prefix}_plot.png'

    eps_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    all_results = []

    print(f"{'Epsilon':<10} | {'RMSE (Bias)':<12} | {'Std (Var)':<10} | {'Speed (ms)'}")
    print("-" * 55)

    for eps in eps_values:
        trial_rmses, trial_times = [], []

        for _ in range(num_trials):
            # RANDOMIZATION: Vary switch timing/magnitude to recover variance
            switch_t = np.random.randint(15, 35)
            jump_mag = np.random.uniform(3.0, 6.0)

            true_x = np.zeros(T)
            for t in range(1, T):
                true_x[t] = 0.9 * true_x[t-1] + (jump_mag if t == switch_t else 0.0) + np.random.normal(0, 0.1)
            obs_y = true_x + np.random.normal(0, 0.1)

            f, g = BimodalDynamics(sigma=0.5), SharpLikelihood()
            dpf = DPF(f, g, N)

            p = tf.random.normal((N,))
            h = [tf.zeros((N, 32))]
            lw = tf.fill((N,), -np.log(N, dtype='float32'))

            start = time.time()
            ests = []
            for t in range(T):
                p, h, lw = dpf.filter_step(p, h, lw, obs_y[t], method='sinkhorn', epsilon=eps)
                ests.append(tf.reduce_sum(p * tf.exp(lw)).numpy())

            trial_times.append((time.time() - start) * 1000)
            trial_rmses.append(np.sqrt(np.mean(np.square(np.array(ests) - true_x))))

        m_rmse, s_rmse = np.mean(trial_rmses), np.std(trial_rmses)
        m_time = np.mean(trial_times)
        all_results.append({'eps': eps, 'rmse': m_rmse, 'std': s_rmse, 'time': m_time})
        print(f"{eps:<10} | {m_rmse:<12.4f} | {s_rmse:<10.4f} | {m_time:.2f}")

    # 1. Save Raw Data as .npy
    np.save(DATA_FILE, all_results)
    print(f"\nResults saved to {DATA_FILE}")

    # 2. Generate and Save Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    eps_p = [r['eps'] for r in all_results]
    rmse_p, std_p, speed_p = [r['rmse'] for r in all_results], [r['std'] for r in all_results], [r['time'] for r in all_results]

    ax1.errorbar(eps_p, rmse_p, yerr=std_p, fmt='-o', color='blue', capsize=5, label='RMSE (Bias $\pm$ Var)')
    ax1.set_xscale('log'); ax1.set_xlabel("$\epsilon$ (Regularization)", fontsize=12); ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_title("EOT Bias and Variance Trade-off", fontsize=14); ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(eps_p, speed_p, '-o', color='red')
    ax2.set_xscale('log'); ax2.set_xlabel("$\epsilon$ (Regularization)", fontsize=12); ax2.set_ylabel("Execution Time (ms)", fontsize=12)
    ax2.set_title("Computational Speed vs. $\epsilon$", fontsize=14); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, bbox_inches='tight') # Eliminate white space
    print(f"Plot saved to {PLOT_FILE}")
    plt.show()

if __name__ == "__main__":
    run_stochastic_tradeoff_benchmark(num_trials=10, T=50, N=64)
