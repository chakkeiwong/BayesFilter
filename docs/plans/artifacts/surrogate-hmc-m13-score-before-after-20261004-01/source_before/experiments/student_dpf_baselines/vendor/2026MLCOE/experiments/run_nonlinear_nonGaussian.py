"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-02-26
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import modular library components
from src.filters.classical import DTYPE, EKF, UKF
from src.filters.particle import BPF
from src.models.classic_ssm import StochasticVolatilityModel, RangeBearingModel

# FORCE CPU EXECUTION for Float64 Precision
tf.config.set_visible_devices([], 'GPU')

# Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)
tf.config.run_functions_eagerly(False)

# ==========================================
# 1. Visualization Functions
# ==========================================

def plot_sv_simulation(x_true, y_obs):
    T = len(x_true)
    plt.figure(figsize=(12, 4))
    plt.plot(x_true, 'b-', linewidth=1.0, label='True Volatility')
    plt.plot(y_obs, 'r*', markersize=4, label='Observations')
    plt.title("Stochastic Volatility: Simulated Sequence")
    plt.xlabel("Time Step"); plt.xlim([0, T])
    y_min, y_max = np.min(y_obs), np.max(y_obs)
    plt.ylim([y_min-2, y_max+2])
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.tick_params(direction='in', top=True, right=True)
    plt.savefig('sv_setup.png')

def plot_sv_results(x_true, x_ekf, x_ukf, x_pf, rmse_ekf, rmse_ukf, rmse_pf):
    plt.figure(figsize=(12, 5))
    plt.plot(x_true, 'k-', alpha=0.2, linewidth=3, label='True State')
    plt.plot(x_ekf, 'b--', linewidth=1.0, label=f'EKF (RMSE={rmse_ekf:.3f})')
    plt.plot(x_ukf, 'g-.', linewidth=1.0, label=f'UKF (RMSE={rmse_ukf:.3f})')
    plt.plot(x_pf, 'r:', linewidth=1.5, label=f'PF (RMSE={rmse_pf:.3f})')
    plt.title("Stochastic Volatility: Filter Comparison")
    plt.xlabel("Time Step"); plt.ylabel("Log-Volatility")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('sv_results.png')

def plot_rb_simulation(x_true):
    plt.figure(figsize=(8, 8))
    plt.plot(x_true[:, 0], x_true[:, 1], 'k-', linewidth=2.0, label='True Trajectory')
    plt.scatter(0, 0, marker='x', color='g', s=200, linewidths=3, label='Sensor')
    plt.title("Range-Bearing: Nonlinear Trajectory Setup")
    plt.xlabel("X Position"); plt.ylabel("Y Position"); plt.legend(); plt.grid(True); plt.axis('equal')
    plt.savefig('rb_setup.png')

def plot_rb_results(x_true, x_ekf, x_ukf, x_pf, rmse_ekf, rmse_ukf, rmse_pf):
    plt.figure(figsize=(10, 8))
    plt.plot(x_true[:, 0], x_true[:, 1], 'k-', linewidth=1.5, label='True')
    plt.plot(x_ekf[:, 0], x_ekf[:, 1], 'b--', linewidth=1.5, label=f'EKF (RMSE={rmse_ekf:.3f})')
    plt.plot(x_ukf[:, 0], x_ukf[:, 1], 'g-.', linewidth=1.5, label=f'UKF (RMSE={rmse_ukf:.3f})')
    plt.plot(x_pf[:, 0], x_pf[:, 1], 'r:', linewidth=2.0, label=f'PF (RMSE={rmse_pf:.3f})')
    plt.scatter(0, 0, marker='x', color='g', s=100, label='Sensor')
    plt.title("Range-Bearing: Filter Trajectory Comparison")
    plt.xlabel("X Position"); plt.ylabel("Y Position"); plt.legend(); plt.grid(True); plt.axis('equal')
    plt.savefig('rb_results.png')


# ==========================================
# 2. Main Driver (TensorFlow / CPU Float64)
# ==========================================

def run_unified_demo():
    print("="*60 + "\nUNIFIED NONLINEAR FILTERING FRAMEWORK (TensorFlow CPU / Float64)\n" + "="*60)

    with tf.device('/CPU:0'):
        # --- Scenario 1: Stochastic Volatility ---
        print("\n[Scenario 1] Stochastic Volatility Model")
        sv = StochasticVolatilityModel()
        x_true_sv_tf, y_obs_sv_tf = sv.generate(T=300)
        x_true_sv = x_true_sv_tf.numpy()
        y_obs_sv = y_obs_sv_tf.numpy()

        # Init Filters (Passing model instance)
        ekf_sv = EKF(sv)
        ukf_sv = UKF(sv)
        pf_sv = BPF(sv, N=5000)

        # Use Rank-1 [dim,] for EKF/UKF internal compatibility
        x0_sv = tf.constant([0.0], dtype=DTYPE)
        P0_sv = tf.constant([[5.0]], dtype=DTYPE)

        ekf_sv.init(x0_sv, P0_sv)
        ukf_sv.init(x0_sv, P0_sv)
        pf_sv.init(x0_sv, P0_sv)

        est_ekf, est_ukf, est_pf = [], [], []

        print("Running SV Filters...")
        start_time = time.time()
        for t in range(len(y_obs_sv)):
            y_raw = y_obs_sv_tf[t]
            # Transform observation and ensure Rank-1 for matvec
            z_trans = tf.reshape(sv.transform_obs(y_raw), [1])

            # Use .step(z) to match existing modular implementation
            est_ekf.append(ekf_sv.step(z_trans).numpy()[0])
            est_ukf.append(ukf_sv.step(z_trans).numpy()[0])
            est_pf.append(pf_sv.step(y_raw).numpy()[0])

        print(f"Scenario 1 Compute Time: {time.time() - start_time:.4f}s")
        x_ekf, x_ukf, x_pf = [np.array(e).reshape(-1, 1) for e in [est_ekf, est_ukf, est_pf]]
        rmse_ekf = np.sqrt(np.mean((x_ekf - x_true_sv)**2))
        rmse_ukf = np.sqrt(np.mean((x_ukf - x_true_sv)**2))
        rmse_pf = np.sqrt(np.mean((x_pf - x_true_sv)**2))

        print(f"SV RMSE -> EKF: {rmse_ekf:.4f} | UKF: {rmse_ukf:.4f} | PF: {rmse_pf:.4f}")
        plot_sv_simulation(x_true_sv, y_obs_sv)
        plot_sv_results(x_true_sv, x_ekf, x_ukf, x_pf, rmse_ekf, rmse_ukf, rmse_pf)

        # --- Scenario 2: Range-Bearing Tracking ---
        print("\n[Scenario 2] Range-Bearing Tracking")
        rb = RangeBearingModel()
        x_true_rb_tf, y_obs_rb_tf = rb.generate(T=200)
        x_true_rb = x_true_rb_tf.numpy()

        # Init Filters
        ekf_rb = EKF(rb)
        ukf_rb = UKF(rb)
        pf_rb = BPF(rb, N=5000)

        # Rank-1 for EKF/UKF matvec compatibility
        x0_rb = tf.constant([0.0, 0.0, 4.0, 2.0], dtype=DTYPE)
        P0_rb = tf.eye(4, dtype=DTYPE)

        ekf_rb.init(x0_rb, P0_rb)
        ukf_rb.init(x0_rb, P0_rb)
        pf_rb.init(x0_rb, P0_rb)

        est_ekf_rb, est_ukf_rb, est_pf_rb = [], [], []

        print("Running RB Filters...")
        start_time = time.time()
        for t in range(len(y_obs_rb_tf)):
            y_t = tf.reshape(y_obs_rb_tf[t], [2]) # Rank-1

            est_ekf_rb.append(ekf_rb.step(y_t).numpy())
            est_ukf_rb.append(ukf_rb.step(y_t).numpy())
            est_pf_rb.append(pf_rb.step(y_t).numpy())

        print(f"Scenario 2 Compute Time: {time.time() - start_time:.4f}s")
        x_ekf_rb, x_ukf_rb, x_pf_rb = [np.array(e) for e in [est_ekf_rb, est_ukf_rb, est_pf_rb]]
        rmse_ekf_rb = np.sqrt(np.mean((x_ekf_rb[:,:2] - x_true_rb[:,:2])**2))
        rmse_ukf_rb = np.sqrt(np.mean((x_ukf_rb[:,:2] - x_true_rb[:,:2])**2))
        rmse_pf_rb = np.sqrt(np.mean((x_pf_rb[:,:2] - x_true_rb[:,:2])**2))

        print(f"RB Pos RMSE -> EKF: {rmse_ekf_rb:.4f} | UKF: {rmse_ukf_rb:.4f} | PF: {rmse_pf_rb:.4f}")
        plot_rb_simulation(x_true_rb)
        plot_rb_results(x_true_rb, x_ekf_rb, x_ukf_rb, x_pf_rb, rmse_ekf_rb, rmse_ukf_rb, rmse_pf_rb)

if __name__ == "__main__":
    run_unified_demo()
