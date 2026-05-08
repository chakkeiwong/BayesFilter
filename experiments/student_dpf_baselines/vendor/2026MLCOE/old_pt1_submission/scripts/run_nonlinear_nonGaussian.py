
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# FORCE CPU EXECUTION for Float64 Precision (Required for M1 Accuracy)
tf.config.set_visible_devices([], 'GPU')

# Import shared utilities and TensorFlow filters/models
from filters import DTYPE, ExtendedKalmanFilter, UnscentedKalmanFilter, BootstrapParticleFilter
from ssm_models import StochasticVolatilityModel, RangeBearingModel

# Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

# Disable eager execution for performance
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
    print(">> Plot saved to sv_setup.png")

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
    print(">> Plot saved to sv_results.png")

def plot_rb_simulation(x_true):
    plt.figure(figsize=(8, 8))
    plt.plot(x_true[:, 0], x_true[:, 1], 'k-', linewidth=2.0, label='True Trajectory')
    plt.plot(x_true[0, 0], x_true[0, 1], 'go', markersize=8, label='Start')
    plt.plot(x_true[-1, 0], x_true[-1, 1], 'rs', markersize=8, label='End')
    plt.scatter(0, 0, marker='x', color='g', s=200, linewidths=3, label='Sensor')
    plt.title("Range-Bearing: Nonlinear Trajectory Setup")
    plt.xlabel("X Position"); plt.ylabel("Y Position")
    plt.legend(); plt.grid(True); plt.axis('equal')
    plt.tick_params(direction='in', top=True, right=True)
    plt.savefig('rb_setup.png')
    print(">> Plot saved to rb_setup.png")

def plot_rb_results(x_true, x_ekf, x_ukf, x_pf, rmse_ekf, rmse_ukf, rmse_pf):
    plt.figure(figsize=(10, 8))
    plt.plot(x_true[:, 0], x_true[:, 1], 'k-', linewidth=1.5, label='True')
    plt.plot(x_ekf[:, 0], x_ekf[:, 1], 'b--', linewidth=1.5, label=f'EKF (RMSE={rmse_ekf:.3f})')
    plt.plot(x_ukf[:, 0], x_ukf[:, 1], 'g-.', linewidth=1.5, label=f'UKF (RMSE={rmse_ukf:.3f})')
    plt.plot(x_pf[:, 0], x_pf[:, 1], 'r:', linewidth=2.0, label=f'PF (RMSE={rmse_pf:.3f})')
    plt.scatter(0, 0, marker='x', color='g', s=100, label='Sensor')

    plt.title("Range-Bearing: Filter Trajectory Comparison")
    plt.xlabel("X Position"); plt.ylabel("Y Position")
    plt.legend(); plt.grid(True); plt.axis('equal')
    plt.savefig('rb_results.png')
    print(">> Plot saved to rb_results.png")


# ==========================================
# 2. Main Driver (TensorFlow / CPU Float64)
# ==========================================

def run_unified_demo():
    print("="*60 + "\nUNIFIED NONLINEAR FILTERING FRAMEWORK (TensorFlow CPU / Float64)\n" + "="*60)

    # CRITICAL: Force all operations to CPU to support float64 and avoid M1 GPU crashes
    with tf.device('/CPU:0'):

        # -------------------------------------------------------------------------
        # [Scenario 1] Stochastic Volatility Model
        # -------------------------------------------------------------------------
        print("\n[Scenario 1] Stochastic Volatility Model")
        sv = StochasticVolatilityModel()

        # Generate Data
        x_true_sv_tf, y_obs_sv_tf = sv.generate(T=300)
        x_true_sv = x_true_sv_tf.numpy()
        y_obs_sv = y_obs_sv_tf.numpy()

        # --- Initialize Filters ---
        P0_sv = tf.constant([[5.0]], dtype=DTYPE)
        x0_sv = tf.constant([[0.0]], dtype=DTYPE)

        ekf_sv = ExtendedKalmanFilter(sv.f, sv.h, sv.Q, sv.R_filter, P0_sv, x0_sv)
        ukf_sv = UnscentedKalmanFilter(sv.f, sv.h, sv.Q, sv.R_filter, P0_sv, x0_sv)

        # 1000 particles is sufficient for 1D SV model
        pf_sv = BootstrapParticleFilter(sv, num_particles=5000)
        pf_sv.initialize(x0_sv, P0_sv)

        # --- Run Filters ---
        est_ekf = []
        est_ukf = []
        est_pf = []

        print("Running SV Filters (Float64 CPU)...")
        start_time = time.time()

        for t in range(len(y_obs_sv)):
            y_raw = y_obs_sv_tf[t]
            z_trans = sv.transform_obs(y_raw)
            z_trans = tf.reshape(z_trans, [1, 1])

            # EKF/UKF
            ekf_sv.predict()
            x_e, _ = ekf_sv.update(z_trans)
            est_ekf.append(x_e.numpy()[0,0])

            ukf_sv.predict()
            x_u, _ = ukf_sv.update(z_trans)
            est_ukf.append(x_u.numpy()[0,0])

            # PF
            pf_sv.predict()
            y_raw_reshaped = tf.reshape(y_raw, [1])
            x_p = pf_sv.update(y_raw_reshaped)
            est_pf.append(x_p.numpy()[0])

        print(f"Scenario 1 Compute Time: {time.time() - start_time:.4f}s")

        x_ekf = np.array(est_ekf).reshape(-1, 1)
        x_ukf = np.array(est_ukf).reshape(-1, 1)
        x_pf = np.array(est_pf).reshape(-1, 1)

        rmse_ekf = np.sqrt(np.mean((x_ekf - x_true_sv)**2))
        rmse_ukf = np.sqrt(np.mean((x_ukf - x_true_sv)**2))
        rmse_pf = np.sqrt(np.mean((x_pf - x_true_sv)**2))

        print(f"SV RMSE -> EKF: {rmse_ekf:.4f} | UKF: {rmse_ukf:.4f} | PF: {rmse_pf:.4f}")

        plot_sv_simulation(x_true_sv, y_obs_sv)
        plot_sv_results(x_true_sv, x_ekf, x_ukf, x_pf, rmse_ekf, rmse_ukf, rmse_pf)


        # -------------------------------------------------------------------------
        # [Scenario 2] Range-Bearing Tracking
        # -------------------------------------------------------------------------
        print("\n[Scenario 2] Range-Bearing Tracking (Nonlinear Trajectory)")
        rb = RangeBearingModel()

        # Generate Data
        x_true_rb_tf, y_obs_rb_tf = rb.generate(T=200)
        x_true_rb = x_true_rb_tf.numpy()
        y_obs_rb = y_obs_rb_tf.numpy()

        # --- Initialize Filters ---
        x0_rb = tf.constant([[0.0, 0.0, 4.0, 2.0]], dtype=DTYPE).numpy()
        P0_rb = tf.eye(4, dtype=DTYPE)

        ekf_rb = ExtendedKalmanFilter(rb.f, rb.h, rb.Q, rb.R_filter, P0_rb, tf.transpose(x0_rb))
        ukf_rb = UnscentedKalmanFilter(rb.f, rb.h, rb.Q, rb.R_filter, P0_rb, tf.transpose(x0_rb))

        pf_rb = BootstrapParticleFilter(rb, num_particles=5000)
        pf_rb.initialize(x0_rb[0], P0_rb)

        # --- Run Filters ---
        est_ekf_rb = []
        est_ukf_rb = []
        est_pf_rb = []

        print("Running RB Filters (Float64 CPU)...")
        start_time = time.time()
        for t in range(len(y_obs_rb)):
            y_t = tf.reshape(y_obs_rb_tf[t], [1, 2])

            ekf_rb.predict()
            x_e, _ = ekf_rb.update(tf.transpose(y_t))
            est_ekf_rb.append(x_e.numpy().flatten())

            ukf_rb.predict()
            x_u, _ = ukf_rb.update(tf.transpose(y_t))
            est_ukf_rb.append(x_u.numpy().flatten())

            pf_rb.predict()
            x_p = pf_rb.update(y_t)
            est_pf_rb.append(x_p.numpy().flatten())

        print(f"Scenario 2 Compute Time: {time.time() - start_time:.4f}s")

        x_ekf_rb = np.array(est_ekf_rb)
        x_ukf_rb = np.array(est_ukf_rb)
        x_pf_rb = np.array(est_pf_rb)

        rmse_ekf_rb = np.sqrt(np.mean((x_ekf_rb[:,:2] - x_true_rb[:,:2])**2))
        rmse_ukf_rb = np.sqrt(np.mean((x_ukf_rb[:,:2] - x_true_rb[:,:2])**2))
        rmse_pf_rb = np.sqrt(np.mean((x_pf_rb[:,:2] - x_true_rb[:,:2])**2))

        print(f"RB Pos RMSE -> EKF: {rmse_ekf_rb:.4f} | UKF: {rmse_ukf_rb:.4f} | PF: {rmse_pf_rb:.4f}")

        plot_rb_simulation(x_true_rb)
        plot_rb_results(x_true_rb, x_ekf_rb, x_ukf_rb, x_pf_rb, rmse_ekf_rb, rmse_ukf_rb, rmse_pf_rb)

if __name__ == "__main__":
    run_unified_demo()
