
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# =============================================================================
# IMPORTS
# =============================================================================
from filters import (
    EKF,
    UKF,
    BPF,
    ESRF,
    GSMC,
    UPF,
    ParticleFlowFilter
)
from ssm_models import AcousticTrackingSSM, DTYPE



# =============================================================================
# 0. TENSORFLOW CONFIGURATION
# =============================================================================
#tf.config.set_visible_devices([], 'GPU') # Force CPU

### To run it on GPU on Apple silicon macs, force 32 bits
#tf.keras.backend.set_floatx('float32')

### High-precision CPU run
tf.config.set_visible_devices([], 'GPU') # Force CPU
tf.keras.backend.set_floatx('float64')

# =============================================================================
# 1. METRICS
# =============================================================================
def compute_omat(x_true, x_est):
    if isinstance(x_true, tf.Tensor): x_true = x_true.numpy()
    if isinstance(x_est, tf.Tensor): x_est = x_est.numpy()
    pos_true = x_true.reshape(4, 4)[:, :2]
    pos_est = x_est.reshape(4, 4)[:, :2]
    from scipy.spatial.distance import cdist as sp_cdist
    C = sp_cdist(pos_true, pos_est)
    row, col = linear_sum_assignment(C)
    return C[row, col].mean()

# =============================================================================
# 2. EXPERIMENTS
# =============================================================================

def run_experiment_fig1(model):
    """
    Replicates Figure 1 from Li (2017): Multi-target tracking trajectories.
    Demonstrates the PFF (LEDH) tracking 4 targets in a 2D sensor field.
    """
    print(f"--- Running Figure 1 Experiment (Tracking Trajectories) [TF-CPU] ---")

    # Configuration
    n_steps = 50
    N_particles = 100

    # 1. Specific Initial States (Li 2017 Section V-A-1)
    # [x, y, vx, vy] for 4 targets
    x0_np = np.array([
        12, 6, 0.001, 0.001,
        32, 32, -0.001, -0.005,
        20, 13, -0.1, 0.01,
        15, 35, 0.002, 0.002
    ])

    # Initial covariance (High uncertainty: 10^2 for pos, 1^2 for vel)
    P0_np = np.diag(np.tile([100, 100, 1, 1], 4))

    x0 = tf.constant(x0_np, dtype=DTYPE)
    P0 = tf.constant(P0_np, dtype=DTYPE)

    # 2. Generate Ground Truth & Observations
    x_true_np = np.zeros((n_steps, 16)); x_true_np[0] = x0_np
    curr_x = x0_np.copy()

    # Pre-compute Cholesky for noise generation
    Q_chol = np.linalg.cholesky(model.Q_true.numpy())
    R_chol = np.linalg.cholesky(model.R_true.numpy())
    F_np = model.F.numpy()

    for k in range(1, n_steps):
        # Process Update
        noise = Q_chol @ np.random.normal(size=16)
        curr_x = F_np @ curr_x + noise

        # Bounce off walls (0, 40) - Essential for this model's stability
        # Indices: 0=x, 1=y, 2=vx, 3=vy (repeated 4 times)
        for i in range(4):
            idx_x, idx_y, idx_vx, idx_vy = i*4, i*4+1, i*4+2, i*4+3

            # Check X bounds
            if not (0 <= curr_x[idx_x] <= 40):
                curr_x[idx_vx] *= -1
                curr_x[idx_x] = np.clip(curr_x[idx_x], 0, 40)

            # Check Y bounds
            if not (0 <= curr_x[idx_y] <= 40):
                curr_x[idx_vy] *= -1
                curr_x[idx_y] = np.clip(curr_x[idx_y], 0, 40)

        x_true_np[k] = curr_x

    # Generate Observations
    y_clean = model.h_func(tf.constant(x_true_np, dtype=DTYPE)).numpy()
    # Add measurement noise
    y_noise = (R_chol @ np.random.normal(size=(n_steps, 25)).T).T
    y_obs = y_clean + y_noise

    # 3. Run PF-PF (LEDH)
    # Li17 uses LEDH (Localized Exact Daum-Huang) flow
    pf = ParticleFlowFilter(model, N=N_particles, mode='ledh', is_pfpf=True)

    # Initialize filter at the Truth (as per Li17 Fig 1 setup)
    pf.init(x0, P0)

    est_traj = np.zeros((n_steps, 16))

    print(f"Running PF-PF (LEDH) on {model.n_targets} targets...")
    for t in range(n_steps):
        z_t = tf.constant(y_obs[t], dtype=DTYPE)
        est = pf.step(z_t)
        est_traj[t] = est.numpy()

        # --- Print RMSE at intervals ---
        if t % 5 == 0:
            # Using Euclidean norm (L2) as in the reference implementation
            err = np.linalg.norm(est_traj[t] - x_true_np[t])
            print(f"  Step {t}/{n_steps} | RMSE: {err:.4f}")

    # 4. Plotting
    plt.figure(figsize=(8, 8))

    # Plot Sensors
    sens = model.sensor_locs.numpy()
    plt.scatter(sens[:, 0], sens[:, 1], c='blue', marker='o', s=30, label='Sensors')

    colors = ['r', 'g', 'c', 'm']

    for c in range(model.n_targets):
        idx_x, idx_y = c*4, c*4+1

        # Plot True Trajectory
        plt.plot(x_true_np[:, idx_x], x_true_np[:, idx_y],
                 c=colors[c], linestyle='-', linewidth=2, label=f'Target {c+1} (True)')

        # Plot Estimated Trajectory
        plt.plot(est_traj[:, idx_x], est_traj[:, idx_y],
                 c=colors[c], linestyle=':', linewidth=3, label=f'Target {c+1} (Est.)')

        # Start marker
        plt.scatter(x_true_np[0, idx_x], x_true_np[0, idx_y], c='k', marker='x', s=100, zorder=10)

    plt.title("Multi-Target Acoustic Tracking (Replicating Li(17) Fig 1) [TF]")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.grid(True, alpha=0.3)

    # Deduplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout()

    # Save Figure
    plt.savefig('fig1_tracking_tf.png', dpi=300, bbox_inches='tight')
    print("Figure 1 saved as 'fig1_tracking_tf.png'")

    plt.show()


def run_experiment_fig2(model, n_trials=10, n_steps=40):
    print(f"--- Running Figure 2 Experiment (OMAT) ---")
    DATA_FILE = 'fig2_omat_data_tf.npy'

    # 1. DEFINE FILTERS
    filters = {
        'EKF': lambda: EKF(model),
        'UKF': lambda: UKF(model),
        'ESRF': lambda: ESRF(model, N=200),
        'GSMC': lambda: GSMC(model, N=200),
        'UPF': lambda: UPF(model, N=50),
        'BPF (100K)': lambda: BPF(model, N=100000),
        'EDH': lambda: ParticleFlowFilter(model, N=200, mode='edh', is_pfpf=False),
        'LEDH': lambda: ParticleFlowFilter(model, N=200, mode='ledh', is_pfpf=False),
        'PF-PF (EDH)': lambda: ParticleFlowFilter(model, N=200, mode='edh', is_pfpf=True),
        'PF-PF (LEDH)': lambda: ParticleFlowFilter(model, N=200, mode='ledh', is_pfpf=True)
    }

    # Check if we should load previous data or run new
    if False and os.path.exists(DATA_FILE):
        print(f"Loading results from {DATA_FILE}...")
        results = np.load(DATA_FILE, allow_pickle=True).item()
    else:
        results = {k: np.zeros((n_trials, n_steps)) for k in filters}

        # True initial state
        x0_np = np.array([12, 6, 0.001, 0.001, 32, 32, -0.001, -0.005, 20, 13, -0.1, 0.01, 15, 35, 0.002, 0.002])
        P0_np = np.diag(np.tile([100, 100, 1, 1], 4))

        x0 = tf.constant(x0_np, dtype=DTYPE)
        P0 = tf.constant(P0_np, dtype=DTYPE)

        for tr in range(n_trials):
            # --- A. Generate Ground Truth Trajectory ---
            x_true_np = np.zeros((n_steps, 16)); x_true_np[0] = x0_np
            curr_x = x0_np.copy()
            for k in range(1, n_steps):
                q_sample = np.random.multivariate_normal(np.zeros(16), model.Q_true.numpy())
                curr_x = model.F.numpy() @ curr_x + q_sample
                # Bounce off walls (0, 40)
                for i in range(4):
                    if not (0 <= curr_x[i*4] <= 40): curr_x[i*4+2] *= -1; curr_x[i*4] = np.clip(curr_x[i*4], 0, 40)
                    if not (0 <= curr_x[i*4+1] <= 40): curr_x[i*4+3] *= -1; curr_x[i*4+1] = np.clip(curr_x[i*4+1], 0, 40)
                x_true_np[k] = curr_x

            # --- B. Generate Noisy Observations ---
            y_clean = model.h_func(tf.constant(x_true_np, dtype=DTYPE)).numpy()
            y_obs = y_clean + np.random.multivariate_normal(np.zeros(25), model.R_true.numpy(), size=n_steps)

            # --- C. Generate Perturbed Initialization ---
            init_noise = np.random.multivariate_normal(np.zeros(16), P0_np)
            x_init_guess_np = x0_np + init_noise
            x_init_guess = tf.constant(x_init_guess_np, dtype=DTYPE)

            # --- D. Run Filters ---
            for name, factory in tqdm(filters.items(), desc=f"Trial {tr+1}"):
                f = factory()
                f.init(x_init_guess, P0)

                for t in range(n_steps):
                    z_t = tf.constant(y_obs[t], dtype=DTYPE)
                    est = f.step(z_t)
                    results[name][tr, t] = compute_omat(x_true_np[t], est)

        np.save(DATA_FILE, results)

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = {'EKF':'olive', 'UKF':'purple', 'UPF':'teal', 'ESRF':'green', 'GSMC':'lime', 'BPF (100K)':'gray', 'BPF (1M)':'orange', 'EDH':'magenta', 'LEDH':'cornflowerblue', 'PF-PF (EDH)':'red', 'PF-PF (LEDH)':'blue'}
    markers = {'EKF':'X', 'UKF':'d', 'UPF':'o', 'ESRF':'v', 'GSMC':'<', 'BPF (100K)':'^', 'BPF (1M)':'>', 'EDH':'*', 'LEDH':'D', 'PF-PF (EDH)':'^', 'PF-PF (LEDH)':'d'}

    time_steps = range(1, n_steps+1)
    for name in results:
        if np.any(results[name]):
            avg = np.mean(results[name], axis=0)
            lw = 2.5 if 'PF-PF' in name else 1.5
            plt.plot(time_steps, avg, label=name, color=colors.get(name,'k'), marker=markers.get(name,'o'), markevery=4, linewidth=lw, markersize=8)

    plt.ylim(0, 15)
    plt.xlim(0, 40); plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', ncol=3, fontsize='small'); plt.ylabel("Average OMAT Error (m)"); plt.xlabel("Time Step")
    plt.title(f"Replication of Li(17) Fig 2 ({n_trials} Trials) [TF]")
    plt.tight_layout()

    # --- SAVE FIGURE ---
    plt.savefig('fig2_replication_tf.png', dpi=300)
    print("Figure 2 saved as 'fig2_replication_tf.png'")

    # plt.show()

def run_experiment_fig4(model, n_trials=5, n_steps=40):
    print(f"--- Running Figure 4 Experiment (ESS vs Time) ---")
    DATA_FILE = 'fig4_ess_data_tf.npy'

    # Define filters with fixed N=500 (except BPF)
    filters = {
        'PF-PF (LEDH)': lambda: ParticleFlowFilter(model, N=500, mode='ledh', is_pfpf=True),
        'PF-PF (EDH)': lambda: ParticleFlowFilter(model, N=500, mode='edh', is_pfpf=True),
        'GSMC': lambda: GSMC(model, N=500),
        'BPF (100K)': lambda: BPF(model, N=100000),
    }

    if False and os.path.exists(DATA_FILE):
        print(f"Loading results from {DATA_FILE}...")
        results = np.load(DATA_FILE, allow_pickle=True).item()
    else:
        results = {k: np.zeros((n_trials, n_steps)) for k in filters}

        # True initial state
        x0_np = np.array([12, 6, 0.001, 0.001, 32, 32, -0.001, -0.005, 20, 13, -0.1, 0.01, 15, 35, 0.002, 0.002])
        P0_np = np.diag(np.tile([100, 100, 1, 1], 4))
        x0 = tf.constant(x0_np, dtype=DTYPE)
        P0 = tf.constant(P0_np, dtype=DTYPE)

        for tr in range(n_trials):
            # --- 1. Generate Ground Truth & Obs ---
            x_true_np = np.zeros((n_steps, 16)); x_true_np[0] = x0_np
            curr_x = x0_np.copy()
            for k in range(1, n_steps):
                q_sample = np.random.multivariate_normal(np.zeros(16), model.Q_true.numpy())
                curr_x = model.F.numpy() @ curr_x + q_sample
                # Bounce bounds
                for i in range(4):
                    if not (0 <= curr_x[i*4] <= 40): curr_x[i*4+2] *= -1; curr_x[i*4] = np.clip(curr_x[i*4], 0, 40)
                    if not (0 <= curr_x[i*4+1] <= 40): curr_x[i*4+3] *= -1; curr_x[i*4+1] = np.clip(curr_x[i*4+1], 0, 40)
                x_true_np[k] = curr_x

            y_clean = model.h_func(tf.constant(x_true_np, dtype=DTYPE)).numpy()
            y_obs = y_clean + np.random.multivariate_normal(np.zeros(25), model.R_true.numpy(), size=n_steps)

            # --- 2. Perturbed Initialization ---
            init_noise = np.random.multivariate_normal(np.zeros(16), P0_np)
            x_init_guess_np = x0_np + init_noise
            x_init_guess = tf.constant(x_init_guess_np, dtype=DTYPE)

            # --- 3. Run Filters ---
            for name, factory in tqdm(filters.items(), desc=f"Trial {tr+1}"):
                f = factory()
                f.init(x_init_guess, P0)

                for t in range(n_steps):
                    z_t = tf.constant(y_obs[t], dtype=DTYPE)
                    f.step(z_t)

                    # Store ESS (handle TF Variable/Tensor conversion)
                    if hasattr(f, 'ess'):
                        val = f.ess
                        if isinstance(val, (tf.Tensor, tf.Variable)):
                            results[name][tr, t] = val.numpy()
                        else:
                            results[name][tr, t] = val

        np.save(DATA_FILE, results)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    colors = {'PF-PF (LEDH)': 'blue', 'PF-PF (EDH)': 'red', 'LEDH': 'cornflowerblue', 'EDH': 'magenta', 'UPF': 'teal', 'GSMC': 'lime', 'BPF (100K)': 'darkblue', 'BPF (1M)': 'orange'}
    markers = {'PF-PF (LEDH)': 'd', 'PF-PF (EDH)': '^', 'LEDH': '>', 'EDH': '*', 'UPF': 'o', 'GSMC': '<', 'BPF (100K)': '^', 'BPF (1M)': '>'}

    time_steps = range(1, n_steps+1)
    for name in results:
        if np.any(results[name]):
            avg_ess = np.mean(results[name], axis=0)
            lw = 2.5 if 'PF-PF' in name else 1.5
            plt.plot(time_steps, avg_ess, label=name,
                     color=colors.get(name, 'k'),
                     marker=markers.get(name, 'o'),
                     markevery=4, linewidth=lw, markersize=8)

    plt.yscale('log')
    plt.ylim(0.8, 500)
    plt.xlim(0, 40)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend(loc='center right', ncol=2, fontsize='small')
    plt.ylabel("Average Effective Sample Size (ESS)")
    plt.xlabel("Time Step")
    plt.title(f"Replication of Li(17) Fig 4: Average ESS vs Time")
    plt.tight_layout()

    # --- SAVE FIGURE ---
    plt.savefig('fig4_ess_replication_tf.png', dpi=300)
    print("Figure 4 saved as 'fig4_ess_replication_tf.png'")

    # plt.show()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    model = AcousticTrackingSSM()
    #run_experiment_fig1(model)
    run_experiment_fig2(model, n_trials=10, n_steps=40)
    run_experiment_fig4(model, n_trials=10, n_steps=40)
