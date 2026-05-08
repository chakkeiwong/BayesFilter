"""
Multi-Target Acoustic Tracking Comparison
=========================================
Replication of Li (2017) and extension with Dai (2022) Optimized Homotopy.
Analyzes tracking trajectories, OMAT error, and ESS stability.

Author: Joowon Lee (UW-Madison Statistics)
Date: 2026-02-27
"""

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# PROJECT CONFIGURATION & MODULAR IMPORTS
# =============================================================================

# Ensure project root is in path for modular discovery
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Modular Imports from src
from src.filters.classical import KF, EKF, UKF, ESRF
from src.filters.particle import BPF, GSMC, UPF
from src.filters.flow_filters import ParticleFlowFilter
from src.models.classic_ssm import AcousticTrackingSSM, DTYPE

# =============================================================================
# 0. TENSORFLOW CONFIGURATION
# =============================================================================
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
    Replicates Figure 1 from Li (2017) with LEDH-OPT comparison.
    Adopts stable parameters from Fig 4 (mu=0.05, n_steps=100, N=200).
    Caches results to 'fig1_traj_data.npy'.
    """
    print(f"--- Running Figure 1 Experiment (Tracking Trajectories) ---")
    DATA_FILE = 'fig1_traj_data.npy'

    # Adopted stable parameters from Fig 2 and Fig 4
    n_steps_time = 50
    N_particles = 200  # Increased for stability
    MU_SETTING = 0.05  # Reduced stiffness penalty from Fig 4
    FLOW_STEPS = 100   # Increased pseudo-time resolution from Fig 4

    if False and os.path.exists(DATA_FILE):
        print(f"Loading cached trajectory data from {DATA_FILE}...")
        data = np.load(DATA_FILE, allow_pickle=True).item()
        x_true_np = data['x_true']
        est_traj_li = data['est_li']
        est_traj_opt = data['est_opt']
    else:
        # --- A. Initialization ---
        x0_np = np.array([12, 6, 0.001, 0.001, 32, 32, -0.001, -0.005, 20, 13, -0.1, 0.01, 15, 35, 0.002, 0.002])
        P0_np = np.diag(np.tile([100, 100, 1, 1], 4))
        x0, P0 = tf.constant(x0_np, dtype=DTYPE), tf.constant(P0_np, dtype=DTYPE)

        x_true_np = np.zeros((n_steps_time, 16)); x_true_np[0] = x0_np
        curr_x = x0_np.copy()
        Q_chol, R_chol, F_np = np.linalg.cholesky(model.Q_true.numpy()), np.linalg.cholesky(model.R_true.numpy()), model.F.numpy()

        # --- B. Generate Ground Truth & Observations ---
        for k in range(1, n_steps_time):
            curr_x = F_np @ curr_x + Q_chol @ np.random.normal(size=16)
            for i in range(4):
                idx_x, idx_y, idx_vx, idx_vy = i*4, i*4+1, i*4+2, i*4+3
                if not (0 <= curr_x[idx_x] <= 40): curr_x[idx_vx] *= -1; curr_x[idx_x] = np.clip(curr_x[idx_x], 0, 40)
                if not (0 <= curr_x[idx_y] <= 40): curr_x[idx_vy] *= -1; curr_x[idx_y] = np.clip(curr_x[idx_y], 0, 40)
            x_true_np[k] = curr_x

        y_obs = model.h_func(tf.constant(x_true_np, dtype=DTYPE)).numpy() + (R_chol @ np.random.normal(size=(n_steps_time, 25)).T).T

        # --- C. Run Filters ---
        pf_li = ParticleFlowFilter(model, N=N_particles, mode='ledh', is_pfpf=True)
        # Apply the mu and n_steps settings that worked for Fig 4
        pf_opt = ParticleFlowFilter(model, N=N_particles, mode='ledh_opt', is_pfpf=True, mu=MU_SETTING)
        pf_opt.n_steps = FLOW_STEPS

        pf_li.init(x0, P0); pf_opt.init(x0, P0)

        est_traj_li, est_traj_opt = np.zeros((n_steps_time, 16)), np.zeros((n_steps_time, 16))

        for t in tqdm(range(n_steps_time), desc="Fig 1 Tracking"):
            z_t = tf.constant(y_obs[t], dtype=DTYPE)
            est_traj_li[t] = pf_li.step(z_t).numpy()
            est_traj_opt[t] = pf_opt.step(z_t).numpy()

        save_dict = {'x_true': x_true_np, 'est_li': est_traj_li, 'est_opt': est_traj_opt}
        np.save(DATA_FILE, save_dict)
        print(f"Trajectory data saved to {DATA_FILE}")

    # --- D. Plotting ---
    plt.figure(figsize=(10, 8))
    sens = model.sensor_locs.numpy()
    plt.scatter(sens[:, 0], sens[:, 1], c='blue', marker='o', s=30, label='Sensors', alpha=0.5)
    colors = ['r', 'g', 'c', 'm']
    for c in range(model.n_targets):
        idx_x, idx_y = c*4, c*4+1
        plt.plot(x_true_np[:, idx_x], x_true_np[:, idx_y], c=colors[c], label=f'True T{c+1}')
        plt.plot(est_traj_li[:, idx_x], est_traj_li[:, idx_y], c=colors[c], linestyle='--', alpha=0.5, label=f'Li17 T{c+1}')
        plt.plot(est_traj_opt[:, idx_x], est_traj_opt[:, idx_y], c=colors[c], linestyle=':', linewidth=3, label=f'Dai22 T{c+1}')

    plt.title("Multi-Target Trajectories: Li(17) vs Dai(22) Optimized Schedule")
    plt.xlim(0, 40); plt.ylim(0, 40); plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    plt.savefig('fig1_dai_comparison.png', dpi=300, bbox_inches='tight')
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
        'PF-PF (LEDH)': lambda: ParticleFlowFilter(model, N=200, mode='ledh', is_pfpf=True),
        # Added LEDH-OPT method
        'PF-PF (LEDH-OPT)': lambda: ParticleFlowFilter(model, N=200, mode='ledh_opt', is_pfpf=True, mu=0.2)
    }

    # Check if we should load previous data or run new
    if os.path.exists(DATA_FILE):
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
    # Standard Li17 colors and markers
    colors = {'EKF':'olive', 'UKF':'purple', 'UPF':'teal', 'ESRF':'green', 'GSMC':'lime', 'BPF (100K)':'gray', 'BPF (1M)':'orange', 'EDH':'magenta', 'LEDH':'cornflowerblue', 'PF-PF (EDH)':'red', 'PF-PF (LEDH)':'blue', 'PF-PF (LEDH-OPT)':'darkred'}
    markers = {'EKF':'X', 'UKF':'d', 'UPF':'o', 'ESRF':'v', 'GSMC':'<', 'BPF (100K)':'^', 'BPF (1M)':'>', 'EDH':'*', 'LEDH':'D', 'PF-PF (EDH)':'^', 'PF-PF (LEDH)':'d', 'PF-PF (LEDH-OPT)':'p'}

    time_steps = range(1, n_steps+1)
    for name in results:
        if np.any(results[name]):
            avg = np.mean(results[name], axis=0)
            lw = 2.5 if 'PF-PF' in name else 1.5
            plt.plot(time_steps, avg, label=name, color=colors.get(name,'k'), marker=markers.get(name,'o'), markevery=4, linewidth=lw, markersize=8)

    plt.ylim(0, 40)
    plt.xlim(0, 40); plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', ncol=3, fontsize='small'); plt.ylabel("Average OMAT Error (m)"); plt.xlabel("Time Step")
    plt.title(f"Replication of Li(17) Fig 2 ({n_trials} Trials) [TF]")
    plt.tight_layout()

    # --- SAVE FIGURE ---
    plt.savefig('fig2_replication_tf.png', dpi=300)
    print("Figure 2 saved as 'fig2_replication_tf.png'")



def run_experiment_fig4(model, n_trials=5, n_steps=40):
    print(f"--- Running Figure 4 Experiment (ESS vs Time) ---")
    DATA_FILE = 'fig4_ess_data_tf.npy'

    # Define filters with the full set from your original code, including the new LEDH-OPT
    filters = {
            'EKF': lambda: EKF(model),
            'UKF': lambda: UKF(model),
            'ESRF': lambda: ESRF(model, N=200),
            'GSMC': lambda: GSMC(model, N=200),
            'UPF': lambda: UPF(model, N=50),
            'BPF (100K)': lambda: BPF(model, N=100000),
            'BPF (1M)': lambda: BPF(model, N=1000000),
            'EDH': lambda: ParticleFlowFilter(model, N=200, mode='edh', is_pfpf=False),
            'LEDH': lambda: ParticleFlowFilter(model, N=200, mode='ledh', is_pfpf=False),
            'PF-PF (EDH)': lambda: ParticleFlowFilter(model, N=200, mode='edh', is_pfpf=True),
            'PF-PF (LEDH)': lambda: ParticleFlowFilter(model, N=200, mode='ledh', is_pfpf=True),
            'PF-PF (LEDH-OPT)': lambda: (lambda f: (setattr(f, 'n_steps', 100), f)[1])(
                ParticleFlowFilter(model, N=200, mode='ledh_opt', is_pfpf=True, mu=0.05)
            )
        }

    if os.path.exists(DATA_FILE):
        print(f"Loading results from {DATA_FILE}...")
        results = np.load(DATA_FILE, allow_pickle=True).item()
    else:
        results = {k: np.zeros((n_trials, n_steps)) for k in filters}

        # Original true initial state and covariance
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
                # Bounce bounds logic from original code
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

                    # Store ESS (with proper deterministic handling)
                    if hasattr(f, 'ess'):
                        val = f.ess
                        if isinstance(val, (tf.Tensor, tf.Variable)):
                            results[name][tr, t] = val.numpy()
                        else:
                            results[name][tr, t] = val
                    else:
                        # Deterministic filters (EKF, UKF, LEDH) maintain full efficiency N
                        results[name][tr, t] = getattr(f, 'N', 200.0 if 'UKF' not in name and 'EKF' not in name else 1.0)

        np.save(DATA_FILE, results)

    plt.figure(figsize=(10, 6))
    colors = {'PF-PF (LEDH)': 'blue', 'PF-PF (LEDH-OPT)': 'red', 'PF-PF (EDH)': 'red', 'LEDH': 'cornflowerblue', 'EDH': 'magenta', 'UPF': 'teal', 'GSMC': 'lime', 'BPF (100K)': 'darkblue', 'BPF (1M)': 'orange'}
    markers = {'PF-PF (LEDH)': 'd', 'PF-PF (LEDH-OPT)': 's', 'PF-PF (EDH)': '^', 'LEDH': '>', 'EDH': '*', 'UPF': 'o', 'GSMC': '<', 'BPF (100K)': '^', 'BPF (1M)': '>'}


    for name in results:
        if name in colors:
            avg = np.mean(results[name], axis=0)
            plt.plot(range(1, n_steps+1), avg, label=name, color=colors[name], marker=markers[name], markevery=4, linewidth=2.5)

    plt.yscale('log'); plt.ylim(0.8, 500); plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend(loc='center right', ncol=2, fontsize='small'); plt.ylabel("Average ESS"); plt.xlabel("Time Step")
    plt.title(f"Replication of Li(17) Fig 4: ESS Stability with Optimized Homotopy (mu=0.05, steps=100)")
    plt.tight_layout(); plt.savefig('fig4_ess_replication_Dai22.png', dpi=300)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42); tf.random.set_seed(42)
    model = AcousticTrackingSSM()
    run_experiment_fig1(model)
    #run_experiment_fig2(model, n_trials=10, n_steps=40)
    #run_experiment_fig4(model, n_trials=10, n_steps=40)
