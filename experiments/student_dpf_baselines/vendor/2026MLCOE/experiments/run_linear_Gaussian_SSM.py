import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tqdm import trange

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.filters.classical import KF
from src.models.classic_ssm import LinearGaussianSSM, DTYPE
from src.utils.plotting import setup_plotting_style

# FORCE CPU USAGE
tf.config.set_visible_devices([], 'GPU')


def run_optimality_validation(dim_x, dim_y, T, M=10000, title_suffix="Low-Dim", filename="kf_opt_val.png"):
    print(f"\n--- Optimality Validation (Conditional Analysis): {title_suffix} (n_x={dim_x}, M={M}) ---")
    model = LinearGaussianSSM(dim_x=dim_x, dim_y=dim_y)
    x_true, y_obs = model.generate(T=T)

    # Modular KF initializes state and covariance in __init__
    kf = KF(
        F=model.F,
        H=model.H,
        Q=model.Q,
        R=model.R,
        P0=model.x0_cov,
        x0=tf.reshape(model.x0_mean, (-1, 1))
    )

    track_x_est, cond_history, mmse_discrepancy = [], [], []

    for t in trange(T, desc=f"Filtering {title_suffix}"):
        # A. Store Previous State for MC Step
        m_prev = kf.x.read_value()
        P_prev = kf.P.read_value()

        # B. Filter Step
        kf.predict()
        m_kf, P_kf = kf.update(tf.reshape(y_obs[t], (-1, 1)))

        # C. Conditional MC Posterior Covariance Calculation
        L_prev = tf.linalg.cholesky(P_prev + tf.eye(dim_x, dtype=DTYPE)*1e-9)
        particles_prev = tf.transpose(m_prev + tf.matmul(L_prev, tf.random.normal((dim_x, M), dtype=DTYPE)))

        particles_pred = tf.matmul(particles_prev, model.A, transpose_b=True) + \
                         tf.matmul(tf.random.normal((M, dim_x), dtype=DTYPE), tf.linalg.cholesky(model.Q), transpose_b=True)

        y_pred = tf.matmul(particles_pred, model.C, transpose_b=True)
        res = tf.reshape(y_obs[t], (1, -1)) - y_pred
        log_lik = -0.5 * tf.reduce_sum(tf.matmul(res, tf.linalg.inv(model.R)) * res, axis=1)
        weights = tf.nn.softmax(log_lik)

        m_mc = tf.reduce_sum(particles_pred * tf.reshape(weights, (-1, 1)), axis=0)
        diff = particles_pred - m_mc
        P_mc = tf.matmul(tf.transpose(diff * tf.reshape(weights, (-1, 1))), diff)

        # D. Collect Metrics
        track_x_est.append(m_kf.numpy().flatten())
        cond_history.append(np.linalg.cond(P_kf.numpy()))
        mmse_discrepancy.append(np.linalg.norm(P_kf.numpy() - P_mc.numpy(), ord='fro'))

    track_x_est = np.array(track_x_est)
    path_mse = np.mean((track_x_est - x_true.numpy())**2)

    print(f"    RMSE: {np.sqrt(path_mse):.4f} | Avg MMSE Error: {np.mean(mmse_discrepancy):.4e}")

    # Plotting exactly 3 subplots as per original script
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].plot(x_true.numpy()[:, 0], 'k-', alpha=0.5, label="True State")
    axes[0].plot(track_x_est[:, 0], 'r--', label=f"KF Estimate")
    axes[0].set_title(f"Tracking ({title_suffix})")
    axes[0].legend()

    axes[1].plot(cond_history, 'b-')
    axes[1].set_yscale('log')
    axes[1].set_title(f"Condition Number")

    axes[2].plot(mmse_discrepancy, color='tab:green', label="F-Norm Difference")
    axes[2].set_title("MMSE Validation: KF vs MC")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    setup_plotting_style()
    np.random.seed(41); tf.random.set_seed(41)

    # Run Low-Dim Validation
    run_optimality_validation(dim_x=5, dim_y=5, T=100, M=10000, title_suffix="Low-Dim")

    # Run High-Dim Validation
    run_optimality_validation(dim_x=200, dim_y=50, T=50, M=1000, title_suffix="High-Dim")
