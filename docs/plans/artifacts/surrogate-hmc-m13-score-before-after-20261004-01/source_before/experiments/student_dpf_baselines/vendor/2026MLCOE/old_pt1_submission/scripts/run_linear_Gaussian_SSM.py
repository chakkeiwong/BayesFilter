
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tqdm import trange

# FORCE CPU USAGE (Fixes Metal plugin colocation errors on Mac M2)
tf.config.set_visible_devices([], 'GPU')

# IMPORT FROM YOUR MODULE
from filters import KF, DTYPE



# =============================================================================
# 1. LINEAR-GAUSSIAN SSM
# =============================================================================
class LinearGaussianSSM:
    """Standard LGSSM Definition using TensorFlow."""
    def __init__(self, dim_x=5, dim_y=5):
        self.dim_x, self.dim_y = dim_x, dim_y

        # Stable transition matrix 
        A_np = np.random.randn(dim_x, dim_x)
        max_eig = np.max(np.abs(np.linalg.eigvals(A_np)))
        if max_eig > 1.0: A_np /= (max_eig + 0.1)

        self.A = tf.constant(A_np, dtype=DTYPE)
        self.C = tf.random.normal((dim_y, dim_x), dtype=DTYPE)
        self.Q = tf.eye(dim_x, dtype=DTYPE) * 0.1
        self.R = tf.eye(dim_y, dtype=DTYPE) * 1.0
        self.x0_mean = tf.zeros(dim_x, dtype=DTYPE)
        self.x0_cov = tf.eye(dim_x, dtype=DTYPE) * 5.0

    @tf.function
    def generate(self, T=100):
        x_ta, y_ta = tf.TensorArray(DTYPE, size=T), tf.TensorArray(DTYPE, size=T)
        x_curr = self.x0_mean + tf.linalg.matvec(tf.linalg.cholesky(self.x0_cov), tf.random.normal([self.dim_x], dtype=DTYPE))
        L_Q, L_R = tf.linalg.cholesky(self.Q), tf.linalg.cholesky(self.R)
        for t in tf.range(T):
            x_curr = tf.linalg.matvec(self.A, x_curr) + tf.linalg.matvec(L_Q, tf.random.normal([self.dim_x], dtype=DTYPE))
            y_curr = tf.linalg.matvec(self.C, x_curr) + tf.linalg.matvec(L_R, tf.random.normal([self.dim_y], dtype=DTYPE))
            x_ta, y_ta = x_ta.write(t, x_curr), y_ta.write(t, y_curr)
        return x_ta.stack(), y_ta.stack()

# =============================================================================
# 2. CONDITIONAL OPTIMALITY EXPERIMENT
# =============================================================================
def run_optimality_validation(dim_x, dim_y, T, M=10000, title_suffix="Low-Dim", filename="kf_opt_val.png"):
    print(f"\n--- Optimality Validation (Conditional Analysis): {title_suffix} (n_x={dim_x}, M={M}) ---")
    model = LinearGaussianSSM(dim_x=dim_x, dim_y=dim_y)
    x_true, y_obs = model.generate(T=T)

    # Initialize KF
    kf = KF(F=model.A, H=model.C, Q=model.Q, R=model.R,
            P0=model.x0_cov, x0=tf.reshape(model.x0_mean, (-1, 1)))

    # Results Storage
    track_x_est, cond_history, mmse_discrepancy = [], [], []

    start_time = time.time()
    for t in range(T):
        # --- A. Store Previous State for MC Step ---
        m_prev = kf.x.read_value()
        P_prev = kf.P.read_value()

        # --- B. Kalman Filter Step ---
        kf.predict()
        y_t = tf.reshape(y_obs[t], (-1, 1))
        m_kf, P_kf = kf.update(y_t)

        # --- C. Conditional MC Posterior Covariance Calculation ---
        # 1. Sample realizations of x_{t-1} from the previous filtered posterior
        L_prev = tf.linalg.cholesky(P_prev + tf.eye(dim_x, dtype=DTYPE)*1e-9)
        particles_prev = tf.transpose(m_prev + tf.matmul(L_prev, tf.random.normal((dim_x, M), dtype=DTYPE)))

        # 2. Propagate particles through dynamics: x_t = A*x_{t-1} + w_t
        L_Q = tf.linalg.cholesky(model.Q)
        particles_pred = tf.matmul(particles_prev, model.A, transpose_b=True) + \
                         tf.matmul(tf.random.normal((M, dim_x), dtype=DTYPE), L_Q, transpose_b=True)

        # 3. Calculate Likelihood Weights: p(y_t | x_t)
        y_pred = tf.matmul(particles_pred, model.C, transpose_b=True)
        res = tf.reshape(y_obs[t], (1, -1)) - y_pred
        R_inv = tf.linalg.inv(model.R)
        log_lik = -0.5 * tf.reduce_sum(tf.matmul(res, R_inv) * res, axis=1)
        weights = tf.nn.softmax(log_lik)

        # 4. Compute Weighted Sample Covariance (The Empirical P_t|t)
        m_mc = tf.reduce_sum(particles_pred * tf.reshape(weights, (-1, 1)), axis=0)
        diff = particles_pred - m_mc
        P_mc = tf.matmul(tf.transpose(diff * tf.reshape(weights, (-1, 1))), diff)

        # --- D. Collect Metrics ---
        track_x_est.append(m_kf.numpy().flatten())
        cond_history.append(np.linalg.cond(P_kf.numpy()))
        mmse_discrepancy.append(np.linalg.norm(P_kf.numpy() - P_mc.numpy(), ord='fro'))

    # Calculate MSE
    track_x_est = np.array(track_x_est)
    path_mse = np.mean((track_x_est - x_true.numpy())**2)

    print(f"   Compute Time: {(time.time() - start_time):.2f} s")
    print(f"   Path RMSE: {np.sqrt(path_mse):.4f}")
    print(f"   Avg MMSE Discrepancy: {np.mean(mmse_discrepancy):.4e}")

    # PLOTTING
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. KF Tracking
    axes[0].plot(x_true.numpy()[:, 0], 'k-', alpha=0.5, label="True State")
    axes[0].plot(track_x_est[:, 0], 'r--', label=f"KF Estimate (MSE={path_mse:.4f})")
    axes[0].set_title(f"KF Tracking ({title_suffix})")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 2. Condition Number
    axes[1].plot(cond_history, 'b-')
    axes[1].set_yscale('log')
    axes[1].set_title(f"Covariance Condition Number ({title_suffix})")
    axes[1].set_ylabel("Condition Number (log scale)"); axes[1].grid(True, alpha=0.3)

    # 3. MMSE Comparison (Conditional Realizations)
    axes[2].plot(mmse_discrepancy, color='tab:green', label=r"$\|\mathbf{P}_{KF} - \mathbf{P}_{MC}\|_F$")
    axes[2].set_title(f"MMSE Validation: KF vs. Conditional MC (M={M})")
    axes[2].set_ylabel("Frobenius Norm of Difference"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(filename); plt.show()

if __name__ == "__main__":
    np.random.seed(41); tf.random.set_seed(41)
    run_optimality_validation(dim_x=5, dim_y=5, T=100, M=10000, title_suffix="Low-Dim", filename="kf_opt_low.png")
    run_optimality_validation(dim_x=200, dim_y=5, T=100, M=1000, title_suffix="High-Dim", filename="kf_opt_high.png")
