
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

# Import shared components
from filters import DTYPE, ExactDaumHuangPFF, LocalEDHFilter
from ssm_models import RangeBearingModel



# Set seeds
np.random.seed(43)
tf.random.set_seed(43)

# Enable eager execution
tf.config.run_functions_eagerly(True)

# ==========================================
# 1. Matrix Kernel PFF (TensorFlow Implementation)
# ==========================================
class MatrixKernelPFF_TF:
    def __init__(self, model, N=100):
        self.model = model
        self.N = N
        self.dim = model.state_dim
        # Flow parameters
        self.dt_flow = 0.05
        self.n_flow_steps = 50

        # State container
        self.X = tf.Variable(tf.zeros((self.N, self.dim), dtype=DTYPE))

    def init(self, x, P):
        """Initialize from Gaussian."""
        # x: (dim,), P: (dim, dim)
        # Using numpy random for init matching
        init_parts = np.random.multivariate_normal(x, P, self.N)
        self.X.assign(tf.cast(init_parts, DTYPE))

    @tf.function
    def step(self, z):
        # 1. Prediction (Process Noise)
        # N(0, Q)
        noise = tf.random.normal((self.N, self.dim), dtype=DTYPE)
        # Q is diagonal, simple scaling
        scale_Q = tf.sqrt(tf.linalg.diag_part(self.model.Q))
        noise = noise * scale_Q

        # X = F @ X^T + noise -> (N, 4)
        X_pred = tf.matmul(self.X, self.model.F_matrix, transpose_b=True) + noise
        self.X.assign(X_pred)

        # 2. Kernel Flow Loop
        z_tf = tf.reshape(tf.cast(z, DTYPE), [-1]) # (2,)

        for k in range(self.n_flow_steps):
            # A. Statistics for Matrix Kernel
            # Covariance of particles
            mean_p = tf.reduce_mean(self.X, axis=0)
            X_centered = self.X - mean_p
            cov_p = tf.matmul(X_centered, X_centered, transpose_a=True) / (self.N - 1.0)
            cov_p = cov_p + tf.eye(self.dim, dtype=DTYPE) * 1e-5

            # Precision and Cholesky (Whitening)
            try:
                prec_p = tf.linalg.inv(cov_p)
                L = tf.linalg.cholesky(prec_p) # L @ L.T = prec_p
            except:
                prec_p = tf.eye(self.dim, dtype=DTYPE)
                L = tf.eye(self.dim, dtype=DTYPE)

            # B. Gradients (Log-Posterior)
            # grad_log_p = grad_lik + grad_prior
            # We compute this for all particles efficiently

            # 1. Grad Likelihood: H^T @ R^-1 @ (z - h(x))
            # Jacobian H for all particles
            # Map over particles to get H_i: (N, 2, 4)
            H_all = tf.map_fn(self.model.H_jacobian, self.X)

            # Residuals: (N, 2)
            preds = self.model._measurement_fn(self.X)
            # Use model.residual for wrapping logic
            # z_tf broadcast against preds rows?
            # model.residual handles (2,) vs (N, 2)
            innov = tf.map_fn(lambda p: self.model.residual(z_tf, p), preds)

            # H.T @ R^-1 @ innov
            # (N, 4, 2) @ (2, 2) @ (N, 2, 1) -> (N, 4, 1)
            # Expand R_inv for batch
            R_inv_batch = tf.expand_dims(self.model.inv_R, 0)
            term_lik = tf.matmul(H_all, R_inv_batch, transpose_a=True) # (N, 4, 2)
            grad_lik = tf.matmul(term_lik, tf.expand_dims(innov, -1)) # (N, 4, 1)
            grad_lik = tf.squeeze(grad_lik, -1) # (N, 4)

            # 2. Grad Prior: -Sigma^-1 @ (x - mu)
            # Using current ensemble stats as proxy for flow prior
            grad_prior = -tf.matmul(self.X - mean_p, prec_p) # (N, 4)

            grad_log_p = grad_lik + grad_prior

            # C. Matrix-Valued Kernel
            # Whitened diffs: Z = X @ L
            X_w = tf.matmul(self.X, L)
            # Pairwise squared dists in whitened space: ||Z_i - Z_j||^2
            # (N, 1, D) - (1, N, D)
            diffs_w = tf.expand_dims(X_w, 1) - tf.expand_dims(X_w, 0) # (N, N, D)
            dists_sq = tf.reduce_sum(tf.square(diffs_w), axis=2) # (N, N)

            # Median Heuristic
            # Get upper triangle to avoid zeros/dupes
            # Flatten and compute median
            flat_dists = tf.reshape(dists_sq, [-1])
            med = tfp.stats.percentile(flat_dists, 50.0)

            h = med / tf.math.log(float(self.N) + 1.0)
            # Guard against zero h
            h = tf.maximum(h, 1e-6)

            K = tf.exp(-0.5 * dists_sq / h) # (N, N)

            # D. Flow (SVGD with Matrix Kernel)
            # 1. Driving Force: K @ grad_log_p
            term1 = tf.matmul(K, grad_log_p) # (N, 4)

            # 2. Repulsive Force (Matrix Kernel Deriv)
            # grad_x K(xj, x) = K * (-1/h) * Sigma^-1 * (x - xj)
            # Sum over j of: K_ij * (X_i - X_j) @ prec_p

            # Real diffs: (N, N, D)
            diffs_real = tf.expand_dims(self.X, 1) - tf.expand_dims(self.X, 0)

            # Project diffs by precision: (N, N, D) @ (D, D) -> (N, N, D)
            # Use tensordot or reshape/matmul
            # Reshape diffs to (N*N, D)
            diffs_flat = tf.reshape(diffs_real, [-1, self.dim])
            diffs_proj_flat = tf.matmul(diffs_flat, prec_p)
            diffs_proj = tf.reshape(diffs_proj_flat, [self.N, self.N, self.dim])

            # Weight by K: K_ij * diff_proj_ij
            # K is (N, N), expand to (N, N, 1)
            K_exp = tf.expand_dims(K, -1)
            weighted_repulsion = K_exp * diffs_proj # (N, N, D)

            # Sum over j (axis 1)
            term2_sum = tf.reduce_sum(weighted_repulsion, axis=1) # (N, D)
            term2 = term2_sum * (1.0 / h)

            phi = (term1 + term2) / float(self.N)

            # E. Clipping
            grad_norm = tf.norm(phi, axis=1, keepdims=True)
            scale = tf.minimum(1.0, 5.0 / (grad_norm + 1e-8))

            self.X.assign_add(self.dt_flow * phi * scale)

        return tf.reduce_mean(self.X, axis=0)

# ==========================================
# 2. Analytic PFF Wrappers (Using existing filters.py classes)
# ==========================================
# We wrap them to match the .step(z) interface and ensure correct init
class AnalyticWrapper:
    def __init__(self, cls, model, N, **kwargs):
        # Instantiate the TF class
        # EDH/LEDH in filters.py expect: (num_particles, f_fn, h_fn, Q, R, P0, ...)
        # We need P0 for init. We'll pass identity for now and re-init properly in .init()
        dummy_P0 = tf.eye(model.state_dim, dtype=DTYPE)
        self.f = cls(N, model.f, model.h, model.Q, model.R_filter, dummy_P0, **kwargs)
        self.model = model

    def init(self, x, P):
        # Initialize particles
        self.f.initialize(x, P)
        # EDH needs explicit P0 update usually, or it tracks internally?
        # filters.py EDH tracks self.P. Let's update it.
        if hasattr(self.f, 'P'):
            self.f.P = tf.cast(P, DTYPE)

    def step(self, z):
        # Predict
        self.f.predict()
        # Update
        z_tf = tf.reshape(z, [-1, 1]) # (2, 1)
        est = self.f.update(z_tf)
        return est.numpy().flatten()

# ==========================================
# 3. Main Comparison Runner
# ==========================================
def run_comparison():
    print("Running Matrix Kernel PFF Comparison...")

    T = 100
    # Use ssm_models.RangeBearingModel (TF version)
    model = RangeBearingModel()

    # Generate Data
    x_true_tf, y_obs_tf = model.generate(T=T)
    x_true = x_true_tf.numpy()
    y_obs = y_obs_tf.numpy()

    N = 100
    init_mean = np.array([0, 0, 4, 2])
    init_cov = np.eye(4) * 2.0

    # Initialize Filters
    # 1. EDH (Analytic)
    edh = AnalyticWrapper(ExactDaumHuangPFF, model, N, steps=30)
    edh.init(init_mean, init_cov)

    # 2. LEDH (Analytic)
    ledh = AnalyticWrapper(LocalEDHFilter, model, N, steps=30, k_neighbors=20)
    ledh.init(init_mean, init_cov)

    # 3. Matrix KPFF (New Class)
    kpff = MatrixKernelPFF_TF(model, N)
    kpff.init(init_mean, init_cov)

    # Results
    est_edh = np.zeros((T, 4))
    est_ledh = np.zeros((T, 4))
    est_kpff = np.zeros((T, 4))

    for t in tqdm(range(T)):
        z = y_obs[t]
        est_edh[t] = edh.step(z)
        est_ledh[t] = ledh.step(z)
        est_kpff[t] = kpff.step(z).numpy()

    # Metrics
    # Pos RMSE (indices 0, 1)
    rmse_edh = np.sqrt(np.mean((est_edh[:, :2] - x_true[:, :2])**2))
    rmse_ledh = np.sqrt(np.mean((est_ledh[:, :2] - x_true[:, :2])**2))
    rmse_kpff = np.sqrt(np.mean((est_kpff[:, :2] - x_true[:, :2])**2))

    print(f"\nFinal RMSE:\nEDH: {rmse_edh:.2f}\nLEDH: {rmse_ledh:.2f}\nKPFF: {rmse_kpff:.2f}")

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(x_true[:, 0], x_true[:, 1], 'k-', linewidth=2, label='True')
    plt.plot(est_edh[:, 0], est_edh[:, 1], 'b--', label=f'EDH ({rmse_edh:.2f})')
    plt.plot(est_ledh[:, 0], est_ledh[:, 1], 'orange', linestyle=':', label=f'LEDH ({rmse_ledh:.2f})')
    plt.plot(est_kpff[:, 0], est_kpff[:, 1], 'g-', linewidth=2, label=f'Matrix KPFF ({rmse_kpff:.2f})')

    plt.scatter(0, 0, marker='x', color='r', s=100, label='Sensor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Tracking Performance: Matrix Kernel PFF vs Baselines")
    plt.axis('equal')

    plt.savefig('matrix_kpff_comparison.png')
    print(">> Plot saved to 'matrix_kpff_comparison.png'")
    plt.show()

if __name__ == "__main__":
    run_comparison()
