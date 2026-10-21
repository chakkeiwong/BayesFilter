
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# IMPORT FILTERS
# =============================================================================
from filters import EDH, LEDH


# =============================================================================
# TENSORFLOW SETUP
# =============================================================================
tf.config.set_visible_devices([], 'GPU') # Force CPU
DTYPE = tf.float64

# =============================================================================
# 1. RANGE-BEARING MODEL (TensorFlow Wrapper)
# =============================================================================
class RangeBearingModelTF:
    """
    TensorFlow version of the Range-Bearing Model compatible with filters.py.
    """
    def __init__(self, dt=0.1, sigma_q=0.5, sigma_r=0.5, sigma_theta=0.1, omega=-0.05):
        self.dt = dt
        self.state_dim = 4
        self.obs_dim = 2
        self.omega = omega

        # Noise Covariances (TF Tensors)
        self.Q_np = np.eye(4) * sigma_q**2
        self.Q_filter = tf.constant(self.Q_np, dtype=DTYPE)

        self.R_np = np.diag([sigma_r**2, sigma_theta**2])
        self.R_filter = tf.constant(self.R_np, dtype=DTYPE)
        self.R_inv_filter = tf.linalg.inv(self.R_filter)

        # Transition Matrix F
        if abs(omega) > 1e-8:
            sin_w = np.sin(omega * dt); cos_w = np.cos(omega * dt)
            F_np = np.array([
                [1, 0, sin_w/omega, -(1-cos_w)/omega],
                [0, 1, (1-cos_w)/omega, sin_w/omega],
                [0, 0, cos_w, -sin_w], [0, 0, sin_w, cos_w]
            ])
        else:
            F_np = np.array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])

        self.F = tf.constant(F_np, dtype=DTYPE)

    def generate(self, T=200):
        """Simulate ground truth and observations using NumPy."""
        np.random.seed(42)
        x = np.zeros((T, 4)); y = np.zeros((T, 2))
        curr_x = np.array([0.0, 0.0, 4.0, 2.0])

        for t in range(T):
            # Process noise
            curr_x = self.F.numpy() @ curr_x + np.random.multivariate_normal(np.zeros(4), self.Q_np)
            # Measurement noise
            meas = self._measurement_fn_np(curr_x)
            curr_y = meas + np.random.multivariate_normal(np.zeros(2), self.R_np)

            x[t] = curr_x
            y[t] = curr_y
        return x, y

    def _measurement_fn_np(self, x):
        """NumPy measurement function for data generation."""
        return np.array([np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])])

    @tf.function
    def h_func(self, x):
        """
        TensorFlow vectorized measurement function.
        Expects x shape: (N, 4)
        """
        # Handle single vector case
        if len(x.shape) == 1:
            x = x[None, :]

        px = x[:, 0]
        py = x[:, 1]

        range_val = tf.sqrt(px**2 + py**2)
        bearing_val = tf.math.atan2(py, px)

        return tf.stack([range_val, bearing_val], axis=1)

    @tf.function
    def jacobian_h(self, x_bar):
        """
        Analytic Jacobian H for the linearized filters (EDH/LEDH).
        Expects x_bar shape: (4,)
        """
        px, py = x_bar[0], x_bar[1]
        r2 = px**2 + py**2
        r = tf.sqrt(r2)

        # Stability check
        safe_r = tf.maximum(r, 1e-4)
        safe_r2 = tf.maximum(r2, 1e-8)

        # H = [ [px/r, py/r, 0, 0], [-py/r^2, px/r^2, 0, 0] ]
        row1 = tf.stack([px/safe_r, py/safe_r, 0.0, 0.0])
        row2 = tf.stack([-py/safe_r2, px/safe_r2, 0.0, 0.0])

        return tf.stack([row1, row2])

# =============================================================================
# 2. EXPERIMENT RUNNER
# =============================================================================
def run_tf_comparison():
    # 1. Setup
    T = 100
    N = 100

    # Initialize TF Model
    model = RangeBearingModelTF()

    # Generate Data
    x_true_np, y_obs_np = model.generate(T=T)

    # Initial Conditions
    init_mean = tf.constant([0, 0, 4, 2], dtype=DTYPE)
    init_cov = tf.eye(4, dtype=DTYPE) * 2.0

    # 2. Initialize Filters from filters.py
    # EDH and LEDH handle prediction internally in their `step` method
    edh = EDH(model, N=N, steps=30)
    edh.init(init_mean, init_cov)

    ledh = LEDH(model, N=N, steps=30)
    ledh.init(init_mean, init_cov)

    # Storage
    est_edh = np.zeros((T, 4))
    est_ledh = np.zeros((T, 4))

    print("Running TensorFlow Range-Bearing Experiment...")

    for t in tqdm(range(T)):
        z_t = tf.constant(y_obs_np[t], dtype=DTYPE)

        # --- Run Filters ---
        est_edh[t] = edh.step(z_t).numpy()
        est_ledh[t] = ledh.step(z_t).numpy()

    # Calculate RMSE
    rmse_edh = np.sqrt(np.mean((est_edh[:, :2] - x_true_np[:, :2])**2))
    rmse_ledh = np.sqrt(np.mean((est_ledh[:, :2] - x_true_np[:, :2])**2))

    print(f"\nFinal RMSE:\nEDH: {rmse_edh:.2f}\nLEDH: {rmse_ledh:.2f}")

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(x_true_np[:, 0], x_true_np[:, 1], 'k-', linewidth=2, label='True Trajectory')
    plt.plot(est_edh[:, 0], est_edh[:, 1], 'b--', label=f'EDH (RMSE: {rmse_edh:.2f})')
    plt.plot(est_ledh[:, 0], est_ledh[:, 1], 'orange', linestyle=':', label=f'LEDH (RMSE: {rmse_ledh:.2f})')
    plt.scatter(0, 0, marker='x', color='r', s=100, label='Sensor Location')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("TensorFlow Range-Bearing Tracking (Imported Filters)")
    plt.axis('equal')

    # Save before show
    plt.savefig('range_bearing_tf_result.png', dpi=300)
    print("Figure saved as 'range_bearing_tf_result.png'")

    plt.show()

if __name__ == "__main__":
    run_tf_comparison()
