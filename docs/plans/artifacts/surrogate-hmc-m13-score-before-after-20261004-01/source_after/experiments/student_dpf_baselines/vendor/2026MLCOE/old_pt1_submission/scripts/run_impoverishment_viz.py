
"""
Author: Joowon Lee (UW-Madison Statistics, ljw9510@gmail.com)
Date: 2026-01-01
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import DTYPE for consistency
from filters import DTYPE



# Set seeds
np.random.seed(41)
tf.random.set_seed(41)

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Define Model Locally to ensure specific low-noise parameters for this demo
# (The one in ssm_models.py might have different default noise)
class RangeBearingModelImpoverishment:
    def __init__(self, dt=0.1, sigma_r=0.1, sigma_theta=0.01):
        # NOTE: Very low measurement noise (sigma_r=0.1) forces peaked likelihoods
        self.dt = dt
        self.state_dim = 4
        # Process Noise (Small enough that particles don't naturally spread out much)
        self.Q = tf.eye(4, dtype=DTYPE) * (0.05**2)

        self.R_diag = tf.constant([sigma_r**2, sigma_theta**2], dtype=DTYPE)
        self.R = tf.linalg.diag(self.R_diag)
        self.inv_R = tf.linalg.inv(self.R)

        self.F = tf.constant([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]
        ], dtype=DTYPE)

    def generate(self, T=100):
        x_list = []
        y_list = []
        # Linear diagonal path
        curr_x = tf.constant([0, 0, 2.0, 1.0], dtype=DTYPE)
        curr_x = tf.reshape(curr_x, (4, 1))

        for t in range(T):
            # Dynamics
            noise_proc = tf.random.normal((4, 1), dtype=DTYPE) * 0.05
            curr_x = tf.matmul(self.F, curr_x) + noise_proc

            # Measurement
            px, py = curr_x[0,0], curr_x[1,0]
            r = tf.sqrt(px**2 + py**2)
            theta = tf.math.atan2(py, px)
            curr_y_clean = tf.stack([r, theta])

            # Add measurement noise
            noise_meas = tf.random.normal((2,), dtype=DTYPE) * tf.sqrt(self.R_diag)
            curr_y = curr_y_clean + noise_meas

            x_list.append(curr_x[:,0])
            y_list.append(curr_y)

        return tf.stack(x_list), tf.stack(y_list)

    def log_likelihood(self, y, particles):
        # particles: (N, 4)
        # y: (2,)

        # Predict
        px = particles[:, 0]
        py = particles[:, 1]
        r_pred = tf.sqrt(px**2 + py**2)
        th_pred = tf.math.atan2(py, px)

        # Residuals
        res_r = y[0] - r_pred
        # Wrap angle
        raw_diff_th = y[1] - th_pred
        res_th = tf.math.floormod(raw_diff_th + np.pi, 2 * np.pi) - np.pi

        residuals = tf.stack([res_r, res_th], axis=1) # (N, 2)

        # -0.5 * sum( (res @ inv_R) * res )
        term = tf.matmul(residuals, self.inv_R)
        mahalanobis = tf.reduce_sum(term * residuals, axis=1)

        return -0.5 * mahalanobis

def run_impoverishment_demo():
    print("Running Sample Impoverishment Demo (TensorFlow)...")

    # Setup
    T = 60
    N = 50 # Small N exaggerates the effect

    # Use local model for specific noise settings
    model = RangeBearingModelImpoverishment(sigma_r=0.1)
    x_true_tf, y_obs_tf = model.generate(T)
    x_true = x_true_tf.numpy()

    # Init Particles
    # N(mean=[0,0,2,1], cov=I)
    particles = tf.random.normal((N, 4), dtype=DTYPE) + tf.constant([[0,0,2,1]], dtype=DTYPE)
    weights = tf.ones(N, dtype=DTYPE) / N

    # Store history: [Time, Particle_Idx, State_Dim]
    # We'll collect list of (N, 2) arrays then stack
    history_list = []
    unique_particles_count = []

    # Run Loop
    for t in range(T):
        # 1. Propagate (No extra jitter beyond process noise to show raw resampling effect)
        # Process noise: N(0, Q)
        noise = tf.random.normal((N, 4), dtype=DTYPE)
        # Q is diagonal 0.05^2, so chol is just * 0.05
        noise = noise * 0.05

        particles = tf.matmul(particles, model.F, transpose_b=True) + noise

        # 2. Weight
        log_w = model.log_likelihood(y_obs_tf[t], particles)

        # Robust update
        max_log_w = tf.reduce_max(log_w)
        w_un_norm = tf.math.exp(log_w - max_log_w)
        weights = weights * w_un_norm
        weights /= tf.reduce_sum(weights)

        # Save state BEFORE resampling (to see diversity before it's killed)
        history_list.append(particles.numpy()[:, :2])

        # 3. Resample (Systematic)
        # Resample often to force effect
        ess = 1.0 / tf.reduce_sum(tf.square(weights))

        if ess < (N / 1.5):
            # Systematic Resampling Logic
            cdf = tf.cumsum(weights)
            u_start = tf.random.uniform([], 0, 1.0/N, dtype=DTYPE)
            u = u_start + tf.cast(tf.range(N), DTYPE) / N
            indices = tf.searchsorted(cdf, u)

            particles = tf.gather(particles, indices)
            weights = tf.ones(N, dtype=DTYPE) / N

        # Count unique particles (Diversity Metric)
        # Count unique X positions as proxy
        # TF equivalent of np.unique is a bit verbose, convert to numpy for counting
        unique_count = len(np.unique(particles.numpy()[:, 0]))
        unique_particles_count.append(unique_count)

    history = np.array(history_list) # (T, N, 2)

    # ==========================================
    # 3. Visualization
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: 2D Trajectories ---
    ax1.set_title(f"Sample Impoverishment (N={N})")
    ax1.plot(x_true[:, 0], x_true[:, 1], 'k-', linewidth=3, label="True Trajectory")

    # Plot faint lines for EVERY particle
    for i in range(N):
        ax1.plot(history[:, i, 0], history[:, i, 1], color='red', alpha=0.3, linewidth=0.8)

    ax1.scatter(0, 0, marker='x', color='g', s=100, label="Sensor")
    ax1.set_xlabel("X Position"); ax1.set_ylabel("Y Position")
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Loss of Diversity ---
    ax2.set_title("Loss of Particle Diversity")
    ax2.plot(unique_particles_count, 'b-o', linewidth=2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Count of Unique Particles")
    ax2.set_ylim(0, N+5)
    ax2.axhline(1, color='r', linestyle='--', label="Total Collapse")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('impoverishment_viz.png')
    print(">> Plot saved to 'impoverishment_viz.png'")
    plt.show()

if __name__ == "__main__":
    run_impoverishment_demo()
