import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import existing model and constants
from src.models.classic_ssm import StochasticVolatilityModel
from src.filters.classical import DTYPE

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable eager execution for the manual loop and plotting
tf.config.run_functions_eagerly(True)

def run_degeneracy_viz():
    print("Running Degeneracy Visualization (TensorFlow)...")

    # 1. Setup Simulation
    N = 200  # Number of particles
    T = 2000 # Time steps

    # Initialize existing model
    model = StochasticVolatilityModel(alpha=0.91, sigma_v=1.0, beta=0.5)

    # Generate Data
    _, y_obs_tf = model.generate(T=T)

    # Initialize Filter State
    particles = tf.random.normal((N, 1), stddev=1.0, dtype=DTYPE)
    # Ensure weights start as 1D vector (N,)
    weights = tf.ones(N, dtype=DTYPE) / N

    # History containers
    history_weights = []
    history_particles = []
    ess_history = []

    # 2. Run Filter Loop
    for t in range(T):
        # A. Propagate
        noise = tf.random.normal((N, 1), stddev=model.sigma_v, dtype=DTYPE)
        particles = model.alpha * particles + noise

        # B. Weight
        y_t = tf.reshape(y_obs_tf[t], [1, 1])
        log_w = model.log_likelihood(y_t, particles)

        log_w = tf.reshape(log_w, [-1])

        # Robust Softmax
        max_log_w = tf.reduce_max(log_w)
        w_un_norm = tf.math.exp(log_w - max_log_w)

        # Update weights
        weights = weights * w_un_norm
        weights /= tf.reduce_sum(weights)

        # Store histories
        history_weights.append(weights.numpy().flatten())
        history_particles.append(particles.numpy().flatten())

        # C. ESS Calculation
        ess = 1.0 / tf.reduce_sum(tf.square(weights))
        ess_history.append(ess.numpy())

        # D. Resample
        if ess < (N / 2.0):
            logits = tf.math.log(weights + 1e-300)
            indices = tf.random.categorical(tf.reshape(logits, [1, N]), N)[0]
            particles = tf.gather(particles, indices)
            weights = tf.ones(N, dtype=DTYPE) / N

    # 3. Find step with MAXIMUM Degeneracy
    worst_step_idx = np.argmin(ess_history)
    worst_weights = history_weights[worst_step_idx]
    worst_particles = history_particles[worst_step_idx] # particle location at the worst case

    print(f"Worst Step Index: {worst_step_idx}")
    print(f"Minimum ESS: {ess_history[worst_step_idx]:.2f} / {N}")

    # ==========================================
    # 4. Professional Visualization (1x3 Subplots)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.3)

    # [Plot 1] ESS over Time
    axes[0].plot(ess_history[:500], color='tab:blue', linewidth=1.5)
    axes[0].axhline(N / 2.0, color='red', linestyle='--', label='Resampling Threshold')
    axes[0].set_title("Effective Sample Size (ESS) over Time", fontsize=12)
    axes[0].set_xlabel("Time Step (t)", fontsize=11)
    axes[0].set_ylabel("ESS", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # [Plot 2] Histogram of Weights
    # Y-axis: log scale
    axes[1].hist(worst_weights, bins=30, color='tab:orange', edgecolor='black')
    axes[1].set_yscale('log') # [중요] 199개가 0에 있어서 로그 스케일 필수
    axes[1].set_title(f"Weight Distribution at t={worst_step_idx}", fontsize=12)
    axes[1].set_xlabel("Particle Weight", fontsize=11)
    axes[1].set_ylabel("Frequency (Log Scale)", fontsize=11)
    axes[1].grid(True, alpha=0.4)

    # [Plot 3] Weight vs. Particle State
    axes[2].scatter(worst_particles, worst_weights, color='tab:green', alpha=0.6, edgecolors='k')
    axes[2].set_title(f"Weights vs. Particle State at t={worst_step_idx}", fontsize=12)
    axes[2].set_xlabel("Particle State (Volatility Value)", fontsize=11)
    axes[2].set_ylabel("Weight", fontsize=11)

    # Highlight the particle with highest weight
    max_idx = np.argmax(worst_weights)
    axes[2].scatter(worst_particles[max_idx], worst_weights[max_idx], color='red', s=100, label='Dominant Particle', zorder=5)
    axes[2].legend()
    axes[2].grid(True, alpha=0.4)

    plt.suptitle("Analysis of Particle Degeneracy in SMC", fontsize=16, y=1.05)
    plt.savefig('degeneracy_professional_viz.png', bbox_inches='tight')
    print(">> Plot saved to 'degeneracy_professional_viz.png'")
    plt.show()

if __name__ == "__main__":
    run_degeneracy_viz()
