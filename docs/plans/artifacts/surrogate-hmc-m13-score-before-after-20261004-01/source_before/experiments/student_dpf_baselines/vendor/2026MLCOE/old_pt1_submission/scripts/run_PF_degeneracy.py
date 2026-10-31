import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import existing model and constants
from ssm_models import StochasticVolatilityModel
from filters import DTYPE

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
    ess_history = []

    # 2. Run Filter Loop
    for t in range(T):
        # A. Propagate
        noise = tf.random.normal((N, 1), stddev=model.sigma_v, dtype=DTYPE)
        particles = model.alpha * particles + noise

        # B. Weight
        y_t = tf.reshape(y_obs_tf[t], [1, 1])
        log_w = model.log_likelihood(y_t, particles)

        # --- FIX: FORCE 1D SHAPE ---
        # This prevents log_w from being (1, N) and broadcasting weights to (1, N)
        log_w = tf.reshape(log_w, [-1])

        # Robust Softmax
        max_log_w = tf.reduce_max(log_w)
        w_un_norm = tf.math.exp(log_w - max_log_w)

        # Update weights
        weights = weights * w_un_norm
        weights /= tf.reduce_sum(weights)

        # Store weights (Force flatten to be safe)
        history_weights.append(weights.numpy().flatten())

        # C. ESS Calculation
        ess = 1.0 / tf.reduce_sum(tf.square(weights))
        ess_history.append(ess.numpy())

        # D. Resample
        if ess < (N / 2.0):
            logits = tf.math.log(weights + 1e-300)
            # Categorical expects (Batch, Classes), so we reshape logits to (1, N)
            indices = tf.random.categorical(tf.reshape(logits, [1, N]), N)[0]

            particles = tf.gather(particles, indices)
            weights = tf.ones(N, dtype=DTYPE) / N

    # 3. Find step with MAXIMUM Degeneracy
    worst_step_idx = np.argmin(ess_history)
    worst_weights = history_weights[worst_step_idx]

    print(f"Worst Step Index: {worst_step_idx}")
    print(f"Minimum ESS: {ess_history[worst_step_idx]:.2f} / {N}")

    # 4. Visualization
    plt.figure(figsize=(10, 6))

    # Style
    plt.rcParams['axes.facecolor'] = '#EAEAF2'
    plt.grid(color='white', linestyle='-', linewidth=1.5)

    # Plot "Sticks"
    indices = np.arange(N)
    plt.vlines(indices, 0, worst_weights, colors='k', linewidth=2.5, alpha=0.6)

    # Highlight Dominant Particle
    max_idx = np.argmax(worst_weights)
    plt.plot(max_idx, worst_weights[max_idx], 'o', color='lime', markersize=12, markeredgecolor='k', zorder=10, label='Dominant')

    # Highlight Depleted Particle
    min_idx = np.argmin(worst_weights)
    plt.plot(min_idx, worst_weights[min_idx], 'o', color='red', markersize=10, markeredgecolor='k', zorder=10, label='Depleted')

    # Labels
    plt.xlabel("Simulations (Particle Index)", fontsize=12)
    plt.ylabel("Weights", fontsize=12)
    plt.title(f"Weight Degeneracy at t={worst_step_idx} (ESS={ess_history[worst_step_idx]:.2f})", fontsize=14)

    plt.xlim(0, N)
    plt.ylim(0, np.max(worst_weights) * 1.15)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend()
    plt.tight_layout()
    plt.savefig('degeneracy_viz.png')
    print(">> Plot saved to 'degeneracy_viz.png'")
    plt.show()

if __name__ == "__main__":
    run_degeneracy_viz()
