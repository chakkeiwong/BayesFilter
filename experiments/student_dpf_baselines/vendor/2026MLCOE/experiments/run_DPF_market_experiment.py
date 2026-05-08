"""
Market Dynamics Benchmark: Differentiable Particle Filters (DPF)
================================================================
This script evaluates the performance of Differentiable Particle Filters on
stochastic volatility tracking. It compares three core resampling strategies:
1. Neural Resampling (Particle Transformer)
2. Soft Resampling (Stochastic with gradient correction)
3. Entropic Optimal Transport (Sinkhorn/EOT) with varying epsilon.

The experiment benchmarks these methods across GRU and LSTM state-transition
models, measuring ELBO convergence and log-volatility tracking accuracy (RMSE).

Author: Joowon Lee
Date: 2026-02-27
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =============================================================================
# PROJECT CONFIGURATION & MODULAR IMPORTS
# =============================================================================

# Ensure project root is in path to discover the 'src' package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the DPF framework from the filters directory
from src.filters.DPF import DPF

# =============================================================================
# 1. MARKET MODELS (Transition & Likelihood)
# =============================================================================

class MarketDynamicsGRU(tf.keras.Model):
    def __init__(self, h_dim=32):
        super().__init__()
        self.cell, self.out, self.proc_log_std = tf.keras.layers.GRUCell(h_dim), tf.keras.layers.Dense(1), tf.Variable(-2.5, dtype=tf.float32)
    def call(self, x, h, n):
        o, [h_n] = self.cell(tf.reshape(x, (-1, 1)), h)
        return tf.squeeze(self.out(o)) + tf.exp(self.proc_log_std) * n, [h_n]

class MarketDynamicsLSTM(tf.keras.Model):
    def __init__(self, h_dim=32):
        super().__init__()
        self.cell, self.out, self.proc_log_std = tf.keras.layers.LSTMCell(h_dim), tf.keras.layers.Dense(1), tf.Variable(-2.5, dtype=tf.float32)
    def call(self, x, h, n):
        o, [h_n, c_n] = self.cell(tf.reshape(x, (-1, 1)), h)
        return tf.squeeze(self.out(o)) + tf.exp(self.proc_log_std) * n, [h_n, c_n]

class MarketLikelihood(tf.keras.Model):
    def call(self, x, y):
        x = tf.clip_by_value(x, -10.0, 5.0)
        return -0.5 * (np.log(2*np.pi) + x + tf.square(y) / (tf.exp(x) + 1e-6))

# =============================================================================
# 2. EXPERIMENT RUNNER
# =============================================================================

def run_experiment(num_trials=1, T=100, N=64, Epochs=60, load_existing=False):
    DATA_FILE, PLOT_FILE, H_DIM = 'market_dpf_long_horizon.npy', 'market_dpf_long_horizon.png', 32
    np.random.seed(42); true_x, obs_y, xt = [], [], np.float32(-2.0)
    for t in range(T):
        xt = 0.8 * xt + 0.6 * np.sin(0.2 * t) + np.float32(np.random.normal(0, 0.15))
        true_x.append(xt); obs_y.append(np.float32(np.random.normal(0, np.exp(xt/2))))
    observations = tf.cast(obs_y, tf.float32)

    # Expanded configurations for EOT with multiple epsilon values
    configs = []
    eot_eps_values = [0.01, 0.05, 0.1, 0.5]
    eot_colors = ['blue', 'purple', 'orange', 'brown']

    for m in ['GRU', 'LSTM']:
        # Particle Transformer
        configs.append({'method': 'transformer', 'model': m, 'label': f'{m}-Transformer', 'color': 'green', 'eps': 0.0})
        # Soft Resampling
        configs.append({'method': 'soft_orig', 'model': m, 'label': f'{m}-Soft', 'color': 'red', 'eps': 0.0})
        # EOT (Sinkhorn) variants
        for eps, color in zip(eot_eps_values, eot_colors):
            configs.append({
                'method': 'sinkhorn',
                'model': m,
                'label': f'{m}-EOT (eps={eps})',
                'color': color,
                'eps': eps
            })

    results = {}
    if load_existing and os.path.exists(DATA_FILE):
        print(f"Loading results from {DATA_FILE}...")
        results = np.load(DATA_FILE, allow_pickle=True).item()
    else:
        for cfg in configs:
            all_losses, all_ests, all_rmses = [], [], []
            print(f"Benchmarking {cfg['label']} across {num_trials} trials...")
            for trial in range(num_trials):
                f = MarketDynamicsGRU(H_DIM) if cfg['model'] == 'GRU' else MarketDynamicsLSTM(H_DIM)
                g, dpf_inst = MarketLikelihood(), DPF(f, MarketLikelihood(), N)
                optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(0.005, 20, 0.9))

                l_hist = []
                for epoch in range(Epochs):
                    with tf.GradientTape() as tape:
                        p, lw, loss = tf.random.normal((N,)) - 2.0, tf.fill((N,), -np.log(N, dtype='float32')), 0.0
                        h = [tf.zeros((N, H_DIM))] if cfg['model'] == 'GRU' else [tf.zeros((N, H_DIM)), tf.zeros((N, H_DIM))]
                        for t in range(T):
                            p, h, lw = dpf_inst.filter_step(p, h, lw, observations[t], method=cfg['method'], epsilon=cfg['eps'])
                            loss -= tf.reduce_logsumexp(g(p, observations[t]) + lw)
                    vars = list(f.trainable_variables) + list(dpf_inst.trainable_variables)
                    grads = tape.gradient(loss, vars)
                    grads, _ = tf.clip_by_global_norm(grads, 1.0)
                    optimizer.apply_gradients(zip(grads, vars))
                    l_hist.append(loss.numpy())
                    if epoch % 10 == 0: print(f" epoch={epoch}: loss={loss.numpy():.2f}")

                p, lw = tf.random.normal((N,)) - 2.0, tf.fill((N,), -np.log(N, dtype='float32'))
                h = [tf.zeros((N, H_DIM))] if cfg['model'] == 'GRU' else [tf.zeros((N, H_DIM)), tf.zeros((N, H_DIM))]
                t_ests = []
                for t in range(T):
                    p, h, lw = dpf_inst.filter_step(p, h, lw, observations[t], method=cfg['method'], epsilon=cfg['eps'])
                    t_ests.append(tf.reduce_sum(p * tf.exp(lw)).numpy())
                all_rmses.append(np.sqrt(np.mean(np.square(np.array(t_ests) - np.array(true_x))))); all_ests.append(t_ests); all_losses.append(l_hist)
            results[cfg['label']] = {'losses': all_losses, 'ests': all_ests, 'rmses': all_rmses, 'color': cfg['color'], 'ls': '-' if cfg['model'] == 'GRU' else '--'}
        np.save(DATA_FILE, results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for label, data in results.items():
        # ELBO Convergence Plot
        m_l, s_l = np.mean(data['losses'], axis=0), np.std(data['losses'], axis=0)
        ax1.plot(m_l, label=label, color=data['color'], ls=data['ls'])
        ax1.fill_between(range(Epochs), m_l-s_l, m_l+s_l, color=data['color'], alpha=0.1)

        # Volatility Tracking Plot
        m_e, s_e = np.mean(data['ests'], axis=0), np.std(data['ests'], axis=0)
        ax2.plot(m_e, label=f"{label} (RMSE: {np.mean(data['rmses']):.3f})", color=data['color'], ls=data['ls'])
        ax2.fill_between(range(T), m_e-s_e, m_e+s_e, color=data['color'], alpha=0.1)

    # Formatting for ax1 (ELBO)
    ax1.set_title("ELBO Convergence Comparison", fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (Negative ELBO)", fontsize=12)
    ax1.legend(fontsize='medium', ncol=2)

    # Formatting for ax2 (Tracking) - Enhanced True Signal
    ax2.plot(true_x, color='black', linewidth=3.5, label="True Signal", alpha=0.8, zorder=10)
    ax2.set_title("Volatility Tracking: Shaded Accuracy", fontsize=14)
    ax2.set_xlabel("Time Step (T)", fontsize=12)
    ax2.set_ylabel("Log-Volatility (x)", fontsize=12)
    ax2.legend(fontsize='medium', ncol=2)

    # Remove white space and save
    plt.tight_layout()
    plt.savefig(PLOT_FILE, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_experiment(num_trials=10, T=100, N=128, Epochs=60, load_existing=True)
