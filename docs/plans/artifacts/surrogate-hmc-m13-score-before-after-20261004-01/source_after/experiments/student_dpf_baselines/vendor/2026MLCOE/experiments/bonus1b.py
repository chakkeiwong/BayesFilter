"""
PHMC vs. PMMH Parameter Estimation Benchmark
============================================
Grid-search comparison between PHMC (Differentiable Particle Flow)
and PMMH (Bootstrap Particle Filter).

Author: Joowon Lee
Date: 2026-02-27
"""

import os
import sys
import time
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress TensorFlow C++ level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Modular imports according to the new file tree
from src.models.classic_ssm import NonlinearBenchmarkSSM
from src.filters.classical import DTYPE
from src.inference.phmc import hmc_pfpf
from src.inference.pmmh import pmmh_bpf
from src.filters.resampling.optimal_transport import SinkhornResampler

# Ensure global precision
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions

def logging_callback(i, state, pkr):
    """
    Real-time logging callback for PHMC.
    Forces a flush to ensure Iteration 000 appears immediately after compilation.
    """
    if i % 5 == 0:
        curr_val = state.numpy()[0]
        is_accepted = pkr.inner_results.is_accepted.numpy()
        log_acc = pkr.inner_results.log_accept_ratio.numpy()

        # tqdm.write handles the progress bar positioning
        tqdm.write(f" Iter {i:03d} | Sigma^2: {curr_val:.4f} | Acc: {is_accepted} | Energy Div: {-log_acc:.4f}")

        # Force the system buffer to empty immediately
        sys.stdout.flush()

if __name__ == "__main__":
    # 1. Global Configuration and Grid Setup
    results_path = 'mcmc_grid_results_restored.npy'
    true_v_sq_val = 10.0
    burn_in_pct = 0.2

    betas = [0.02, 0.05, 0.1, 1]
    step_sizes = [0.1, 0.2]
    leapfrogs = [1, 2]
    epsilons = [0.1, 0.5]

    def get_mcmc_diagnostics(samples, runtime, true_val):
        s = np.abs(samples).flatten()
        idx = int(len(s) * burn_in_pct)
        s_post = s[idx:]
        ess = tfp.mcmc.effective_sample_size(s_post).numpy()
        ess_per_sec = ess / runtime if runtime > 0 else 0
        mean_val = np.mean(s_post)
        mse = np.mean((s_post - true_val)**2)
        return ess, ess_per_sec, mean_val, mse, s

    # 2. Synchronized Data Generation
    print("Generating observations...")
    exp_model = NonlinearBenchmarkSSM(sigma_v_sq=true_v_sq_val)
    T, obs = 20, []
    # Explicit DTYPE casting to prevent float32/float64 mismatch
    true_x = [tfd.Normal(tf.cast(0, DTYPE), tf.sqrt(tf.cast(5.0, DTYPE))).sample((1,))]

    for i in range(1, T + 1):
        exp_model.current_n.assign(tf.cast(i, DTYPE))
        noise_std = tf.sqrt(tf.cast(true_v_sq_val, DTYPE))
        nx = exp_model.propagate(true_x[-1]) + tfd.Normal(tf.cast(0, DTYPE), noise_std).sample((1,))
        true_x.append(nx)

        obs_noise = tfd.Normal(tf.cast(0, DTYPE), tf.cast(1.0, DTYPE)).sample((1,))
        obs.append(exp_model.h_func(nx) + obs_noise)

    # 3. Handle PHMC grid search
    if os.path.exists(results_path):
        print(f"Loading existing MCMC data from {results_path}...")
        all_results = np.load(results_path, allow_pickle=True).item()
    else:
        print("No saved data found. Running full experiments...")
        all_results = {}
        num_samples = 200
        hmc_grid = list(itertools.product(betas, step_sizes, leapfrogs, epsilons))

        for b, s, L, e in hmc_grid:
            print(f"\n[PHMC] Config: Beta={b}, Step={s}, L={L}, Eps={e}")
            print(">>> Compiling XLA Graph (Initial tracing for Iter 000)...")
            sys.stdout.flush()

            # FIX: Initialize the Sinkhorn object with the grid epsilon
            current_resampler = SinkhornResampler(epsilon=e)

            start_t = time.time()
            # FIX: Pass resampler=current_resampler and remove eps=e
            h_samples, h_acc = hmc_pfpf(
                model=exp_model,
                observations=obs,
                num_samples=num_samples,
                beta=b,
                stepsize=s,
                num_leapfrog_steps=L,
                resampler=current_resampler,
                callback=logging_callback
            )

            runtime = time.time() - start_t
            config_key = f"b{b}_s{s}_L{L}_e{e}"
            all_results[config_key] = {
                'samples': h_samples, 'acc': h_acc, 'runtime': runtime,
                'beta': b, 's': s, 'L': L, 'eps': e
            }
        np.save(results_path, all_results)

    # 4. PMMH Baseline
    if 'pmmh' not in all_results:
        print("\nPMMH baseline missing. Running fresh PMMH-BPF...")
        num_samples_pmmh = 200
        start_t_p = time.time()
        p_samples, p_acc = pmmh_bpf(model=exp_model, observations=obs, num_iter=num_samples_pmmh)
        p_runtime = time.time() - start_t_p
        all_results['pmmh'] = {'samples': p_samples, 'acc': p_acc, 'runtime': p_runtime}
        np.save(results_path, all_results)

    # 5. Visualizations
    p_data = all_results['pmmh']
    p_ess, p_eps, p_mean, p_mse, p_flat = get_mcmc_diagnostics(p_data['samples'], p_data['runtime'], true_v_sq_val)
    pmmh_label = (f'PMMH-BPF Baseline | Acc: {p_data["acc"]:.1%}\n'
                  f'ESS: {p_ess:.1f}, ESS/s: {p_eps:.2f}, Time: {p_data["runtime"]:.1f}s')

    for current_beta in betas:
        print(f"\nPlotting results for Beta = {current_beta}...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
        plt.subplots_adjust(right=0.7)

        for key, data in all_results.items():
            if key == 'pmmh' or data.get('beta') != current_beta:
                continue

            ess, eps, mean, mse, s_flat = get_mcmc_diagnostics(data['samples'], data['runtime'], true_v_sq_val)
            label = (f"PHMC: $s$={data['s']}, $L$={data['L']}, $\\epsilon$={data['eps']} | "
                     f"Acc: {data['acc']:.1%}\nESS: {ess:.1f}, ESS/s: {eps:.2f}, Time: {data['runtime']:.1f}s")

            line, = ax1.plot(s_flat, label=label, alpha=0.7)
            ax2.hist(s_flat, bins=40, density=True, alpha=0.2, color=line.get_color(),
                     label=f"PHMC: $s$={data['s']} (Mean {mean:.2f}, ESS {ess:.1f})")

        ax1.plot(p_flat, label=pmmh_label, color='black', linewidth=2, linestyle=':')
        ax1.axhline(y=true_v_sq_val, color='red', linestyle='-', label=f'True Value ({true_v_sq_val})')
        ax1.set_title(f'MCMC Trace Comparison: $\\beta = {current_beta}$', fontsize=14)
        ax1.set_ylabel('$\sigma_v^2$')
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', framealpha=0.8)
        ax1.grid(True, alpha=0.3)

        ax2.axvline(x=true_v_sq_val, color='red', linestyle='-', label='True Value')
        ax2.hist(p_flat, bins=40, density=True, color='black', alpha=0.1, histtype='step',
                 linewidth=2, label=f'PMMH-BPF: Mean {p_mean:.2f}, ESS {p_ess:.1f}')
        ax2.set_title(f'Marginal Posterior Histograms: $\\beta = {current_beta}$', fontsize=14)
        ax2.set_xlabel('$\sigma_v^2$')
        ax2.set_ylabel('Density')
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', framealpha=0.8)
        ax2.grid(True, alpha=0.3)

        plt.savefig(f'mcmc_analysis_beta_{current_beta}.png', bbox_inches='tight')
        plt.show()
