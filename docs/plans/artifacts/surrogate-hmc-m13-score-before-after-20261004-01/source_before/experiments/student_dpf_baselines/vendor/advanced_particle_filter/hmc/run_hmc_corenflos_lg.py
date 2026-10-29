"""
HMC entry points for the Corenflos linear-Gaussian SSM.

Provides two entry points:

  run_hmc_corenflos_lg_dpf:
        DPF-based HMC. resampler in {'soft', 'sinkhorn', 'amortized'}.
        Mirrors hmc/run_hmc_poc.main.

  run_hmc_corenflos_lg_kalman:
        Kalman-exact HMC. Uses the closed-form Kalman log-likelihood
        instead of a particle filter. Ground truth for posterior accuracy.

All entry points return a uniform dict matching the structure of
hmc/run_hmc_poc.main's return so the Panel 1 notebook can summarise
them with a single comparison table.
"""

import math
import time
from typing import Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from advanced_particle_filter.tf_filters import (
    TFDifferentiableParticleFilter,
)
from advanced_particle_filter.tf_models.corenflos_lg import (
    simulate_corenflos_lg,
    corenflos_lg_observation_log_prob,
    SIGMA_X2, SIGMA_Y2, STATE_DIM,
)
from advanced_particle_filter.hmc.parameterization_corenflos_lg import (
    unpack_batched, log_prior_batched, warm_start_corenflos_lg, TOTAL_DIM,
)
from advanced_particle_filter.diagnostics.batched_kalman import (
    batched_kalman_log_likelihood,
)


DTYPE = tf.float64


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_target_log_prob_dpf(observations, dpf, rng, n_mc):
    """Target log-prob for DPF-based HMC. Mirrors run_hmc_poc.build_target_log_prob_fn."""

    @tf.function(reduce_retracing=True)
    def target_log_prob_fn(z):
        B_chain = tf.shape(z)[0]
        z_rep = tf.repeat(z, repeats=n_mc, axis=0)
        params = unpack_batched(z_rep)
        result = dpf.filter(params, observations, rng)
        log_ev_flat = result.log_evidence
        log_ev_2d = tf.reshape(log_ev_flat, [B_chain, n_mc])
        log_ev = tf.reduce_mean(log_ev_2d, axis=-1)
        lp = log_prior_batched(z)
        return lp + log_ev

    return target_log_prob_fn


def _build_target_log_prob_kalman(observations, T_static):
    """Target log-prob using exact Kalman log-lik (no particles, no MC)."""
    Sigma_obs = tf.constant(SIGMA_Y2 * np.eye(STATE_DIM), dtype=DTYPE)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def target_log_prob_fn(z):
        params = unpack_batched(z)
        log_lik = batched_kalman_log_likelihood(
            params.mu, params.Phi, params.Sigma_eta_chol,
            Sigma_obs, observations, T_static,
        )
        lp = log_prior_batched(z)
        return lp + log_lik

    return target_log_prob_fn


def compute_rhat(samples: tf.Tensor) -> tf.Tensor:
    """Gelman-Rubin R-hat per parameter dimension.

    Args:
        samples: [n_samples, B_chain, n_params]
    Returns:
        rhat: [n_params]
    """
    return tfp.mcmc.diagnostic.potential_scale_reduction(samples)


def _peak_memory_mb() -> float:
    """Peak GPU memory in MB; returns 0 if unavailable."""
    try:
        info = tf.config.experimental.get_memory_info('GPU:0')
        return info['peak'] / (1024 * 1024)
    except Exception:
        return 0.0


def _reset_peak_memory():
    try:
        tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point: DPF-based HMC
# ---------------------------------------------------------------------------
def run_hmc_corenflos_lg_dpf(
    y_obs: tf.Tensor,
    truth: dict,
    *,
    resampler: str,
    n_particles: int,
    # DPF hyperparameters
    alpha: float = 0.5,
    sinkhorn_eps: float = 0.5,
    sinkhorn_iters: int = 100,
    amortized_ckpt_dir: Optional[str] = None,
    amortized_eps: float = 0.5,
    # MC replicas for variance reduction in the gradient
    n_mc: int = 2,
    # HMC chain hyperparameters
    B_chain: int = 4,
    num_burnin: int = 100,
    num_samples: int = 200,
    num_leapfrog: int = 5,
    initial_step_size: float = 0.01,
    target_accept_prob: float = 0.7,
    warmup_jitter: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run DPF+HMC on the Corenflos LG model. Returns metrics dict."""
    if verbose:
        print(f"\n[Panel 1] DPF-HMC, resampler={resampler!r}, N={n_particles}")

    dpf_rng = tf.random.Generator.from_seed(seed + 1)

    # Build DPF
    dpf_kwargs = dict(
        n_particles=n_particles,
        resampler=resampler,
        alpha=alpha,
        epsilon=sinkhorn_eps,
        sinkhorn_iters=sinkhorn_iters,
        obs_log_prob_fn=corenflos_lg_observation_log_prob,
        dtype=DTYPE,
    )
    if resampler == 'amortized':
        if amortized_ckpt_dir is None:
            raise ValueError("amortized_ckpt_dir required for resampler='amortized'")
        dpf_kwargs['amortized_ckpt_dir'] = amortized_ckpt_dir
        dpf_kwargs['amortized_eps'] = amortized_eps
        dpf_kwargs['amortized_d'] = STATE_DIM
    dpf = TFDifferentiableParticleFilter(**dpf_kwargs)

    target_log_prob_fn = _build_target_log_prob_dpf(
        y_obs, dpf, dpf_rng, n_mc=n_mc,
    )

    z0 = warm_start_corenflos_lg(
        y_obs, B_chain=B_chain, jitter=warmup_jitter, seed=seed,
    )
    if verbose:
        print(f"  Warm start z0 = ")
        print(f"    {z0.numpy().round(3)}")

    # Sanity: gradient finite at warm start.
    z_tv = tf.Variable(z0)
    with tf.GradientTape() as tape:
        lp = target_log_prob_fn(z_tv)
    grad = tape.gradient(lp, z_tv)
    grad_finite = bool(tf.reduce_all(tf.math.is_finite(grad)).numpy())
    if verbose:
        print(f"  log-post warm = {lp.numpy().round(2)}")
        print(f"  grad-norm     = {tf.norm(grad, axis=-1).numpy().round(2)}")
        print(f"  grad finite   = {grad_finite}")
    if not grad_finite:
        raise RuntimeError(
            f"Non-finite gradient at warm start for resampler={resampler!r}.")

    return _run_hmc_and_summarise(
        target_log_prob_fn, z0, truth,
        num_burnin=num_burnin, num_samples=num_samples,
        num_leapfrog=num_leapfrog, initial_step_size=initial_step_size,
        target_accept_prob=target_accept_prob,
        method_label=f"dpf-{resampler}",
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Entry point: Kalman-exact HMC
# ---------------------------------------------------------------------------
def run_hmc_corenflos_lg_kalman(
    y_obs: tf.Tensor,
    truth: dict,
    *,
    B_chain: int = 4,
    num_burnin: int = 100,
    num_samples: int = 200,
    num_leapfrog: int = 5,
    initial_step_size: float = 0.01,
    target_accept_prob: float = 0.7,
    warmup_jitter: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Kalman-exact HMC on the Corenflos LG model."""
    if verbose:
        print(f"\n[Panel 1] Kalman-HMC (exact gradient)")

    T = int(y_obs.shape[0])
    target_log_prob_fn = _build_target_log_prob_kalman(y_obs, T_static=T)

    z0 = warm_start_corenflos_lg(
        y_obs, B_chain=B_chain, jitter=warmup_jitter, seed=seed,
    )
    if verbose:
        print(f"  Warm start z0 = ")
        print(f"    {z0.numpy().round(3)}")

    # Sanity: gradient finite at warm start.
    z_tv = tf.Variable(z0)
    with tf.GradientTape() as tape:
        lp = target_log_prob_fn(z_tv)
    grad = tape.gradient(lp, z_tv)
    grad_finite = bool(tf.reduce_all(tf.math.is_finite(grad)).numpy())
    if verbose:
        print(f"  log-post warm = {lp.numpy().round(2)}")
        print(f"  grad-norm     = {tf.norm(grad, axis=-1).numpy().round(2)}")
        print(f"  grad finite   = {grad_finite}")
    if not grad_finite:
        raise RuntimeError("Non-finite gradient at warm start for Kalman HMC.")

    return _run_hmc_and_summarise(
        target_log_prob_fn, z0, truth,
        num_burnin=num_burnin, num_samples=num_samples,
        num_leapfrog=num_leapfrog, initial_step_size=initial_step_size,
        target_accept_prob=target_accept_prob,
        method_label="kalman",
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Shared HMC runner + summary
# ---------------------------------------------------------------------------
def _run_hmc_and_summarise(
    target_log_prob_fn, z0, truth,
    *, num_burnin, num_samples, num_leapfrog, initial_step_size,
    target_accept_prob, method_label, verbose,
):
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=num_leapfrog,
        step_size=initial_step_size,
    )
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps=int(0.8 * num_burnin),
        target_accept_prob=target_accept_prob,
    )

    _reset_peak_memory()
    t0 = time.time()
    samples_z, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin,
        current_state=z0,
        kernel=adaptive_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )
    elapsed = time.time() - t0
    peak_mb = _peak_memory_mb()

    samples_np = samples_z.numpy()                         # [n_samp, B, 2]
    accept_rate = float(tf.reduce_mean(tf.cast(is_accepted, DTYPE)).numpy())
    rhat_per_param = compute_rhat(samples_z).numpy()
    rhat_max = float(np.max(rhat_per_param))
    rhat_med = float(np.median(rhat_per_param))

    # Posterior summary on theta (= z directly)
    flat = samples_np.reshape(-1, TOTAL_DIM)
    theta_mean = flat.mean(0)
    theta_std = flat.std(0)

    if verbose:
        theta_true = truth['theta'].numpy()
        print(f"  done in {elapsed:.1f}s,  peak mem = {peak_mb:.1f} MB")
        print(f"  accept = {accept_rate:.2%},  R-hat max/med = "
              f"{rhat_max:.3f} / {rhat_med:.3f}")
        print(f"  theta mean = {theta_mean.round(3)},  std = {theta_std.round(3)}")
        print(f"  theta true = {theta_true.round(3)}")

    return dict(
        method=method_label,
        samples_z=samples_np,
        accept_rate=accept_rate,
        rhat=rhat_per_param,
        rhat_max=rhat_max,
        rhat_median=rhat_med,
        theta_mean=theta_mean,
        theta_std=theta_std,
        theta_true=truth['theta'].numpy(),
        elapsed=elapsed,
        peak_memory_mb=peak_mb,
    )


# ---------------------------------------------------------------------------
# Top-level driver: run all four methods on one dataset
# ---------------------------------------------------------------------------
def main(
    *,
    T: int = 150,
    theta_true=(0.5, 0.5),
    data_seed: int = 0,
    # method-specific N
    soft_N: int = 500,
    sinkhorn_N: int = 64,
    amortized_N: int = 500,
    # method hyperparameters (shared where possible)
    sinkhorn_eps: float = 0.5,
    sinkhorn_iters: int = 100,
    amortized_eps: float = 0.5,
    amortized_ckpt_dir: Optional[str] = None,
    # HMC config (uniform across methods)
    B_chain: int = 4,
    num_burnin: int = 100,
    num_samples: int = 200,
    num_leapfrog: int = 5,
    initial_step_size: float = 0.01,
    n_mc: int = 2,
    seed: int = 0,
    # which methods to run; useful for debug
    methods: tuple = ('kalman', 'soft', 'sinkhorn', 'amortized'),
    verbose: bool = True,
) -> dict:
    """Run all panel-1 methods on a single dataset; return outputs keyed by method."""
    if verbose:
        print("=" * 74)
        print("  Panel 1: Corenflos LG SSM, four-method HMC comparison")
        print(f"  T={T}, theta_true={theta_true}, B_chain={B_chain}, "
              f"burnin/samples = {num_burnin}/{num_samples}")
        print("=" * 74)

    # --- Data ---
    if verbose:
        print("\n[Data] Generating Corenflos LG trajectory ...")
    x_true, y_obs, truth = simulate_corenflos_lg(theta_true, T=T, seed=data_seed)

    results = {}

    if 'kalman' in methods:
        results['kalman'] = run_hmc_corenflos_lg_kalman(
            y_obs, truth,
            B_chain=B_chain,
            num_burnin=num_burnin, num_samples=num_samples,
            num_leapfrog=num_leapfrog, initial_step_size=initial_step_size,
            seed=seed, verbose=verbose,
        )
    if 'soft' in methods:
        results['soft'] = run_hmc_corenflos_lg_dpf(
            y_obs, truth, resampler='soft', n_particles=soft_N,
            B_chain=B_chain, n_mc=n_mc,
            num_burnin=num_burnin, num_samples=num_samples,
            num_leapfrog=num_leapfrog, initial_step_size=initial_step_size,
            seed=seed, verbose=verbose,
        )
    if 'sinkhorn' in methods:
        results['sinkhorn'] = run_hmc_corenflos_lg_dpf(
            y_obs, truth, resampler='sinkhorn', n_particles=sinkhorn_N,
            sinkhorn_eps=sinkhorn_eps, sinkhorn_iters=sinkhorn_iters,
            B_chain=B_chain, n_mc=n_mc,
            num_burnin=num_burnin, num_samples=num_samples,
            num_leapfrog=num_leapfrog, initial_step_size=initial_step_size,
            seed=seed, verbose=verbose,
        )
    if 'amortized' in methods:
        results['amortized'] = run_hmc_corenflos_lg_dpf(
            y_obs, truth, resampler='amortized', n_particles=amortized_N,
            amortized_ckpt_dir=amortized_ckpt_dir,
            amortized_eps=amortized_eps,
            B_chain=B_chain, n_mc=n_mc,
            num_burnin=num_burnin, num_samples=num_samples,
            num_leapfrog=num_leapfrog, initial_step_size=initial_step_size,
            seed=seed, verbose=verbose,
        )

    return dict(
        results=results,
        truth=truth,
        y_obs=y_obs.numpy(),
        x_true=x_true.numpy(),
    )
