"""
Driver for the Find-Mode-Then-Sample pipeline.

Runs all three phases end-to-end and prints a summary comparing to ground
truth (on the 2-asset contagion scenario from hmc/run_hmc_poc.py).

Usage:
    from advanced_particle_filter.mle.run_mle_laplace import main
    results = main(n_particles=1000, run_phase3=True)
"""

import time
import numpy as np
import tensorflow as tf

from ..tf_filters import TFDifferentiableParticleFilter
from ..hmc.parameterization import (
    unpack_batched, TOTAL_DIM, STATE_DIM,
    MU_START, MU_END, PHI_START, PHI_END, SIGMA_START, SIGMA_END,
)
from ..hmc.run_hmc_poc import make_contagion_data

from .adam_mle import run_adam_mle
from .laplace import compute_laplace_covariance
from .whitened_hmc import run_whitened_hmc
from .preconditioned_hmc import run_preconditioned_hmc


DTYPE = tf.float64


def _print_theta_at_mode(z_hat, truth):
    """Convert z_hat to (mu, Phi, Sigma_eta) and compare to truth."""
    z_tf = tf.cast(tf.reshape(z_hat, [1, TOTAL_DIM]), DTYPE)
    params = unpack_batched(z_tf)
    mu_hat = params.mu[0].numpy()
    Phi_hat = params.Phi[0].numpy()
    L_hat = params.Sigma_eta_chol[0].numpy()
    Sigma_hat = L_hat @ L_hat.T

    print(f"  mu_hat    = {mu_hat.round(3)}")
    print(f"  mu_true   = {truth['mu'].numpy()}")
    print(f"  Phi_hat   =\n{Phi_hat.round(3)}")
    print(f"  Phi_true  =\n{truth['Phi'].numpy()}")
    print(f"  Sigma_hat =\n{Sigma_hat.round(4)}")
    print(f"  Sigma_true=\n{truth['Sigma_eta'].numpy().round(4)}")


def _print_posterior_summary(samples_z, truth, label="HMC"):
    """Print posterior summaries from samples in original z-coords."""
    flat = samples_z.reshape(-1, TOTAL_DIM)
    flat_tf = tf.constant(flat, dtype=DTYPE)
    p = unpack_batched(flat_tf)
    mus = p.mu.numpy()
    Phis = p.Phi.numpy()
    L = p.Sigma_eta_chol.numpy()
    Sigmas = np.einsum('mij,mkj->mik', L, L)

    print(f"\n  {label} mu:    mean={mus.mean(0).round(3)} "
          f"std={mus.std(0).round(3)}  true={truth['mu'].numpy()}")
    print(f"  {label} Phi mean:\n{Phis.mean(0).round(3)}")
    print(f"  {label} Phi std:\n{Phis.std(0).round(3)}")
    print(f"  {label} Sigma_eta mean:\n{Sigmas.mean(0).round(4)}")


def main(
    # data
    T: int = 75,
    data_seed: int = 0,
    # DPF
    n_particles: int = 1000,
    alpha: float = 0.5,
    # Phase 1 — tuned for SVSSM DPF gradient SNR
    B_restart: int = 4,
    adam_steps: int = 3000,      # longer horizon; Adam w/ noisy grads needs it
    adam_lr: float = 0.005,
    adam_n_mc: int = 16,         # 16 MC replicas: variance reduces 4x vs n_mc=4,
                                  # signal/noise ~2x. Effective batch 64 at B_restart=4.
    adam_jitter: float = 0.1,
    # Phase 2
    laplace_eps: float = 0.05,
    laplace_n_seeds: int = 10,
    laplace_chunk_size: int = 8,
    laplace_kappa_target: float = 10.0,
    # Phase 3 (optional)
    run_phase3: bool = True,
    phase3_method: str = 'preconditioned',  # 'preconditioned' or 'whitened'
    hmc_B_chain: int = 4,
    hmc_burnin: int = 50,
    hmc_samples: int = 200,
    hmc_leapfrog: int = 10,
    hmc_step: float = 0.3,     # lower default: correct for preconditioned;
                                # whitened path will override internally
    hmc_n_mc: int = 2,
    # MAP: include prior everywhere
    include_prior: bool = True,
    verbose: bool = True,
):
    t_total = time.time()

    print("=" * 74)
    print("  Find-Mode-Then-Sample pipeline  (SVSSM 2-asset contagion)")
    print("=" * 74)

    # --- Data ---
    print("\n[Data] Generating synthetic observations ...")
    h_true, y_obs, truth = make_contagion_data(T=T, seed=data_seed)
    print(f"  T={T}, state_dim={STATE_DIM}")
    print(f"  mu_true = {truth['mu'].numpy()}")
    print(f"  Phi_true=\n{truth['Phi'].numpy()}")

    # --- DPF setup: soft resampler at the target N ---
    dpf_rng = tf.random.Generator.from_seed(data_seed + 1)
    dpf = TFDifferentiableParticleFilter(
        n_particles=n_particles,
        resampler="soft",
        alpha=alpha,
        dtype=DTYPE,
    )
    print(f"  DPF: soft resampler, N={n_particles}, alpha={alpha}")

    # ================================================================
    # Phase 1: Adam MAP
    # ================================================================
    print("\n" + "-" * 74)
    print("  Phase 1: MAP via Adam + DPF-soft")
    print("-" * 74)

    adam_result = run_adam_mle(
        observations=y_obs,
        dpf=dpf,
        rng=dpf_rng,
        B_restart=B_restart,
        jitter=adam_jitter,
        n_steps=adam_steps,
        learning_rate=adam_lr,
        n_mc=adam_n_mc,
        verbose=verbose,
    )
    z_hat = adam_result.z_hat

    print(f"\n[Phase 1 summary] converged in {adam_result.elapsed:.1f}s")
    print(f"  best restart index: {adam_result.best_restart}")
    print(f"  best log_posterior: "
          f"{float(adam_result.final_log_post[adam_result.best_restart].numpy()):+.3f}")
    print(f"\n  Parameters at MAP mode:")
    _print_theta_at_mode(z_hat, truth)

    # ================================================================
    # Phase 2: Laplace
    # ================================================================
    print("\n" + "-" * 74)
    print("  Phase 2: Laplace covariance via FD Hessian")
    print("-" * 74)

    laplace_result = compute_laplace_covariance(
        observations=y_obs,
        dpf=dpf,
        z_hat=z_hat,
        eps=laplace_eps,
        n_seeds=laplace_n_seeds,
        include_prior=include_prior,
        chunk_size=laplace_chunk_size,
        kappa_target=laplace_kappa_target,
        verbose=verbose,
    )

    Sigma = laplace_result.Sigma_laplace.numpy()
    L_chol = laplace_result.L_chol.numpy()
    std_marg = np.sqrt(np.diag(Sigma))
    print(f"\n[Phase 2 summary] Hessian computed in "
          f"{laplace_result.elapsed:.1f}s")
    print(f"  eigvals clipped:       {laplace_result.n_eigvals_clipped}/"
          f"{TOTAL_DIM} (kappa_target={laplace_result.kappa_target:.1f})")
    print(f"  log_posterior at mode: "
          f"{laplace_result.logpost_at_mode:+.3f}")
    print(f"  marginal std (Laplace):  {std_marg.round(3)}")

    # ================================================================
    # Phase 3: HMC (optional)
    # ================================================================
    whitened_result = None
    if run_phase3:
        print("\n" + "-" * 74)
        if phase3_method == 'preconditioned':
            print("  Phase 3: Preconditioned HMC (|H|-mass-matrix, no whitening)")
        else:
            print("  Phase 3: Whitened HMC around the mode")
        print("-" * 74)

        hmc_rng = tf.random.Generator.from_seed(data_seed + 10000)

        if phase3_method == 'preconditioned':
            whitened_result = run_preconditioned_hmc(
                observations=y_obs,
                dpf=dpf,
                rng=hmc_rng,
                z_hat=z_hat,
                H=laplace_result.H,           # raw Hessian, indefinite OK
                B_chain=hmc_B_chain,
                num_burnin=hmc_burnin,
                num_samples=hmc_samples,
                num_leapfrog=hmc_leapfrog,
                initial_step_size=hmc_step,
                n_mc=hmc_n_mc,
                include_prior=include_prior,
                verbose=verbose,
            )
        elif phase3_method == 'whitened':
            whitened_result = run_whitened_hmc(
                observations=y_obs,
                dpf=dpf,
                rng=hmc_rng,
                z_hat=z_hat,
                L_chol=tf.constant(L_chol, dtype=DTYPE),
                B_chain=hmc_B_chain,
                num_burnin=hmc_burnin,
                num_samples=hmc_samples,
                num_leapfrog=hmc_leapfrog,
                initial_step_size=max(hmc_step, 0.5),  # whitened wants larger
                n_mc=hmc_n_mc,
                include_prior=include_prior,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"phase3_method must be 'preconditioned' or 'whitened', "
                f"got {phase3_method!r}"
            )

        print(f"\n[Phase 3 summary] HMC done in {whitened_result.elapsed:.1f}s")
        print(f"  method      = {phase3_method}")
        print(f"  accept_rate = {whitened_result.accept_rate:.2%}")
        _print_posterior_summary(
            whitened_result.samples_z, truth, label="HMC-post"
        )

        # Compare HMC marginal std to Laplace marginal std (diagnostic only;
        # for 'preconditioned' method we don't claim Laplace Σ is accurate)
        hmc_std = whitened_result.samples_z.reshape(-1, TOTAL_DIM).std(axis=0)
        print(f"\n  Laplace marginal std: {std_marg.round(3)}")
        print(f"  HMC     marginal std: {hmc_std.round(3)}")
        ratio = hmc_std / np.maximum(std_marg, 1e-12)
        print(f"  ratio HMC/Laplace:    {ratio.round(2)}")
        if np.max(np.abs(ratio - 1)) < 0.3:
            print("  -> Laplace was within 30% of HMC marginals: "
                  "Gaussian approx was adequate.")

    # --- Overall summary ---
    total_time = time.time() - t_total
    print("\n" + "=" * 74)
    print(f"  Pipeline complete in {total_time:.1f}s total.")
    print("=" * 74)

    return dict(
        adam=adam_result,
        laplace=laplace_result,
        whitened=whitened_result,
        truth=truth,
        y_obs=y_obs.numpy(),
        total_elapsed=total_time,
    )


if __name__ == "__main__":
    main()
