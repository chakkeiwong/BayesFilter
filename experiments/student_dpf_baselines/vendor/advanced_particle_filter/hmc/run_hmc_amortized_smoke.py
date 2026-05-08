"""
Smoke test: HMC + DPF + SVSSM with the amortized OT resampler.

Mirrors hmc/run_hmc_poc.py but uses the trained CouplingOperator as the
DPF resampler. Run with the same configuration to compare:
  - timing (HMC wall-clock)
  - posterior summary (mu, Phi, Sigma_eta means and stds vs ground truth)
  - convergence diagnostics (R-hat, accept rate)
  - gradient sanity (norm at warm start, finiteness)

against the soft and sinkhorn baselines.

Usage (Colab):
    from advanced_particle_filter.hmc.run_hmc_amortized_smoke import main_amortized
    out = main_amortized(
        ckpt_dir='/content/drive/MyDrive/mgn_ot_operator/checkpoints_option_b',
        amortized_eps=0.5,
    )

For comparison, run with the original baselines too:
    from advanced_particle_filter.hmc.run_hmc_poc import main
    out_soft     = main(resampler='soft')
    out_sinkhorn = main(resampler='sinkhorn')

The three outputs share the same dict structure so they can be diffed
directly.
"""

import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .tf_models.svssm import simulate_svssm, SVSSMParams
from .tf_filters import TFDifferentiableParticleFilter
from .hmc.parameterization import (
    unpack_batched, log_prior_batched, TOTAL_DIM, STATE_DIM,
    MU_START, MU_END, PHI_START, PHI_END, SIGMA_START, SIGMA_END,
)
from .hmc.run_hmc_poc import (
    make_contagion_data, build_target_log_prob_fn, warm_start, compute_rhat,
)


DTYPE = tf.float64


def main_amortized(
    ckpt_dir: str = None,
    amortized_eps: float = 0.5,
    T: int = 75,
    B_chain: int = 4,
    n_mc: int = 2,
    n_particles: int = 100,
    num_burnin: int = 150,
    num_samples: int = 200,
    num_leapfrog: int = 5,
    initial_step_size: float = 0.01,
    jitter: float = 0.02,
    seed: int = 0,
):
    """Run HMC+DPF+SVSSM with the amortized OT resampler.

    Args:
        ckpt_dir: Path to the trained operator's checkpoint root (the
                  directory containing best/ and latest/ subdirs). If None
                  (default), uses the bundled checkpoint at
                  dpf_pretrained/mgn_ot_operator/checkpoints_option_b/.
        amortized_eps: Eps fed to the operator's eps-conditioning.
                  Default 0.5 (Corenflos default).
        Other args: same as hmc.run_hmc_poc.main.
    """
    print("=" * 74)
    print("  SVSSM + DPF + HMC  PoC  (2-asset risk contagion, AMORTIZED OT)")
    print("=" * 74)

    print(f"\nConfig:")
    print(f"  T = {T},  state dim = {STATE_DIM}")
    print(f"  B_chain = {B_chain}, n_mc = {n_mc}, batch B = {B_chain * n_mc}")
    print(f"  n_particles = {n_particles}")
    print(f"  resampler = 'amortized'")
    print(f"  amortized_eps = {amortized_eps}")
    print(f"  ckpt_dir = {ckpt_dir}")
    print(f"  HMC: {num_burnin} burnin, {num_samples} samples, "
          f"{num_leapfrog} leapfrog, step={initial_step_size}")

    # --- Data ---
    print("\n[1/4] Generating synthetic data ...")
    h_true, y_obs, truth = make_contagion_data(T=T, seed=seed)
    print(f"  mu_true             = {truth['mu'].numpy()}")
    print(f"  Phi_true eigenvalues (for info): "
          f"{np.linalg.eigvals(truth['Phi'].numpy())}")
    print(f"  Sigma_eta_true diag = "
          f"{np.diag(truth['Sigma_eta'].numpy())}")

    # --- Build target log-prob ---
    print("\n[2/4] Building target_log_prob_fn (DPF + prior + amortized resampler) ...")
    dpf_rng = tf.random.Generator.from_seed(seed + 1)
    dpf = TFDifferentiableParticleFilter(
        n_particles=n_particles,
        resampler='amortized',
        amortized_ckpt_dir=ckpt_dir,
        amortized_eps=amortized_eps,
        amortized_d=STATE_DIM,
        dtype=DTYPE,
    )
    target_log_prob_fn = build_target_log_prob_fn(
        y_obs, dpf, dpf_rng, n_mc=n_mc,
    )

    # --- Warm start and sanity check ---
    z0 = warm_start(y_obs, B_chain=B_chain, jitter=jitter)
    print(f"\n  Warm start z0 shape = {z0.shape}")
    params0 = unpack_batched(z0)
    print(f"  Chain 0 warm start:")
    print(f"    mu0         = {params0.mu[0].numpy()}")
    print(f"    Phi0        = {params0.Phi[0].numpy().round(3)}")
    S0 = params0.Sigma_eta_chol[0] @ tf.transpose(params0.Sigma_eta_chol[0])
    print(f"    Sigma_eta0  = {S0.numpy().round(3)}")

    print(f"\n  Sanity-check gradient at warm start ...")
    z_tv = tf.Variable(z0)
    with tf.GradientTape() as tape:
        lp = target_log_prob_fn(z_tv)
    grad = tape.gradient(lp, z_tv)
    print(f"    log-posterior per chain = {lp.numpy().round(2)}")
    print(f"    gradient norm per chain = "
          f"{tf.norm(grad, axis=-1).numpy().round(2)}")
    grad_finite = bool(tf.reduce_all(tf.math.is_finite(grad)).numpy())
    print(f"    gradient all finite     = {grad_finite}")
    if not grad_finite:
        n_nonfin = int(tf.reduce_sum(tf.cast(
            tf.math.logical_not(tf.math.is_finite(grad)), tf.int32)).numpy())
        print(f"    WARNING: {n_nonfin} non-finite grad entries.")
        print(f"    (This often indicates the amortized resampler is producing "
              f"outputs that the SVSSM observation model can't handle, e.g., "
              f"particles in regions of zero observation probability.)")

    # --- HMC ---
    print("\n[3/4] Running HMC ...")
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=num_leapfrog,
        step_size=initial_step_size,
    )
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps=int(0.8 * num_burnin),
        target_accept_prob=0.70,
    )

    t0 = time.time()
    samples_z, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin,
        current_state=z0,
        kernel=adaptive_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )
    elapsed = time.time() - t0

    samples_z_np = samples_z.numpy()
    accept_rate = float(
        tf.reduce_mean(tf.cast(is_accepted, DTYPE)).numpy()
    )
    print(f"  HMC done in {elapsed:.1f}s")
    print(f"  overall accept rate = {accept_rate:.2%}")

    # --- Diagnostics and posterior summary ---
    print("\n[4/4] Posterior summary ...")

    if B_chain >= 2:
        rhat = compute_rhat(samples_z).numpy()
        print(f"\n  R-hat per unconstrained param (target < 1.1):")
        print(f"    mu:        {rhat[MU_START:MU_END].round(3)}")
        print(f"    Phi (flat):{rhat[PHI_START:PHI_END].round(3)}")
        print(f"    Sigma:     {rhat[SIGMA_START:SIGMA_END].round(3)}")

    flat = samples_z_np.reshape(-1, TOTAL_DIM)
    flat_tf = tf.constant(flat)
    params_flat = unpack_batched(flat_tf)
    mus = params_flat.mu.numpy()
    Phis = params_flat.Phi.numpy()
    Lchols = params_flat.Sigma_eta_chol.numpy()
    Sigmas = np.einsum('mij,mkj->mik', Lchols, Lchols)

    print(f"\n  mu posterior:")
    print(f"    mean = {mus.mean(0).round(3)}     "
          f"true = {truth['mu'].numpy()}")
    print(f"    std  = {mus.std(0).round(3)}")

    print(f"\n  Phi posterior mean:")
    print(Phis.mean(0).round(3))
    print(f"  Phi true:")
    print(truth['Phi'].numpy())

    print(f"\n  Sigma_eta posterior mean:")
    print(Sigmas.mean(0).round(4))
    print(f"  Sigma_eta true:")
    print(truth['Sigma_eta'].numpy().round(4))

    print("\n" + "=" * 74)
    print("  Amortized PoC complete.")
    print("=" * 74)

    return dict(
        samples_z=samples_z_np,
        accept_rate=accept_rate,
        truth=truth,
        elapsed=elapsed,
        mus=mus, Phis=Phis, Sigmas=Sigmas,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m advanced_particle_filter.hmc.run_hmc_amortized_smoke "
              "<ckpt_dir>")
        sys.exit(1)
    main_amortized(ckpt_dir=sys.argv[1])
