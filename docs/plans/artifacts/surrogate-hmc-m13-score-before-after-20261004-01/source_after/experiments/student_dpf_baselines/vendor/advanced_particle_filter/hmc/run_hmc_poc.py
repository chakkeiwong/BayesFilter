"""
Run HMC to calibrate the SVSSM using the differentiable particle filter.

Two batch axes are fully decoupled:
    B_chain = number of HMC chains (each explores its own theta trajectory)
    B_mc    = number of MC replicas per gradient evaluation (variance reduction)

Inside target_log_prob_fn(z):
    z has shape [B_chain, 9].
    We replicate each chain's params B_mc times, flatten to B = B_chain*B_mc,
    run the DPF once with batch B, then reshape and average over the MC axis.
    Result: [B_chain] log-posterior.

The outer HMC sees only the B_chain dimension.
"""

import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from advanced_particle_filter.tf_models.svssm import (
    simulate_svssm, SVSSMParams,
)
from advanced_particle_filter.tf_filters import (
    TFDifferentiableParticleFilter,
)
from advanced_particle_filter.hmc.parameterization import (
    unpack_batched, log_prior_batched, TOTAL_DIM, STATE_DIM,
    MU_START, MU_END, PHI_START, PHI_END, SIGMA_START, SIGMA_END,
)


DTYPE = tf.float64


# ============================================================================
# Data: 2-asset risk contagion scenario
# ============================================================================

def make_contagion_data(T: int, seed: int = 0):
    """
    Two-asset SVSSM with asymmetric spillover (asset 2 -> asset 1).

    Returns:
        h_true:   [T, 2] latent log-variance trajectory
        y_obs:    [T, 2] observations
        truth:    dict of ground-truth tensors
    """
    mu_true = tf.constant([-1.0, 0.5], dtype=DTYPE)
    Phi_true = tf.constant([[0.85, 0.12],
                            [0.02, 0.90]], dtype=DTYPE)

    # Sigma_eta diag std = (0.15, 0.4), corr = 0.3
    std_eta = np.array([0.15, 0.4])
    rho = 0.3
    Sigma_eta_true = np.array([
        [std_eta[0]**2,                     rho * std_eta[0] * std_eta[1]],
        [rho * std_eta[0] * std_eta[1],    std_eta[1]**2],
    ])
    Lchol_np = np.linalg.cholesky(Sigma_eta_true)
    Sigma_eta_chol_true = tf.constant(Lchol_np, dtype=DTYPE)

    rng = tf.random.Generator.from_seed(seed)
    h_true, y_obs = simulate_svssm(
        mu_true, Phi_true, Sigma_eta_chol_true, T, rng
    )

    return h_true, y_obs, dict(
        mu=mu_true, Phi=Phi_true, Sigma_eta_chol=Sigma_eta_chol_true,
        Sigma_eta=tf.constant(Sigma_eta_true, dtype=DTYPE),
    )


# ============================================================================
# Target log-posterior with multi-chain + MC replication
# ============================================================================

def build_target_log_prob_fn(
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    rng: tf.random.Generator,
    n_mc: int = 4,
):
    """
    Construct target_log_prob_fn(z) for TFP HMC.

    Args:
        observations: [T, d]
        dpf:          configured TFDifferentiableParticleFilter
        rng:          tf.random.Generator (used inside PF sampling)
        n_mc:         MC replicas per gradient evaluation (variance reduction)

    Returns:
        target_log_prob_fn: callable z:[B_chain, 9] -> [B_chain] log-posterior
    """

    @tf.function(reduce_retracing=True)
    def target_log_prob_fn(z):
        # z: [B_chain, 9]
        B_chain = tf.shape(z)[0]

        # Replicate each chain's z n_mc times:
        # z_rep shape becomes [B_chain * n_mc, 9], ordered as
        # [chain0_mc0, chain0_mc1, ..., chain0_mc(n_mc-1),
        #  chain1_mc0, ...] because tf.repeat with axis=0 interleaves per row.
        z_rep = tf.repeat(z, repeats=n_mc, axis=0)  # [B_chain*n_mc, 9]

        # Unpack batched parameters
        params = unpack_batched(z_rep)

        # Run DPF — returns log_evidence of shape [B_chain*n_mc]
        result = dpf.filter(params, observations, rng)
        log_ev_flat = result.log_evidence  # [B_chain * n_mc]

        # Reshape to [B_chain, n_mc] and average over MC axis.
        # Averaging log-evidences is a biased estimator of log E[p(y|theta)],
        # but for gradient purposes it is variance-reduced and unbiased
        # for the gradient up to the bias of the log-of-mean approximation.
        # See Del Moral (2004) and Fearnhead/Clifford (2003) for discussion.
        log_ev_2d = tf.reshape(log_ev_flat, [B_chain, n_mc])
        log_ev = tf.reduce_mean(log_ev_2d, axis=-1)  # [B_chain]

        # Prior per chain (operates on unreplicated z)
        lp = log_prior_batched(z)  # [B_chain]

        return lp + log_ev

    return target_log_prob_fn


# ============================================================================
# Warm start from data moments
# ============================================================================

def warm_start(y_obs: tf.Tensor, B_chain: int, jitter: float = 0.02):
    """
    Data-driven warm start for B_chain chains, respecting the mu ordering
    constraint: mu_1 = z[0], mu_2 = z[0] + exp(z[1]).

    Warm-start strategy:
      - mu: from log(Var(y)) per asset, sorted so mu_1 < mu_2.
        Then z[0] = mu_1, z[1] = log(mu_2 - mu_1).
      - Phi: diagonal at 0.85 (typical persistence), off-diagonal at 0.
      - Sigma_eta: log-diag at log(0.2) (moderate innovation), L_21 = 0.

    Args:
        y_obs: [T, d]
        B_chain: number of chains
        jitter: per-chain perturbation scale (default 0.02 — small to
                avoid gradient cliffs observed in earlier experiments)

    Returns:
        z0: [B_chain, 9]
    """
    dtype = y_obs.dtype

    # Approximate mu from marginal variance of y
    var_y = tf.math.reduce_variance(y_obs, axis=0)            # [d]
    mu_approx = tf.math.log(var_y + 1e-6)                     # [d]

    # Sort to ensure mu_1 < mu_2
    mu_sorted = tf.sort(mu_approx)                             # [d]
    mu_1 = mu_sorted[0]
    mu_2 = mu_sorted[1]

    # Map to unconstrained: z[0] = mu_1, z[1] = log(mu_2 - mu_1)
    # Guard against mu_2 == mu_1 with a minimum gap
    gap = tf.maximum(mu_2 - mu_1, tf.constant(0.1, dtype=dtype))
    z_mu_0 = mu_1
    z_mu_1 = tf.math.log(gap)

    # Phi: diagonal at 0.85, off-diagonal at 0
    z_phi = tf.constant([0.85, 0.0, 0.0, 0.85], dtype=dtype)

    # Sigma_eta Cholesky: log(0.2) on diagonal, 0 off-diagonal
    log_02 = tf.math.log(tf.constant(0.2, dtype=dtype))
    z_sigma = tf.stack([log_02, log_02, tf.constant(0.0, dtype=dtype)])

    z_single = tf.concat([
        tf.stack([z_mu_0, z_mu_1]),
        z_phi,
        z_sigma,
    ], axis=0)  # [9]

    # Broadcast + small jitter per chain
    z0 = tf.tile(z_single[tf.newaxis, :], [B_chain, 1])       # [B_chain, 9]
    rng = tf.random.Generator.from_seed(1234)
    z0 = z0 + jitter * rng.normal(z0.shape, dtype=dtype)
    return z0


# ============================================================================
# R-hat convergence diagnostic
# ============================================================================

def compute_rhat(samples: tf.Tensor) -> tf.Tensor:
    """
    Potential scale reduction factor R-hat (Gelman-Rubin).

    Args:
        samples: [num_samples, B_chain, num_params]

    Returns:
        rhat: [num_params]
    """
    # TFP provides this directly
    return tfp.mcmc.diagnostic.potential_scale_reduction(samples)


# ============================================================================
# Main
# ============================================================================

def main(
    T: int = 75,
    B_chain: int = 4,
    n_mc: int = 2,
    n_particles: int = 100,
    resampler: str = "soft",
    alpha: float = 0.5,
    epsilon: float = 0.1,
    sinkhorn_iters: int = 30,
    num_burnin: int = 150,
    num_samples: int = 200,
    num_leapfrog: int = 5,
    initial_step_size: float = 0.01,
    jitter: float = 0.02,
    seed: int = 0,
):
    print("=" * 74)
    print("  SVSSM + DPF + HMC  PoC  (2-asset risk contagion)")
    print("=" * 74)

    print(f"\nConfig:")
    print(f"  T = {T},  state dim = {STATE_DIM}")
    print(f"  B_chain = {B_chain}, n_mc = {n_mc}, batch B = {B_chain * n_mc}")
    print(f"  n_particles = {n_particles}")
    print(f"  resampler = {resampler!r}", end="")
    if resampler == "soft":
        print(f" (alpha = {alpha})")
    else:
        print(f" (epsilon = {epsilon}, sinkhorn_iters = {sinkhorn_iters})")
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
    print("\n[2/4] Building target_log_prob_fn (DPF + prior) ...")
    dpf_rng = tf.random.Generator.from_seed(seed + 1)
    dpf = TFDifferentiableParticleFilter(
        n_particles=n_particles,
        resampler=resampler,
        alpha=alpha,
        epsilon=epsilon,
        sinkhorn_iters=sinkhorn_iters,
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
    print(f"    gradient all finite     = "
          f"{bool(tf.reduce_all(tf.math.is_finite(grad)).numpy())}")

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
    # NOTE: we do NOT wrap sample_chain in tf.function here because the
    # inner target_log_prob_fn is already a @tf.function. Wrapping again
    # would cause redundant retracing and, with the long inner time loop,
    # bloat the outer graph. Eager outer loop + compiled inner step is the
    # sweet spot for peak memory.
    samples_z, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin,
        current_state=z0,
        kernel=adaptive_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )
    elapsed = time.time() - t0

    samples_z_np = samples_z.numpy()  # [num_samples, B_chain, 9]
    accept_rate = float(
        tf.reduce_mean(tf.cast(is_accepted, DTYPE)).numpy()
    )
    print(f"  HMC done in {elapsed:.1f}s")
    print(f"  overall accept rate = {accept_rate:.2%}")

    # --- Diagnostics and posterior summary ---
    print("\n[4/4] Posterior summary ...")

    # R-hat across chains (needs at least 2 chains)
    if B_chain >= 2:
        rhat = compute_rhat(samples_z).numpy()
        print(f"\n  R-hat per unconstrained param (target < 1.1):")
        print(f"    mu:        {rhat[MU_START:MU_END].round(3)}")
        print(f"    Phi (flat):{rhat[PHI_START:PHI_END].round(3)}")
        print(f"    Sigma:     {rhat[SIGMA_START:SIGMA_END].round(3)}")

    # Flatten chain × sample for posterior summaries
    flat = samples_z_np.reshape(-1, TOTAL_DIM)  # [num_samples*B_chain, 9]
    # Unpack each draw to (mu, Phi, Sigma_eta)
    flat_tf = tf.constant(flat)
    params_flat = unpack_batched(flat_tf)
    mus = params_flat.mu.numpy()               # [M, 2]
    Phis = params_flat.Phi.numpy()             # [M, 2, 2]
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
    print("  PoC complete.")
    print("=" * 74)

    return dict(
        samples_z=samples_z_np,
        accept_rate=accept_rate,
        truth=truth,
        elapsed=elapsed,
        mus=mus, Phis=Phis, Sigmas=Sigmas,
    )


if __name__ == "__main__":
    main()
